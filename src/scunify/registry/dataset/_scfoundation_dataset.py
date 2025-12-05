import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


class ScFoundationDataset(Dataset):
    def __init__(self, adata, config):
        self.adata = adata
        self.config = config
        self.pad_token_id = config.model_param[config.inference["version"]]["mae_autobin"]["pad_token_id"]

        # DataLoader 배치 크기 기준 분기 (기본 1)
        self.bs = int(config.inference.get("batch_size", 1))

        # gene_list
        self.gene_list = (
            pd.read_csv(config.resources["gene_list"], sep="\t", header=None, usecols=[0])
            .iloc[:, 0]
            .astype(str)
            .to_numpy()
        )

        self.N = int(adata.n_obs)
        var = np.asarray(adata.var_names).astype(str)
        v2i = {g: i for i, g in enumerate(var)}

        present = np.isin(self.gene_list, var)
        self.tgt_idx = np.nonzero(present)[0]
        self.src_idx = np.array([v2i[g] for g in self.gene_list[present]], dtype=np.int64)

        self.G = int(len(self.gene_list))

        # 전처리 옵션 (원본 설정 그대로)
        pp = config.preprocessing
        self.normalize_total = pp.get("normalize_total", None)
        self.log1p = bool(pp.get("log1p", False))

        # tgthighres (원본과 동일)
        tg = config.inference["tgthighres"]
        self.tg_mode = tg[0]
        self.tg_val = float(tg[1:])

        # 전역 K (배치>1에서 사용). 원본 로직 그대로: (feats>0).sum()의 전 행 최대값.
        if self.bs > 1:
            Kmax = 0
            X = self.adata.X
            for i in range(self.N):
                # 원본 행 로드
                rowX = X[i, :]
                if sp.issparse(rowX):
                    row = rowX.toarray().ravel().astype(np.float32, copy=False)
                else:
                    row = np.asarray(rowX).ravel().astype(np.float32, copy=False)

                # gene_list 순서로 맵핑 (없는 유전자는 0)
                x = np.zeros(self.G, dtype=np.float32)
                if self.src_idx.size:
                    x[self.tgt_idx] = row[self.src_idx]

                # 전처리 (원본 방식)
                if self.normalize_total is not None:
                    s = float(x.sum())
                    if s > 0:
                        x = x / s * float(self.normalize_total)
                if self.log1p:
                    np.log1p(x, out=x)

                # 보조 특성 2개 (원본과 동일: log10(0) 가능 → -inf 허용)
                tc = float(row.sum())
                if self.tg_mode == "f":
                    resolution = np.log10(tc * self.tg_val)
                elif self.tg_mode == "a":
                    resolution = np.log10(tc) + self.tg_val
                else:  # 't'
                    resolution = self.tg_val
                logtc = np.log10(tc)

                feats = np.concatenate([x, np.array([resolution, logtc], dtype=np.float32)], axis=0)
                k = int((feats > 0).sum())
                if k > Kmax:
                    Kmax = k
            self.K = int(Kmax) if Kmax > 0 else 1
        else:
            self.K = None  # bs==1에서는 per-row 동적 K 사용

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # 행 로드
        X = self.adata.X
        rowX = X[idx, :]
        if sp.issparse(rowX):
            row = rowX.toarray().ravel()
        else:
            row = np.asarray(rowX).ravel()
        row = row.astype(np.float32, copy=False)

        # gene_list 재배치
        x = np.zeros(self.G, dtype=np.float32)
        if self.src_idx.size:
            x[self.tgt_idx] = row[self.src_idx]

        # 전처리
        if self.normalize_total is not None:
            s = float(x.sum())
            if s > 0:
                x = x / s * float(self.normalize_total)
        if self.log1p:
            np.log1p(x, out=x)

        # 보조 특성 2개 (그대로)
        tc = float(row.sum())
        if self.tg_mode == "f":
            resolution = np.log10(tc * self.tg_val)
        elif self.tg_mode == "a":
            resolution = np.log10(tc) + self.tg_val
        else:  # 't'
            resolution = self.tg_val
        logtc = np.log10(tc)

        feats = np.concatenate([x, np.array([resolution, logtc], dtype=np.float32)], axis=0)  # (F=G+2,)
        data = torch.from_numpy(feats).unsqueeze(0)  # (1, F)
        labels = (data > 0).to(torch.float32)  # (1, F)
        F = data.shape[1]

        # bs==1 → 동적 K (셀별), bs>1 → 전역 K
        if self.bs == 1:
            Kvar = int(labels.sum().item())
            if Kvar < 1:
                Kvar = 1
        else:
            Kvar = self.K

        # gatherData 원본과 동일
        fake_data = torch.full((1, Kvar), self.pad_token_id, dtype=data.dtype, device=data.device)
        data_pad = torch.hstack([data, fake_data])

        fake_label = torch.ones((1, Kvar), dtype=labels.dtype, device=labels.device)
        lbl = labels.clone()
        lbl[lbl == 0] = -float("inf")

        tmp = torch.tensor([(i + 1) * 20000 for i in range(F, 0, -1)], dtype=lbl.dtype, device=lbl.device)
        lbl = torch.hstack([lbl + tmp, fake_label])

        topk = torch.topk(lbl, Kvar, dim=1).indices
        values = torch.gather(data_pad, 1, topk)
        pad = values == self.pad_token_id

        gene_ids = torch.arange(F, device=data.device, dtype=torch.long).unsqueeze(0)
        fake_ids = torch.full((1, Kvar), self.pad_token_id, device=data.device, dtype=torch.long)
        pos_src = torch.hstack([gene_ids, fake_ids])
        pos = torch.gather(pos_src, 1, topk).long()

        return (
            values.squeeze(0),  # (Kvar,)
            pad.squeeze(0),  # (Kvar,) bool
            pos.squeeze(0),  # (Kvar,) long
            torch.tensor(idx, dtype=torch.long),
        )
