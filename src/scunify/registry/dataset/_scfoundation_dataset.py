"""
ScFoundation Dataset - 원논문 get_embedding.py를 정확히 따라 구현
Foundations/scFoundation/model/get_embedding.py Line 140-190 참조
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


class ScFoundationDataset(Dataset):
    """
    Foundations get_embedding.py의 cell embedding 생성 로직을 정확히 재현.
    
    주요 특징:
    - batch_size=1만 지원 (원논문과 동일)
    - 원논문의 Line 140-190 로직 그대로 구현
    - main_gene_selection 로직 포함 (19264 genes로 변환)
    """
    
    def __init__(self, adata, config):
        self.config = config
        self.pad_token_id = config.model_param[config.inference["version"]]["mae_autobin"]["pad_token_id"]
        self.collator = None
        self.sampler = None
        # batch_size=1 강제 (원논문과 동일)
        if config.inference.get("batch_size", 1) != 1:
            raise ValueError("ScFoundationDataset only supports batch_size=1 (same as original paper)")
        
        # Gene list 로드
        gene_list_df = pd.read_csv(config.resources["gene_list"], sep='\t', header=0)
        self.gene_list = list(gene_list_df['gene_name'])
        
        # 원본 데이터를 pandas DataFrame으로 변환
        idx = adata.obs_names.tolist()
        try:
            col = adata.var.gene_name.tolist()
        except:
            col = adata.var_names.tolist()
        
        if sp.issparse(adata.X):
            gexpr_feature = adata.X.toarray()
        else:
            gexpr_feature = adata.X
        
        self.gexpr_feature = pd.DataFrame(gexpr_feature, index=idx, columns=col)
        
        # main_gene_selection: 19264 genes로 변환 (원논문과 동일)
        if self.gexpr_feature.shape[1] < 19264:
            print(f'Converting gene feature from {self.gexpr_feature.shape[1]} to 19264 genes')
            self.gexpr_feature = self._main_gene_selection(self.gexpr_feature, self.gene_list)
            assert self.gexpr_feature.shape[1] >= 19264, "Gene selection failed"
        
        self.N = self.gexpr_feature.shape[0]
        
        # 전처리 옵션
        self.pre_normalized = config.preprocessing.get("option", "F")  # 'F', 'T', 'A'
        
        # tgthighres
        tg = config.inference["tgthighres"]
        self.tg_mode = tg[0]  # 'f', 'a', or 't'
        self.tg_val = float(tg[1:])  # numeric value
        
        print(f"ScFoundationDataset initialized: {self.N} cells, 19264 genes")
        print(f"  pre_normalized={self.pre_normalized}, tgthighres={tg}")
    
    def _main_gene_selection(self, X_df, gene_list):
        """
        원논문 main_gene_selection 함수 재현.
        19264 genes로 변환, 없는 gene은 0으로 padding.
        """
        to_fill_columns = list(set(gene_list) - set(X_df.columns))
        
        if to_fill_columns:
            padding_df = pd.DataFrame(
                np.zeros((X_df.shape[0], len(to_fill_columns))),
                columns=to_fill_columns,
                index=X_df.index
            )
            X_df = pd.DataFrame(
                np.concatenate([X_df.values, padding_df.values], axis=1),
                index=X_df.index,
                columns=list(X_df.columns) + list(padding_df.columns)
            )
        
        # gene_list 순서로 재배치
        X_df = X_df[gene_list]
        return X_df
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        """
        원논문 get_embedding.py Line 159-184를 정확히 재현.
        
        Foundations 코드:
        ```python
        if args.pre_normalized == 'F':
            tmpdata = (np.log1p(gexpr_feature.iloc[i,:]/(gexpr_feature.iloc[i,:].sum())*1e4)).tolist()
        elif args.pre_normalized == 'T':
            tmpdata = (gexpr_feature.iloc[i,:]).tolist()
        elif args.pre_normalized == 'A':
            tmpdata = (gexpr_feature.iloc[i,:-1]).tolist()
        
        if args.pre_normalized == 'A':
            totalcount = gexpr_feature.iloc[i,-1]
        else:
            totalcount = gexpr_feature.iloc[i,:].sum()
        
        if args.tgthighres[0] == 'f':
            pretrain_gene_x = torch.tensor(tmpdata+[np.log10(totalcount*float(args.tgthighres[1:])),np.log10(totalcount)])
        elif args.tgthighres[0] == 'a':
            pretrain_gene_x = torch.tensor(tmpdata+[np.log10(totalcount)+float(args.tgthighres[1:]),np.log10(totalcount)])
        elif args.tgthighres[0] == 't':
            pretrain_gene_x = torch.tensor(tmpdata+[float(args.tgthighres[1:]),np.log10(totalcount)])
        ```
        """
        # Pre-normalization (원논문 Line 159-165)
        if self.pre_normalized == 'F':
            # normalize_total=10000 + log1p
            cell_sum = self.gexpr_feature.iloc[idx, :].sum()
            if cell_sum > 0:
                tmpdata = np.log1p(self.gexpr_feature.iloc[idx, :] / cell_sum * 1e4)
            else:
                tmpdata = self.gexpr_feature.iloc[idx, :]
            tmpdata = tmpdata.tolist()
        elif self.pre_normalized == 'T':
            tmpdata = self.gexpr_feature.iloc[idx, :].tolist()
        elif self.pre_normalized == 'A':
            tmpdata = self.gexpr_feature.iloc[idx, :-1].tolist()
        else:
            raise ValueError(f'pre_normalized must be T, F or A, got {self.pre_normalized}')
        
        # Totalcount (원논문 Line 167-170)
        if self.pre_normalized == 'A':
            totalcount = self.gexpr_feature.iloc[idx, -1]
        else:
            totalcount = self.gexpr_feature.iloc[idx, :].sum()
        
        # Resolution token (원논문 Line 172-179)
        if self.tg_mode == 'f':
            resolution = np.log10(totalcount * self.tg_val)
        elif self.tg_mode == 'a':
            resolution = np.log10(totalcount) + self.tg_val
        elif self.tg_mode == 't':
            resolution = self.tg_val
        else:
            raise ValueError(f'tgthighres must start with f, a or t, got {self.tg_mode}')
        
        logtc = np.log10(totalcount) if totalcount > 0 else -np.inf
        
        # pretrain_gene_x: [19264 genes, resolution, logtc] (원논문 Line 173-179)
        pretrain_gene_x = torch.tensor(
            tmpdata + [resolution, logtc],
            dtype=torch.float32
        ).unsqueeze(0)  # (1, 19266)
        
        # data_gene_ids: [0, 1, 2, ..., 19265] (원논문 Line 180)
        data_gene_ids = torch.arange(19266, dtype=torch.long).unsqueeze(0)  # (1, 19266)
        
        # value_labels: mask for non-zero values (원논문 Line 182)
        value_labels = (pretrain_gene_x > 0).float()  # (1, 19266)
        
        # gatherData 적용 (원논문 Line 183)
        x, x_padding = self._gatherData(pretrain_gene_x, value_labels, self.pad_token_id)
        position_gene_ids, _ = self._gatherData(data_gene_ids, value_labels, self.pad_token_id)
        
        # 반환: (values, padding_mask, position_ids, cell_id)
        return (
            x.squeeze(0),  # (K,)
            x_padding.squeeze(0),  # (K,) bool
            position_gene_ids.squeeze(0),  # (K,) long
            torch.tensor(idx, dtype=torch.long),  # cell id
        )
    
    def _gatherData(self, data, labels, pad_token_id):
        """
        원논문 load.py의 gatherData 함수 재현.
        
        data: (1, F) tensor
        labels: (1, F) binary mask
        pad_token_id: int
        
        Returns:
        - gathered_data: (1, K) where K = max(labels.sum())
        - padding_labels: (1, K) bool mask
        """
        # labels.sum(1)의 최대값
        max_num = int(labels.sum(1).max().item())
        
        # Padding 추가
        fake_data = torch.full((data.shape[0], max_num), pad_token_id, dtype=data.dtype)
        data_padded = torch.cat([data, fake_data], dim=1)
        
        # Labels에 우선순위 부여 (원논문 로직)
        fake_label = torch.ones((labels.shape[0], max_num), dtype=labels.dtype)
        none_labels = (labels == 0)
        labels_copy = labels.clone().float()
        labels_copy[none_labels] = -float('inf')
        
        # 위치 기반 우선순위 추가
        F = labels.shape[1]
        tmp_data = torch.tensor(
            [(i + 1) * 20000 for i in range(F, 0, -1)],
            dtype=labels_copy.dtype
        )
        labels_copy = labels_copy + tmp_data
        labels_padded = torch.cat([labels_copy, fake_label], dim=1)
        
        # Top-K 선택
        topk_indices = labels_padded.topk(max_num, dim=1).indices
        
        # Gather
        gathered_data = torch.gather(data_padded, 1, topk_indices)
        padding_labels = (gathered_data == pad_token_id)
        
        return gathered_data, padding_labels
