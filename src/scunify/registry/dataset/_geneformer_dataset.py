import logging
import pickle

import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset, SequentialSampler

logger = logging.getLogger(__name__)


class GeneformerDataset(Dataset):
    """
    Dataset for Geneformer V2 inference.

    Performs rank value encoding on raw counts:
    1. Normalize: counts / n_counts * 10000
    2. Scale by gene median: normalized / gene_median
    3. Rank genes by scaled expression (descending)
    4. Convert to token IDs
    5. Add [CLS] (front) and [EOS] (back) special tokens
    6. Truncate to model_input_size (4096)
    """

    # model_variant -> gene_dicts key mapping
    _VARIANT_TO_DICT = {
        "V1-10M": "30M",
        "V2-104M": "104M",
        "V2-104M_CLcancer": "104M",
        "V2-316M": "104M",
    }

    def __init__(self, adata, config):
        # Resolve gene dictionary paths based on model_variant
        inference_cfg = config.get("inference", {})
        variant = inference_cfg.get("model_variant", "V2-104M")
        dict_key = self._VARIANT_TO_DICT.get(variant, "104M")
        resources = config.get("resources", {})
        gene_dict_paths = resources["gene_dicts"][dict_key]

        with open(gene_dict_paths["gene_median_file"], "rb") as f:
            gene_median_dict = pickle.load(f)
        with open(gene_dict_paths["token_dict_file"], "rb") as f:
            gene_token_dict = pickle.load(f)
        with open(gene_dict_paths["ensembl_mapping_file"], "rb") as f:
            gene_mapping_dict = pickle.load(f)

        self.cls_token_id = gene_token_dict["<cls>"]
        self.eos_token_id = gene_token_dict["<eos>"]

        model_input_size = config.get("model_input_size", 4096)

        # Build ensembl_id mapping for adata
        # Priority: adata.var["ensembl_id"] > gene_name_id_dict mapping
        if "ensembl_id" in adata.var.columns:
            ensembl_ids = adata.var["ensembl_id"].values
        else:
            # Try mapping gene names to ensembl IDs
            with open(gene_dict_paths["gene_name_id_file"], "rb") as f:
                gene_name_id_dict = pickle.load(f)
            # gene_name_id_dict: {gene_name: ensembl_id}
            ensembl_ids = np.array([gene_name_id_dict.get(g, None) for g in adata.var_names])
            logger.info(
                f"Mapped {np.sum(ensembl_ids != None)}/{len(ensembl_ids)} gene names to Ensembl IDs"
            )

        # Map to canonical ensembl IDs via gene_mapping_dict
        # (identical to TranscriptomeTokenizer.sum_ensembl_ids)
        gene_keys_set = set(gene_token_dict.keys())
        gene_mapping_dict_filtered = {k: v for k, v in gene_mapping_dict.items() if v in gene_keys_set}

        canonical_ids = np.array([
            gene_mapping_dict_filtered.get(str(eid).upper(), None) if eid is not None else None
            for eid in ensembl_ids
        ])

        # Filter to genes in gene_token_dict (original: genelist_dict)
        valid_mask = np.array([cid is not None for cid in canonical_ids])
        valid_indices = np.where(valid_mask)[0]
        valid_canonical_ids = canonical_ids[valid_indices]

        logger.info(f"Valid genes for Geneformer: {len(valid_indices)}/{adata.n_vars}")

        # Collapse duplicate canonical IDs by summing counts
        # (identical to TranscriptomeTokenizer.sum_ensembl_ids)
        unique_canonical_ids, inverse_indices = np.unique(valid_canonical_ids, return_inverse=True)
        n_duplicates = len(valid_canonical_ids) - len(unique_canonical_ids)
        if n_duplicates > 0:
            logger.info(f"Collapsing {n_duplicates} duplicate canonical IDs by summing counts")

        # Get n_counts for normalization
        if "n_counts" in adata.obs.columns:
            self.n_counts = adata.obs["n_counts"].values.astype(np.float64)
        else:
            # Compute n_counts from raw data
            X = adata.X
            if issparse(X):
                self.n_counts = np.asarray(X.sum(axis=1)).flatten().astype(np.float64)
            else:
                self.n_counts = X.sum(axis=1).astype(np.float64)

        # Extract valid gene columns and collapse duplicates by summing
        X = adata.X
        if issparse(X):
            valid_X = X[:, valid_indices].toarray().astype(np.float64)
        else:
            valid_X = np.asarray(X[:, valid_indices], dtype=np.float64)

        # Always collapse via inverse_indices to ensure column order matches
        # unique_canonical_ids (np.unique sorts alphabetically)
        n_unique = len(unique_canonical_ids)
        collapsed_X = np.zeros((adata.n_obs, n_unique), dtype=np.float64)
        for i in range(n_unique):
            cols = np.where(inverse_indices == i)[0]
            collapsed_X[:, i] = valid_X[:, cols].sum(axis=1)
        self.count_matrix = collapsed_X

        # Precompute arrays for unique canonical genes
        self.norm_factor_vector = np.array([gene_median_dict[cid] for cid in unique_canonical_ids])
        self.gene_tokens = np.array([gene_token_dict[cid] for cid in unique_canonical_ids])

        self.model_input_size = model_input_size
        self.n_cells = adata.n_obs
        self.sampler = SequentialSampler(self)

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        raw_counts = self.count_matrix[idx]
        n_counts = self.n_counts[idx]

        # Normalize: counts / n_counts * 10000
        if n_counts > 0:
            normalized = raw_counts / n_counts * 10000.0
        else:
            normalized = raw_counts

        # Scale by gene median
        scaled = normalized / self.norm_factor_vector

        # Mask zero-expression genes
        nonzero_mask = raw_counts > 0
        nonzero_scaled = scaled[nonzero_mask]
        nonzero_tokens = self.gene_tokens[nonzero_mask]

        # Rank by scaled expression (descending)
        sorted_indices = np.argsort(-nonzero_scaled)
        ranked_tokens = nonzero_tokens[sorted_indices]

        # Truncate to leave space for CLS and EOS
        max_gene_tokens = self.model_input_size - 2
        ranked_tokens = ranked_tokens[:max_gene_tokens]

        # Add special tokens: [CLS] at front, [EOS] at back
        input_ids = np.concatenate([
            [self.cls_token_id],
            ranked_tokens,
            [self.eos_token_id],
        ]).astype(np.int64)

        return {
            "input_ids": torch.from_numpy(input_ids),
            "length": len(input_ids),
            "cid": idx,
        }

    @staticmethod
    def collator(batch):
        """Dynamic padding with length-based attention mask."""
        max_len = max(b["length"] for b in batch)
        bsz = len(batch)

        input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
        attn_mask = torch.zeros(bsz, max_len, dtype=torch.long)
        cids = []

        for i, b in enumerate(batch):
            seq_len = b["length"]
            input_ids[i, :seq_len] = b["input_ids"]
            # Use length-based mask (NOT input_ids != 0)
            # because CLS token ID = 0 = pad_token_id
            attn_mask[i, :seq_len] = 1
            cids.append(b["cid"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "cid": torch.tensor(cids, dtype=torch.long),
        }
