import logging

import anndata as ad
import numba
import numpy as np
import torch
from pyensembl import EnsemblRelease
from scipy.sparse import issparse
from torch.utils.data import Dataset, SequentialSampler

logger = logging.getLogger(__name__)

# ── pyensembl release config ─────────────────────────────────────────
_ENSEMBL_RELEASE = {
    "human": 110,  # GRCh38
    "mouse": 110,  # GRCm39
}
_PYENSEMBL_SPECIES = {
    "human": "human",
    "mouse": "mouse",
}

# ── Token ID constants (matching HF theislab/nicheformer) ──────────────
PAD_TOKEN = 0
AUX_TOKENS = 30  # gene token IDs start at index 30

SPECIES_TOKENS = {
    "human": 5, "NCBITaxon:9606": 5,
    "mouse": 6, "NCBITaxon:10090": 6,
}

MODALITY_TOKENS = {
    "dissociated": 3,
    "spatial": 4,
}

ASSAY_TOKENS = {
    # Ontology term IDs
    "EFO:0008992": 7,   # MERFISH
    "EFO:0030029": 8,   # CosMx / GeoMx
    "EFO:0030080": 9,   # 10x (generic)
    "EFO:0030003": 10,  # 10x 3'
    "EFO:0009899": 11,  # 10x 3'v2
    "EFO:0009922": 12,  # 10x 3'v3
    "EFO:0030004": 13,  # 10x 5'
    "EFO:0011025": 14,  # 10x 5'v1
    "EFO:0009900": 15,  # 10x 5'v2
    "EFO:0008931": 18,  # Smart-seq2
    # Shorthand aliases
    "merfish": 7, "cosmx": 8, "10x": 9,
    "10x_3": 10, "10x_3v2": 11, "10x_3v3": 12,
    "10x_5": 13, "10x_5v1": 14, "10x_5v2": 15,
    "smart_seq2": 18,
}


# ── Tokenization (original Nicheformer algorithm) ─────────────────────

def _sf_normalize(X):
    """Size-factor normalize to 10 000 counts per cell."""
    counts = X.sum(axis=1, keepdims=True)
    counts[counts == 0] = 1.0
    return X * (10000.0 / counts)


@numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(x, max_seq_len=4096, aux_tokens=30):
    """Rank genes by expression (descending) and convert to token IDs.

    Identical to the original nicheformer ``_sub_tokenize_data``.
    """
    out = np.empty((x.shape[0], max_seq_len), dtype=np.int32)
    for i in range(x.shape[0]):
        cell = x[i]
        nonzero_mask = np.nonzero(cell)[0]
        sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len]
        sorted_indices = sorted_indices + aux_tokens
        scores = np.zeros(max_seq_len, dtype=np.int32)
        scores[:len(sorted_indices)] = sorted_indices.astype(np.int32)
        out[i, :] = scores
    return out


# ── Dataset ────────────────────────────────────────────────────────────

class NicheformerDataset(Dataset):
    """Dataset for Nicheformer inference in scUnify.

    Tokenisation pipeline (original algorithm — no log1p):
        raw counts → sf_normalize(10 000) → ÷ technology_mean
        → nonzero mask → argsort(descending) → top *max_seq_len*
        → gene_index + 30 → int32 token IDs

    Context tokens prepended in order: ``[species, assay, modality, genes…]``
    then truncated to *context_length* (default 1 500).

    Attention-mask convention follows HuggingFace: **1 = attend, 0 = pad**.
    """

    def __init__(self, adata, config):
        resources = config.get("resources", {})
        inference_cfg = config.get("inference", {})

        # ── Gene reference ──────────────────────────────────────────
        gene_ref_path = resources["gene_ref_file"]
        ref_adata = ad.read_h5ad(gene_ref_path)
        n_ref = ref_adata.n_vars
        logger.info(f"Reference genes: {n_ref}")

        # ── Technology mean (already in reference gene order) ───────
        tech_mean_raw = np.load(resources["technology_mean_file"]).copy()
        tech_mean_raw[tech_mean_raw == 0] = 1.0

        # ── Gene symbol → ENSEMBL ID conversion ─────────────────────
        ensembl_key = inference_cfg.get("ensembl_key", None)
        adata = _ensure_ensembl_var_names(
            adata,
            ensembl_key=ensembl_key,
            species=inference_cfg.get("species", "human"),
        )

        # ── Gene alignment via ad.concat (HF tokenizer strategy) ──
        # Outer-join concat aligns genes by var_names, filling missing
        # genes with 0. Then keep only reference genes (adata-only genes
        # would produce token IDs beyond the model's vocabulary size).
        aligned = ad.concat([ref_adata, adata], join="outer", axis=0)
        aligned = aligned[1:]  # remove reference row
        aligned = aligned[:, ref_adata.var_names]  # keep only ref genes
        n_matched = int((aligned.X.sum(axis=0) != 0).sum()) if issparse(aligned.X) else int((aligned.X.sum(axis=0) != 0).sum())
        logger.info(
            f"Gene alignment: {n_ref} ref genes, "
            f"{n_matched} matched from adata ({adata.n_vars} input genes)"
        )

        # tech_mean stays in reference gene order (no expansion needed)
        tech_mean = tech_mean_raw

        # ── Build aligned expression matrix ─────────────────────────
        X = aligned.X
        if issparse(X):
            aligned_X = X.toarray().astype(np.float64)
        else:
            aligned_X = np.asarray(X, dtype=np.float64)
        del aligned

        # ── Tokenise ────────────────────────────────────────────────
        aligned_X = np.nan_to_num(aligned_X)
        aligned_X = _sf_normalize(aligned_X)
        aligned_X = aligned_X / tech_mean.reshape(1, -1)

        max_seq_len = inference_cfg.get("max_seq_len", 4096)
        self.tokens = _sub_tokenize_data(aligned_X, max_seq_len, AUX_TOKENS)
        del aligned_X

        # ── Context tokens ──────────────────────────────────────────
        self.species_token = _resolve_token(
            inference_cfg.get("species", "human"), SPECIES_TOKENS, "species"
        )
        self.assay_token = _resolve_token(
            inference_cfg.get("assay", "10x_3v3"), ASSAY_TOKENS, "assay"
        )
        self.modality_token = _resolve_token(
            inference_cfg.get("modality", "dissociated"), MODALITY_TOKENS, "modality"
        )
        self.context_length = inference_cfg.get("context_length", 1500)

        self.n_cells = adata.n_obs
        self.sampler = SequentialSampler(self)

    # ── per-cell output ─────────────────────────────────────────────
    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        gene_tokens = self.tokens[idx]  # (max_seq_len,) int32, 0-padded

        # Prepend context: [species, assay, modality]
        context = np.array(
            [self.species_token, self.assay_token, self.modality_token],
            dtype=np.int32,
        )
        full_seq = np.concatenate([context, gene_tokens])[: self.context_length]

        # Pad to exactly context_length (safety — always true when max_seq_len ≥ context_length-3)
        if len(full_seq) < self.context_length:
            padded = np.zeros(self.context_length, dtype=np.int32)
            padded[: len(full_seq)] = full_seq
            full_seq = padded

        # Attention mask: 1 for real tokens, 0 for PAD (HF convention)
        n_genes = int(np.count_nonzero(gene_tokens))
        n_real = min(3 + n_genes, self.context_length)
        attn_mask = np.zeros(self.context_length, dtype=np.int64)
        attn_mask[:n_real] = 1

        return {
            "input_ids": torch.from_numpy(full_seq.astype(np.int64)),
            "attention_mask": torch.from_numpy(attn_mask),
            "cid": idx,
        }

    @staticmethod
    def collator(batch):
        """Stack fixed-length sequences into a batch."""
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "cid": torch.tensor([b["cid"] for b in batch], dtype=torch.long),
        }


# ── helpers ────────────────────────────────────────────────────────────

def _resolve_token(value, token_dict, name):
    """Map a human-readable string (or raw int) to a token ID."""
    if isinstance(value, int):
        return value
    token_id = token_dict.get(str(value))
    if token_id is None:
        raise ValueError(
            f"Unknown {name}: {value!r}. "
            f"Valid options: {list(token_dict.keys())}"
        )
    return token_id


def _ensure_ensembl_var_names(adata, ensembl_key=None, species="human"):
    """Ensure adata.var_names are ENSEMBL Gene IDs.

    If ``ensembl_key`` is a column name in adata.var, use that column.
    If ``ensembl_key`` is None/False, check whether var_names already look
    like ENSEMBL IDs (ENSG.../ENSMUSG...).  If not, convert gene symbols
    to ENSEMBL IDs via pyensembl.
    """
    prefix = "ENSMUSG" if species == "mouse" else "ENSG"

    # Case 1: explicit column in adata.var
    if ensembl_key and ensembl_key in adata.var.columns:
        logger.info(f"Using adata.var['{ensembl_key}'] as ENSEMBL IDs")
        adata = adata.copy()
        adata.var_names = adata.var[ensembl_key].astype(str).values
        adata.var_names_make_unique()
        return adata

    # Case 2: already ENSEMBL IDs
    sample = adata.var_names[:20]
    n_ensembl = sum(1 for g in sample if str(g).startswith(("ENSG", "ENSMUSG")))
    if n_ensembl >= len(sample) * 0.8:
        logger.info("var_names already appear to be ENSEMBL IDs, skipping conversion")
        return adata

    # Case 3: convert gene symbols → ENSEMBL IDs via pyensembl
    logger.info(f"Converting gene symbols → ENSEMBL IDs via pyensembl (species={species})")
    release_num = _ENSEMBL_RELEASE.get(species, 110)
    ensembl = EnsemblRelease(release=release_num, species=_PYENSEMBL_SPECIES.get(species, "human"))
    ensembl.download()
    ensembl.index()

    symbol_to_id = {}
    for gene_name in adata.var_names:
        gene_name_str = str(gene_name)
        try:
            ids = ensembl.gene_ids_of_gene_name(gene_name_str)
            # Filter to matching species prefix
            ids = [g for g in ids if g.startswith(prefix)]
            if ids:
                symbol_to_id[gene_name_str] = ids[0]
        except ValueError:
            pass

    n_mapped = len(symbol_to_id)
    n_total = len(adata.var_names)
    logger.info(f"pyensembl mapped {n_mapped}/{n_total} gene symbols to ENSEMBL IDs")

    if n_mapped == 0:
        raise ValueError(
            f"pyensembl could not map any gene symbols to ENSEMBL IDs. "
            f"Check that var_names contain valid gene symbols for species='{species}'."
        )

    # Subset to mapped genes and rename
    mapped_genes = [g for g in adata.var_names if str(g) in symbol_to_id]
    adata = adata[:, mapped_genes].copy()
    adata.var["gene_symbol"] = adata.var_names.tolist()
    adata.var_names = [symbol_to_id[str(g)] for g in adata.var_names]
    adata.var_names_make_unique()

    logger.info(f"After ENSEMBL conversion: {adata.n_vars} genes retained")
    return adata
