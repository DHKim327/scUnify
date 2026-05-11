"""Base PertData — single-adata wrapper for perturbation tasks.

Reads everything from a single ``NORMAN.h5ad`` (or equivalent):

    adata.X, adata.var.gene_name, adata.obs.condition, adata.obs.total_count
    adata.uns['gene2go']                    (dict; '/' → '_SLASH_' escape)
    adata.uns['splits_<type>_<seed>_<gss>'] (set2conditions)
    adata.uns['splits_subgroup']            (simulation subgroup info)
    adata.uns['rank_genes_groups_cov_all']  (DE genes)

Sister cache files (built lazily, next to the .h5ad):

    <adata_path>.cell_graph_<format>.pkl                         (PyG cell graphs)
    <adata_path>.expanded_19264.h5ad                             (scfoundation only)
    <adata_path>.go.csv                                          (scfoundation only — paper-faithful)
    <adata_path>.simulation_<seed>_<gss>_0.4_20_co_expression_network.csv
                                                                 (scfoundation only — paper-faithful)
    <adata_path>_caches/<format>/                                (GEARS workdir; fed by sister files)

Subclass and override:

- ``format``     — string used in cache filename ("scgpt" / "scfoundation")
- ``_expand_adata(adata)`` — optional preprocessing (e.g. 5045→19264)
- ``create_cell_graph_dataset(...)`` — backbone-specific cell-graph stack
"""
from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0


class CidPyGLoaderWrapper:
    """Wrap a PyG ``DataLoader`` to inject a sequential ``batch.cid`` per batch.

    Perturbation tasks yield ``torch_geometric.data.Data`` batches; the
    framework extraction loop (``_collect_outputs_one_pass``) needs a
    ``cid`` tensor per batch to order predictions back into the source
    ``adata``. PyG ``Data`` does not support integer indexing, so we
    inject ``batch.cid`` lazily here without touching the cached
    cell-graph pickle.

    The cid is a per-pass cumulative counter (0..N-1 across the loader).
    Single-GPU extraction reads it directly; in distributed extraction
    each rank yields its own local cumulative cid and the framework's
    ``accelerator.gather`` plus ``argsort`` restores deterministic
    ordering as long as the dataloader iterates the same items in the
    same order on every rank — which it does because the loader is
    constructed with ``shuffle=False`` for inference.
    """

    def __init__(self, base_loader):
        self._base = base_loader

    def __iter__(self):
        offset = 0
        for batch in self._base:
            n = int(batch.num_graphs)
            batch.cid = torch.arange(offset, offset + n, dtype=torch.long)
            offset += n
            yield batch

    def __len__(self):
        return len(self._base)

    @property
    def dataset(self):
        return self._base.dataset

    @property
    def batch_size(self):
        return self._base.batch_size


def _filter_pert_in_go(condition: str, pert_names: np.ndarray) -> bool:
    """byte-level identical to ``gears.utils.filter_pert_in_go`` —
    excludes conditions whose perturbed gene is missing from the GO graph."""
    if condition == "ctrl":
        return True
    a, b = condition.split("+")
    num_ctrl = (a == "ctrl") + (b == "ctrl")
    num_in_perts = (a in pert_names) + (b in pert_names)
    return num_ctrl + num_in_perts == 2


class BasePertData:
    """Base PertData driven by a single ``adata.h5ad``.

    Subclasses choose the cell-graph format (``format`` class attr) and
    override ``create_cell_graph_dataset`` for the per-paper layout.
    """

    format: str = "base"

    # ------------------------------------------------------------------ #
    #  __init__ — read everything from adata + uns
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        adata_path: str | os.PathLike,
        *,
        split_type: str = "simulation",
        seed: int = 1,
        train_gene_set_size: float = 0.75,
    ):
        self.adata_path = str(adata_path)
        self.split_type = split_type
        self.seed = seed
        self.train_gene_set_size = train_gene_set_size

        # 1. Load adata, run subclass expand hook
        adata = sc.read_h5ad(self.adata_path)
        adata.obs_names_make_unique()
        adata = self._expand_adata(adata)
        self.adata = adata

        # 2. gene2go from uns (decode '_SLASH_' → '/')
        if "gene2go" not in adata.uns:
            raise KeyError(
                f"{self.adata_path}.uns is missing 'gene2go' — re-run prep "
                f"or pre-package gene2go via uns."
            )
        g2g_raw = dict(adata.uns["gene2go"])
        gene2go = {k.replace("_SLASH_", "/"): list(v) for k, v in g2g_raw.items()}
        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
        self.gene2go = gene2go

        # 3. Filter conditions whose perts are missing from GO graph
        cond = self.adata.obs.condition.astype(str)
        keep_mask = cond.apply(lambda x: _filter_pert_in_go(x, self.pert_names))
        not_in_go = np.array(cond[~keep_mask].unique())
        if len(not_in_go) > 0:
            print(f"[PertData] excluded {len(not_in_go)} condition(s) not in GO graph: {not_in_go}")
        self.adata = self.adata[keep_mask.values, :]

        # 4. set2conditions split + subgroup (uns is authoritative)
        split_key = f"splits_{split_type}_{seed}_{train_gene_set_size}"
        if split_key not in self.adata.uns:
            raise KeyError(
                f"{self.adata_path}.uns is missing '{split_key}' — pre-bake "
                f"the split into uns or use a matching (seed, gss)."
            )
        s2c = dict(self.adata.uns[split_key])
        self.set2conditions = {k: list(v) for k, v in s2c.items()}
        self.split = split_type   # ``no_test`` not supported here
        self.subgroup = (
            dict(self.adata.uns["splits_subgroup"])
            if "splits_subgroup" in self.adata.uns
            else None
        )

        # 5. GEARS-compatible cache layout (only needed by scfoundation; for
        #    scgpt this is harmless filesystem noise). data_path/dataset_name
        #    are the attrs the GEARS class reads to find go.csv / coexpress
        #    cache; we route them under <adata_path>_caches/<format>/.
        self._cache_root = f"{self.adata_path}_caches"
        self._cache_dir = os.path.join(self._cache_root, self.format)
        os.makedirs(self._cache_dir, exist_ok=True)
        self.data_path = self._cache_root
        self.dataset_name = self.format
        self.dataset_path = self._cache_dir   # legacy alias used by some utils
        self.gi_go = False
        self._sync_sister_caches()

        # 6. Cell-graph cache (sister file next to the adata)
        self.gene_names = self.adata.var.gene_name
        self.ctrl_adata = self.adata[self.adata.obs["condition"] == "ctrl"]
        self.dataset_processed = self._build_or_load_cell_graphs()
        # Stale-cache guard: if the cache was built before
        # ``_filter_pert_in_go`` (or with a different go.csv), it can carry
        # graphs for conditions that no longer exist in the filtered
        # ``self.adata``. Drop them so ``cell_graphs`` /
        # ``inference_adata`` row counts match.
        valid_conditions = set(self.adata.obs.condition.astype(str).unique())
        stale = [k for k in self.dataset_processed if k not in valid_conditions]
        if stale:
            print(
                f"[PertData] cache had {len(stale)} stale condition(s) absent "
                f"from filtered adata — dropping: "
                f"{stale[:5]}{'...' if len(stale) > 5 else ''}"
            )
            self.dataset_processed = {
                k: v for k, v in self.dataset_processed.items()
                if k in valid_conditions
            }

    # ------------------------------------------------------------------ #
    #  Subclass hook: optional adata preprocessing (default no-op)
    # ------------------------------------------------------------------ #
    def _expand_adata(self, adata):
        return adata

    # ------------------------------------------------------------------ #
    #  GEARS cache layout — write gene2go.pkl + copy sister CSVs once
    # ------------------------------------------------------------------ #
    def _sync_sister_caches(self) -> None:
        """Populate ``<cache_root>/`` with files GEARS expects:

        - ``gene2go.pkl``         (decoded; from ``adata.uns['gene2go']``)
        - ``<format>/go.csv``     (paper-faithful — copied from sister
          ``<adata_path>.go.csv`` if present; otherwise GEARS auto-builds
          ~slow on first init)
        - ``<format>/<split>_<seed>_<gss>_0.4_20_co_expression_network.csv``
          (paper-faithful — copied from corresponding sister file)
        """
        # gene2go.pkl (used by ``utils.get_go_auto`` fallback). Always write
        # — cheap, ensures GEARS finds the data without reaching for the
        # internet.
        g2g_path = os.path.join(self._cache_root, "gene2go.pkl")
        if not os.path.exists(g2g_path):
            with open(g2g_path, "wb") as f:
                pickle.dump(self.gene2go, f)

        # Paper-faithful pre-built artifacts shipped as sister files.
        # Only ``go.csv`` is gene-vocab-independent (it is built from
        # ``pert_names`` keyed by gene2go). The co-expression graph is
        # adata-gene-vocab specific — backbones with a different gene list
        # (e.g. scfoundation's 19264 expand) need a freshly-built one, so
        # we never ship the co-express CSV; GEARS auto-builds it on first
        # init for the actual runtime vocab.
        sister = f"{self.adata_path}.go.csv"
        dst = os.path.join(self._cache_dir, "go.csv")
        if os.path.exists(sister) and not os.path.exists(dst):
            shutil.copy(sister, dst)

    # ------------------------------------------------------------------ #
    #  Cache build / load
    # ------------------------------------------------------------------ #
    @property
    def cell_graph_cache(self) -> str:
        """Sister cache path: ``<adata_path>.cell_graph_<format>.pkl``."""
        return f"{self.adata_path}.cell_graph_{self.format}.pkl"

    def _build_or_load_cell_graphs(self) -> dict:
        cache = self.cell_graph_cache
        if os.path.isfile(cache):
            print(f"[PertData] loading cell-graph cache: {cache}")
            with open(cache, "rb") as f:
                return pickle.load(f)
        print(f"[PertData] building cell-graph cache (format={self.format}) → {cache}")
        dl = self._create_dataset_file()
        with open(cache, "wb") as f:
            pickle.dump(dl, f)
        print(f"[PertData] saved {cache}")
        return dl

    def _create_dataset_file(self) -> dict:
        dl = {}
        for p in tqdm(self.adata.obs["condition"].unique()):
            dl[p] = self.create_cell_graph_dataset(self.adata, p, num_samples=1)
        return dl

    # ------------------------------------------------------------------ #
    #  pert_idx — paper logic (verbatim from gears.PertData.get_pert_idx)
    # ------------------------------------------------------------------ #
    def get_pert_idx(self, pert_category: str):
        if pert_category == "ctrl":
            return None
        try:
            return [
                int(np.where(p == self.pert_names)[0][0])
                for p in pert_category.split("+")
                if p != "ctrl"
            ]
        except IndexError:
            return None

    # ------------------------------------------------------------------ #
    #  Subclass MUST override (per-paper recipe)
    # ------------------------------------------------------------------ #
    def create_cell_graph_dataset(self, split_adata, pert_category: str, num_samples: int = 1):
        raise NotImplementedError(
            f"{type(self).__name__}.create_cell_graph_dataset must be overridden "
            f"with the backbone-specific cell-graph layout."
        )

    # ------------------------------------------------------------------ #
    #  DataLoader build (paper-faithful — 같은 split.pkl 사용)
    # ------------------------------------------------------------------ #
    def get_dataloader(self, batch_size: int, test_batch_size: int | None = None):
        if test_batch_size is None:
            test_batch_size = batch_size
        self.node_map = {x: it for it, x in enumerate(self.adata.var.gene_name)}

        cell_graphs = {}
        splits = ["train", "val", "test"]
        for s in splits:
            cell_graphs[s] = []
            for p in self.set2conditions.get(s, []):
                # GO filter may have removed conditions from dataset_processed
                # but split.pkl still lists them — silently skip stale entries.
                graphs_p = self.dataset_processed.get(p)
                if graphs_p is not None:
                    cell_graphs[s].extend(graphs_p)

        train_loader = DataLoader(
            cell_graphs["train"], batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            cell_graphs["val"], batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            cell_graphs["test"], batch_size=test_batch_size, shuffle=False
        )
        # ``all_loader`` — train+val+test concatenated in deterministic order
        # (split-major, then condition order from set2conditions). Used by
        # perturbation mixins for ``save.outputs`` extraction so all 91k+
        # cells get a perturbation_pred + cell_embedding under the trained
        # model (paper-faithful metric still uses test_loader only).
        all_graphs = cell_graphs["train"] + cell_graphs["val"] + cell_graphs["test"]
        all_loader = DataLoader(
            all_graphs, batch_size=test_batch_size, shuffle=False
        )
        # Save per-split sizes — mixins use these to slice the source adata
        # in the same order as ``all_graphs`` for ``inference_adata``.
        self._all_split_order = {
            "train": list(self.set2conditions.get("train", [])),
            "val":   list(self.set2conditions.get("val", [])),
            "test":  list(self.set2conditions.get("test", [])),
        }
        self.dataloader = {
            "train_loader": train_loader,
            "val_loader":   val_loader,
            "test_loader":  test_loader,
            "all_loader":   all_loader,
        }
        return self.dataloader
