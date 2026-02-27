# core/data_actor.py
"""Per-model DataLoader Actor.

A stateful Ray actor that runs within each model's conda environment
(scunify_scgpt, scunify_uce, scunify_scfoundation), enabling workers in the
same environment to share data via zero-copy reads from the Object Store.

Key features:
  - Each data path is loaded only once.
  - Serialization compatibility is guaranteed within the same env.
  - Data is shared via Ray Object Store (zero-copy).
"""

import ray


@ray.remote
class DataLoaderActor:
    """Data loader actor running within a specific model environment.

    Usage::

        # Create a per-model actor in the target conda env
        actor = DataLoaderActor.options(
            runtime_env={"conda": "scunify_scgpt"}
        ).remote()

        # Request data load
        adata_ref = ray.get(actor.get_or_load.remote("/path/to/data.h5ad"))

        # Use data in a worker
        adata = ray.get(adata_ref)
    """

    def __init__(self):
        import os
        # Set matplotlib backend to avoid Jupyter inline backend conflicts
        os.environ["MPLBACKEND"] = "Agg"

        self.cache = {}  # {path: ObjectRef}
        self._loaded_paths = []  # Load order tracking

    def get_or_load(self, path: str):
        """Load AnnData from path and store in the Object Store.

        Args:
            path: Path to an h5ad file.

        Returns:
            ray.ObjectRef: Reference to the AnnData in the Object Store.
        """
        import os
        os.environ["MPLBACKEND"] = "Agg"
        import matplotlib
        matplotlib.use("Agg")

        import numpy as np
        import scanpy as sc
        import scipy.sparse as sp

        if path not in self.cache:
            adata = sc.read_h5ad(path)

            # Convert to float32 for memory efficiency and model compatibility
            if sp.issparse(adata.X):
                adata.X = adata.X.astype(np.float32)
            else:
                adata.X = np.asarray(adata.X, dtype=np.float32, order="C")

            self.cache[path] = ray.put(adata)
            self._loaded_paths.append(path)

        return self.cache[path]

    def preload(self, paths: list[str]):
        """Preload multiple data paths.

        Args:
            paths: List of h5ad file paths.

        Returns:
            dict: {path: ObjectRef} mapping.
        """
        result = {}
        for path in paths:
            result[path] = self.get_or_load(path)
        return result

    def clear_cache(self, path: str = None):
        """Clear cached data.

        Args:
            path: Specific path to clear. If None, clears all.
        """
        if path:
            self.cache.pop(path, None)
            if path in self._loaded_paths:
                self._loaded_paths.remove(path)
        else:
            self.cache.clear()
            self._loaded_paths.clear()

    def get_cache_info(self) -> dict:
        """Return cache status information."""
        return {
            "n_cached": len(self.cache),
            "paths": list(self.cache.keys()),
            "load_order": self._loaded_paths.copy(),
        }

    def is_cached(self, path: str) -> bool:
        """Check whether a specific path is cached."""
        return path in self.cache


def create_model_actors(model_names: list[str]) -> dict:
    """Create per-model DataLoader actors.

    Args:
        model_names: List of model names (e.g., ["scGPT", "UCE", "scFoundation"]).

    Returns:
        dict: {model_name: DataLoaderActor} mapping.
    """
    actors = {}

    for model_name in model_names:
        env_name = f"scunify_{model_name.lower()}"

        actor = DataLoaderActor.options(
            runtime_env={"conda": env_name},
            name=f"data_loader_{model_name.lower()}",
            lifetime="detached",  # Persists after runner shutdown
        ).remote()

        actors[model_name] = actor

    return actors
