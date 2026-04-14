"""Shared utilities for task Mixins."""


def infer_emb_dim(cfg) -> int:
    """Infer cell embedding dimension from model architecture config.

    Reads ``cfg.model_param`` (loaded from architecture YAML) and
    returns the expected output dimension of ``get_cell_embedding()``.
    """
    model_name = cfg.get("model_name", "").lower().replace(" ", "")
    arch = getattr(cfg, "model_param", {})

    if model_name == "scgpt":
        return int(arch.get("d_model", 512))

    if model_name == "geneformer":
        variant = cfg.get("model", {}).get("variant", "V2-104M")
        variant_cfg = arch.get(variant, {})
        return int(variant_cfg.get("hidden_size", 512))

    if model_name == "nicheformer":
        default_cfg = arch.get("default", arch)
        return int(default_cfg.get("dim_model", 512))

    if model_name == "scfoundation":
        version = cfg.get("model", {}).get("version", "cell")
        hidden = int(
            arch.get(version, {})
            .get("mae_autobin", {})
            .get("encoder", {})
            .get("hidden_dim", 768)
        )
        pool_type = cfg.get("model", {}).get("pool_type", "all")
        return hidden * 4 if pool_type == "all" else hidden

    if model_name == "uce":
        nlayers = cfg.get("model", {}).get("nlayers", 4)
        layer_cfg = arch.get(nlayers, arch.get(str(nlayers), {}))
        return int(layer_cfg.get("output_dim", 1280))

    # Fallback
    return 512
