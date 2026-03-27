import torch
import torch.nn as nn


class ScFoundationWrapper(nn.Module):
    """Base wrapper for scFoundation — full MaeAutobin loading, no forward.

    Keeps the complete model (encoder + decoder + to_final) so that both
    inference (encoder-only) and training (full MAE) wrappers can inherit.
    """

    def __init__(self, config):
        super(ScFoundationWrapper, self).__init__()
        self.model = load(config)


# -------------------------------- Load model utils --------------------------------#
from ...utils import load_yaml
from .modules.scfoundation.mae_autobin import MaeAutobin
from .modules.scfoundation.performer import PerformerModule
from .modules.scfoundation.transformer import pytorchTransformerModule


def load(config) -> torch.nn.Module:
    model_param = load_yaml(config._architecture_dir)[config.inference["version"]]
    model_name = list(model_param.keys())[0]
    params = model_param[model_name]

    model = build_model(model_name, params)
    state_dict = remove_model_prefix(
        torch.load(config.resources["model_file"])[config.inference["version"]]["state_dict"]
    )
    model.load_state_dict(state_dict)
    return model


def build_model(model_name, params) -> torch.nn.Module:
    if model_name == "mae_autobin":
        encoder = build_module(params["encoder"])
        decoder = build_module(params["decoder"])

        model = MaeAutobin(
            num_tokens=params["n_class"],
            max_seq_len=params["seq_len"],
            embed_dim=params["encoder"]["hidden_dim"],
            decoder_embed_dim=params["decoder"]["hidden_dim"],
            bin_alpha=params["bin_alpha"],
            bin_num=params["bin_num"],
            pad_token_id=params["pad_token_id"],
            mask_token_id=params["mask_token_id"],
        )
        model.encoder = encoder
        model.decoder = decoder
        return model

    raise NotImplementedError(f"Unsupported model type: {config.model}")


def build_module(module_cfg) -> torch.nn.Module:
    module_type = module_cfg["module_type"]

    common_kwargs = {
        "max_seq_len": module_cfg["seq_len"],
        "dim": module_cfg["hidden_dim"],
        "depth": module_cfg["depth"],
        "heads": module_cfg["heads"],
    }

    if module_type == "performer":
        return PerformerModule(
            **common_kwargs,
            dim_head=module_cfg["dim_head"],
            ff_dropout=module_cfg.get("ff_dropout", 0.0),
            attn_dropout=module_cfg.get("attn_dropout", 0.0),
        )
    elif module_type == "transformer":
        return pytorchTransformerModule(**common_kwargs)

    raise ValueError(f"Unknown module type: {module_type}")


def remove_model_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k[len("model.") :]  # Strip "model." prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict
