import torch
import torch.nn as nn


class ScGPTWrapper(nn.Module):
    def __init__(self, config):
        super(ScGPTWrapper, self).__init__()
        self.model = load(config)

    def forward(self, input_gene_ids, expr, src_key_padding_mask):
        embedding = self.model._encode(
            input_gene_ids,
            expr,
            src_key_padding_mask=src_key_padding_mask,
        )
        # Extract CLS token embedding
        embedding = embedding[:, 0, :]  # (batch_size, embed_dim)
        
        # L2 normalization (same as original scGPT implementation)
        # Original: cell_embeddings / np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding


# -------------------------------- Load model utils --------------------------------#
from ...utils import load_yaml
from .modules.scgpt.gene_tokenizer import GeneVocab
from .modules.scgpt.model import TransformerModel


def load(config):
    vocab = GeneVocab.from_file(config.resources["vocab_file"])
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    model_param = load_yaml(config._architecture_dir)
    model_param["ntoken"] = len(vocab)
    model_param["vocab_pad_token_idx"] = vocab[model_param["pad_token"]]

    model = TransformerModel(**model_param)
    model = load_pretrained(model, torch.load(config.resources["model_file"], map_location="cpu"), verbose=False)
    return model


def load_pretrained(
    model,
    pretrained_params,
    strict=False,
    prefix=None,
    verbose=True,
):
    use_flash_attn = getattr(model, "use_fast_transformer", True)
    if not use_flash_attn:
        pretrained_params = {k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_params.items()}

    if prefix is not None and len(prefix) > 0:
        if isinstance(prefix, str):
            prefix = [prefix]
        pretrained_params = {k: v for k, v in pretrained_params.items() if any(k.startswith(p) for p in prefix)}

    model_dict = model.state_dict()
    if strict:
        if verbose:
            for k, v in pretrained_params.items():
                print(f"Loading parameter {k} with shape {v.shape}")
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)
    else:
        if verbose:
            for k, v in pretrained_params.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    print(f"Loading parameter {k} with shape {v.shape}")
        pretrained_params = {
            k: v for k, v in pretrained_params.items() if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)

    return model
