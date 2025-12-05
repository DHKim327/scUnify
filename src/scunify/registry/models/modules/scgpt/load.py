from .gene_tokenizer import *
from .model import TransformerModel


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


def load(args):
    model_dir = args.dir_param.model_loc
    model_file = model_dir / f"{args.dir_param.model_file}"

    vocab = GeneVocab.from_file(args.preproc_param.vocab)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=args.model_param.embsize,
        nhead=args.model_param.nheads,
        d_hid=args.model_param.d_hid,
        nlayers=args.model_param.nlayers,
        nlayers_cls=args.model_param.n_layers_cls,
        n_cls=1,
        vocab_pad_token=args.preproc_param.vocab_pad_token,
        dropout=args.model_param.dropout,
        pad_token=args.model_param.pad_token,
        pad_value=args.model_param.pad_value,
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    model = load_pretrained(model, torch.load(model_file, map_location="cpu"), verbose=False)
    return model
