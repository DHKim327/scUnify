import torch
import torch.nn as nn


class UCEWrapper(nn.Module):
    def __init__(self, config):
        super(UCEWrapper, self).__init__()
        model = load(config)
        self.pe_embedding = model.pe_embedding
        self.encoder = model

    def forward(self, batch_sentences, mask):
        batch_sentences = batch_sentences.permute(1, 0)
        batch_sentences = self.pe_embedding(batch_sentences.long())
        batch_sentences = nn.functional.normalize(batch_sentences, dim=2)  # Normalize token outputs now
        _, embedding = self.encoder(batch_sentences, mask=mask)
        return embedding


from ...utils import load_yaml
from .modules.uce.model import TransformerModel


def load(config):
    nlayers = config.inference["nlayers"]
    model_loc = config.resources[f"{nlayers}_layer_model"]
    args = load_yaml(config._architecture_dir)[nlayers]
    #### Set up the model ####
    token_dim = args["token_dim"]
    emsize = 1280  # embedding dimension
    d_hid = args["d_hid"]  # dimension of the feedforward network model in nn.TransformerEncoder
    nhead = 20  # number of heads in nn.MultiheadAttention
    dropout = 0.05  # dropout probability
    output_dim = args["output_dim"]

    model = TransformerModel(
        token_dim=token_dim,
        d_model=emsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        dropout=dropout,
        output_dim=output_dim,
    )

    empty_pe = torch.zeros(145469, 5120)
    empty_pe.requires_grad = False
    model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
    model.load_state_dict(torch.load(model_loc, map_location="cpu"), strict=True)

    token_file = config.resources["token_file"]
    all_pe = get_ESM2_embeddings(token_file, token_dim)
    # This will make sure that you don't overwrite the tokens in case you're embedding species from the training data
    # We avoid doing that just in case the random seeds are different across different versions.
    if all_pe.shape[0] != 145469:
        all_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(all_pe)

    return model


# Load in ESM2 embeddings and special tokens
def get_ESM2_embeddings(token_file, token_dim):
    all_pe = torch.load(token_file)
    if all_pe.shape[0] == 143574:
        torch.manual_seed(23)
        CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, token_dim))
        # 1895 is the total number of chromosome choices, it is hardcoded for now
        all_pe = torch.vstack((all_pe, CHROM_TENSORS))  # Add the chrom tensors to the end
        all_pe.requires_grad = False

    return all_pe
