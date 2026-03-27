from ..registry.dataset import ScGPTDataset
from .base._baseinferencer import BaseInferencer
from .models import ScGPTInferenceWrapper


class ScGPTInferencer(BaseInferencer):
    def build_dataset(self, adata):
        return ScGPTDataset(adata, self.cfg)

    def build_model(self):
        return ScGPTInferenceWrapper(self.cfg)

    def forward_step(self, model, batch):
        input_gene_ids = batch["gene"]
        src_key_padding_mask = input_gene_ids.eq(batch["pad_token_id"])
        emb = model(input_gene_ids, expr=batch["expr"], src_key_padding_mask=src_key_padding_mask)
        cid = batch["cid"]
        return emb, cid
