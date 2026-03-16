from ..registry.dataset import GeneformerDataset
from ..registry.models import GeneformerWrapper
from .base._baseinferencer import BaseInferencer


class GeneformerInferencer(BaseInferencer):
    def build_dataset(self, adata):
        return GeneformerDataset(adata, self.cfg)

    def build_model(self):
        return GeneformerWrapper(self.cfg)

    def forward_step(self, model, batch):
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        emb = model(input_ids, attn_mask)
        cid = batch["cid"]
        return emb, cid
