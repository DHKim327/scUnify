from ..registry.dataset import NicheformerDataset
from ..registry.models import NicheformerWrapper
from .base._baseinferencer import BaseInferencer


class NicheformerInferencer(BaseInferencer):
    def build_dataset(self, adata):
        return NicheformerDataset(adata, self.cfg)

    def build_model(self):
        return NicheformerWrapper(self.cfg)

    def forward_step(self, model, batch):
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        emb = model(input_ids, attn_mask)
        cid = batch["cid"]
        return emb, cid
