from ..registry.dataset import UCEDataset
from ..registry.models import UCEWrapper
from .base._baseinferencer import BaseInferencer


class UCEInferencer(BaseInferencer):
    def build_dataset(self, adata):
        return UCEDataset(adata, self.cfg)

    def build_model(self):
        return UCEWrapper(self.cfg)

    def forward_step(self, model, batch):
        batch_sentences, mask, cid, _ = batch
        emb = model(batch_sentences, mask)
        return emb, cid
