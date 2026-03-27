from ..registry.dataset import UCEDataset
from .base._baseinferencer import BaseInferencer
from .models import UCEInferenceWrapper


class UCEInferencer(BaseInferencer):
    def build_dataset(self, adata):
        return UCEDataset(adata, self.cfg)

    def build_model(self):
        return UCEInferenceWrapper(self.cfg)

    def forward_step(self, model, batch):
        batch_sentences, mask, cid, _ = batch
        emb = model(batch_sentences, mask)
        return emb, cid
