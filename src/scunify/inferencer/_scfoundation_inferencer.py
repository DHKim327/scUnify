from ..registry.dataset import ScFoundationDataset
from .base._baseinferencer import BaseInferencer
from .models import ScFoundationInferenceWrapper


class ScFoundationInferencer(BaseInferencer):
    def build_dataset(self, adata):
        return ScFoundationDataset(adata, self.cfg)

    def build_model(self):
        return ScFoundationInferenceWrapper(self.cfg)

    def forward_step(self, model, batch):
        values, pad, pos, cid = batch
        emb = model(values, pad, pos)
        return emb, cid
