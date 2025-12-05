from ..registry.dataset import ScFoundationDataset
from ..registry.models import ScFoundationWrapper
from .base._baseinferencer import BaseInferencer


class ScFoundationInferencer(BaseInferencer):
    def build_dataset(self, adata):
        return ScFoundationDataset(adata, self.cfg)

    def build_model(self):
        return ScFoundationWrapper(self.cfg)

    def forward_step(self, model, batch):
        values, pad, pos, cid = batch
        emb = model(values, pad, pos)
        return emb, cid
