import torch.nn as nn

from ...registry.models import UCEWrapper


class UCEInferenceWrapper(UCEWrapper):
    """Inference wrapper — pe_embedding + encode → cell embedding."""

    def forward(self, batch_sentences, mask):
        batch_sentences = batch_sentences.permute(1, 0)
        batch_sentences = self.pe_embedding(batch_sentences.long())
        batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
        _, embedding = self.encoder(batch_sentences, mask=mask)
        return embedding
