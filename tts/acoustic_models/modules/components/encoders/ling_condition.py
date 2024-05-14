import torch

from tts.acoustic_models.modules.common import SoftLengthRegulator
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, ComponentOutput
from tts.acoustic_models.modules.params import EncoderParams

__all__ = ["LinguisticCondition", "LinguisticConditionParams"]


class LinguisticConditionParams(EncoderParams):
    cat_ling_feat: bool = False
    cat_lm_feat: bool = False
    p_dropout: float = 0.1


class LinguisticCondition(Component):
    params: LinguisticConditionParams

    def __init__(self, params: LinguisticConditionParams, input_dim):
        super().__init__(params, input_dim)

        self.hard_lr = SoftLengthRegulator(hard=True)
        self.seq_dropout = torch.nn.Dropout1d(params.p_dropout)

    @property
    def output_dim(self):
        dim = self.input_dim
        if self.params.cat_ling_feat:
            dim += self.input_dim
        if self.params.cat_lm_feat:
            dim += self.params.lm_proj_dim
        return dim

    def add_ling_features(self, x: torch.Tensor, inputs: ComponentInput):
        if self.params.cat_ling_feat:
            ling_feat = inputs.embeddings["ling_feat"]
            ling_feat = self.seq_dropout(ling_feat)
            x = torch.cat([x, ling_feat], dim=2)

        if self.params.cat_lm_feat:
            lm_emb = inputs.embeddings["lm_feat"]
            if lm_emb.shape[1] != x.shape[1]:
                token_length = inputs.model_inputs.additional_inputs["token_lengths"]
                lm_emb, _ = self.hard_lr(lm_emb, token_length, x.shape[1])

            lm_emb = self.seq_dropout(lm_emb)
            x = torch.cat([x, lm_emb], dim=2)

        return x

    def forward_step(self, inputs: ComponentInput) -> ComponentOutput:
        x, x_lens, x_mask = self.get_content_and_mask(inputs)

        y = self.add_ling_features(x, inputs)

        return ComponentOutput.copy_from(inputs).set_content(y).apply_mask(x_mask)
