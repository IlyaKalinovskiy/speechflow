import math
import typing as tp

from collections import namedtuple

import torch.nn

from torch.nn import functional as F

from speechflow.training.utils.tensor_utils import apply_mask, get_mask_from_lengths
from tts.acoustic_models.modules.common.matcha_tts.flow_matching import CFM
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import DecoderOutput, VarianceAdaptorOutput
from tts.acoustic_models.modules.params import DecoderParams

__all__ = [
    "CFMDecoder",
    "CFMDecoderParams",
]


class CFMDecoderParams(DecoderParams):
    with_prior_loss: bool = True
    prior_decoder_type: str = None  # type: ignore
    prior_decoder_params: dict = None  # type: ignore
    prior_decoder_num_layers: int = 2
    prior_decoder_hidden_dim: int = 512
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0
    speaker_emb_dim: int = 0
    n_timesteps: int = 10
    temperature: float = 0.667
    p_dropout: float = 0.05
    attention_head_dim: int = 64
    n_blocks: int = 1
    n_mid_blocks: int = 2
    n_heads: int = 2
    norm_type: str = "layer_norm"
    act_fn: str = "snakebeta"


class CFMDecoder(Component):
    """Decoder from Matcha-TTS paper https://browse.arxiv.org/pdf/2309.03199.pdf."""

    params: CFMDecoderParams

    def __init__(self, params: CFMDecoderParams, input_dim):
        super().__init__(params, input_dim)

        cfm_params = namedtuple("cfm_params", ["solver", "sigma_min"])("euler", 1e-4)
        decoder_params = {
            "channels": [params.decoder_inner_dim] * params.decoder_num_layers,
            "dropout": params.p_dropout,
            "attention_head_dim": params.attention_head_dim,
            "n_blocks": params.n_blocks,
            "n_mid_blocks": params.n_mid_blocks,
            "n_heads": params.n_heads,
            "norm_type": params.norm_type,
            "act_fn": params.act_fn,
        }

        if self.params.prior_decoder_type:
            from tts.acoustic_models.modules.components import encoders

            dec_cls = getattr(encoders, params.prior_decoder_type)
            dec_params_cls = getattr(encoders, f"{params.prior_decoder_type}Params")
            dec_params = dec_params_cls.init_from_parent_params(
                params, params.prior_decoder_params
            )

            dec_params.encoder_num_layers = params.prior_decoder_num_layers
            dec_params.encoder_inner_dim = params.prior_decoder_hidden_dim
            dec_params.encoder_output_dim = params.decoder_output_dim

            self.prior_decoder = dec_cls(dec_params, input_dim)
        else:
            if params.with_prior_loss and input_dim != params.decoder_output_dim:
                self.prior_decoder = torch.nn.Linear(input_dim, params.decoder_output_dim)
            else:
                self.prior_decoder = torch.nn.Identity()

        self.decoder = CFM(
            in_channels=params.decoder_output_dim * 2,
            out_channel=params.decoder_output_dim,
            cfm_params=cfm_params,
            decoder_params=decoder_params,
            speaker_emb_dim=params.speaker_emb_dim,
            cond_dim=params.condition_dim,
        )

    @property
    def output_dim(self):
        return self.params.decoder_output_dim

    def forward_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        content = self.get_content(inputs)[0]
        content_lengths = self.get_content_lengths(inputs)[0]

        y = getattr(inputs.model_inputs, self.params.decoder_target)
        y_mask = get_mask_from_lengths(content_lengths)

        if self.params.prior_decoder_type:
            prior, _ = self.prior_decoder.process_content(
                content, content_lengths, inputs.model_inputs
            )
        else:
            prior = self.prior_decoder(content)

        if self.params.with_prior_loss:
            prior_loss = torch.sum(
                0.5 * apply_mask((prior - y) ** 2 + math.log(2 * math.pi), y_mask)
            ) / (torch.sum(y_mask) * self.params.decoder_output_dim)
            inputs.additional_losses["prior_loss"] = prior_loss  # type: ignore

        pad = 16 - prior.shape[1] % 16
        if pad != 16:
            prior = F.pad(prior.transpose(2, 1), (0, pad), value=-4).transpose(1, 2)
            y = F.pad(y.transpose(2, 1), (0, pad), value=-4).transpose(1, 2)
            y_mask = F.pad(y_mask | True, (0, pad), value=True)

        if self.params.condition:
            cond = self.get_condition(inputs, self.params.condition)
            cond = cond.squeeze(1)
        else:
            cond = None

        if self.params.speaker_emb_dim > 0:
            spks = inputs.embeddings["speaker"]
        else:
            spks = None

        cfm_loss, _ = self.decoder.compute_loss(
            x1=y.transpose(2, 1),
            mask=y_mask,
            mu=prior.transpose(2, 1),
            spks=spks,
            cond=cond,
        )
        inputs.additional_losses["cfm_loss"] = cfm_loss

        if pad != 16:
            prior = prior[:, :-pad, :]

        return DecoderOutput.copy_from(inputs).set_content(prior, content_lengths)

    def generate_step(self, inputs: VarianceAdaptorOutput, **kwargs) -> DecoderOutput:  # type: ignore
        content = self.get_content(inputs)[0]
        content_lengths = self.get_content_lengths(inputs)[0]
        content_mask = get_mask_from_lengths(content_lengths)

        if self.params.prior_decoder_type:
            prior, _ = self.prior_decoder.process_content(
                content, content_lengths, inputs.model_inputs
            )
        else:
            prior = self.prior_decoder(content)

        pad = 16 - prior.shape[1] % 16
        if pad != 16:
            prior = F.pad(prior.transpose(2, 1), (0, pad), value=-4).transpose(1, 2)
            content_mask = F.pad(content_mask | True, (0, pad), value=True)

        if self.params.condition:
            cond = self.get_condition(inputs, self.params.condition)
            cond = cond.squeeze(1)
        else:
            cond = None

        if self.params.speaker_emb_dim > 0:
            spks = inputs.embeddings["speaker"]
        else:
            spks = None

        n_timesteps = self.params.n_timesteps
        temperature = self.params.temperature

        decoder_outputs = self.decoder(
            prior.transpose(2, 1),
            content_mask,
            n_timesteps,
            temperature,
            spks=spks,
            cond=cond,
        )

        content = decoder_outputs.transpose(2, 1)
        if pad != 16:
            content = content[:, :-pad, :]

        outputs = DecoderOutput.copy_from(inputs).set_content(content, content_lengths)
        return outputs
