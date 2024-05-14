import typing as tp

from collections import namedtuple

from torch.nn import functional as F

from speechflow.training.utils.tensor_utils import get_mask_from_lengths
from tts.acoustic_models.modules.common.matcha_tts.flow_matching import CFM
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import DecoderOutput, VarianceAdaptorOutput
from tts.acoustic_models.modules.params import PostnetParams

__all__ = [
    "CFMPostnet",
    "CFMPostnetParams",
]


class CFMPostnetParams(PostnetParams):
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0
    n_timesteps: int = 10
    temperature: float = 0.667
    p_dropout: float = 0.05
    attention_head_dim: int = 64
    n_blocks: int = 1
    n_mid_blocks: int = 2
    n_heads: int = 2
    norm_type: str = "layer_norm"
    act_fn: str = "snakebeta"


class CFMPostnet(Component):
    """Postnet from Matcha-TTS paper https://browse.arxiv.org/pdf/2309.03199.pdf."""

    params: CFMPostnetParams

    def __init__(self, params: CFMPostnetParams, input_dim):
        super().__init__(params, input_dim)

        cfm_params = namedtuple("cfm_params", ["solver", "sigma_min"])("euler", 1e-4)
        decoder_params = {
            "channels": [params.postnet_inner_dim] * params.postnet_num_layers,
            "dropout": params.p_dropout,
            "attention_head_dim": params.attention_head_dim,
            "n_blocks": params.n_blocks,
            "n_mid_blocks": params.n_mid_blocks,
            "n_heads": params.n_heads,
            "norm_type": params.norm_type,
            "act_fn": params.act_fn,
        }

        self.decoder = CFM(
            in_channels=params.postnet_output_dim * 2,
            out_channel=params.postnet_output_dim,
            cfm_params=cfm_params,
            decoder_params=decoder_params,
            cond_dim=params.condition_dim,
        )

    @property
    def output_dim(self):
        return self.params.postnet_output_dim

    def forward_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        content = self.get_content(inputs)[0]
        content_lengths = self.get_content_lengths(inputs)[0]
        y = inputs.model_inputs.spectrogram
        y_mask = get_mask_from_lengths(content_lengths)

        pad = 16 - content.shape[1] % 16
        if pad != 16:
            content = F.pad(content.transpose(2, 1), (0, pad), value=-4).transpose(1, 2)
            y = F.pad(y.transpose(2, 1), (0, pad), value=-4).transpose(1, 2)
            y_mask = F.pad(y_mask | True, (0, pad), value=True)

        if self.params.condition:
            cond = self.get_condition(inputs, self.params.condition)
            cond = cond.squeeze(1)
        else:
            cond = None

        cfm_loss, _ = self.decoder.compute_loss(
            x1=y.transpose(2, 1),
            mask=y_mask,
            mu=content.transpose(2, 1),
            spks=None,
            cond=cond,
        )

        inputs.additional_losses["cfm_loss"] = cfm_loss

        outputs = DecoderOutput.copy_from(inputs).set_content(None, content_lengths)
        return outputs

    def generate_step(self, inputs: VarianceAdaptorOutput, **kwargs) -> DecoderOutput:  # type: ignore
        content = self.get_content(inputs)[0]
        content_lengths = self.get_content_lengths(inputs)[0]

        # TODO: исправить ошибку обработки маски в CFM
        content_mask = get_mask_from_lengths(content_lengths) | True
        content = content * (content_mask.unsqueeze(-1)) - 4 * (
            ~content_mask.unsqueeze(-1)
        )
        """
        cat_content = torch.cat([c[: content_lengths[i]] for i, c in enumerate(content)])
        content = cat_content.unsqueeze(0)
        content_mask = get_mask_from_lengths2(content_lengths.sum().unsqueeze(0))
        """

        pad = 16 - content.shape[1] % 16
        if pad != 16:
            content = F.pad(content.transpose(2, 1), (0, pad), value=-4).transpose(1, 2)
            content_mask = F.pad(content_mask, (0, pad), value=True)

        if self.params.condition:
            cond = self.get_condition(inputs, self.params.condition)
            cond = cond.squeeze(1)
        else:
            cond = None

        n_timesteps = self.params.n_timesteps
        temperature = self.params.temperature

        decoder_outputs = self.decoder(
            content.transpose(2, 1),
            content_mask,
            n_timesteps,
            temperature,
            spks=None,
            cond=cond,
        )

        content = decoder_outputs.transpose(2, 1)
        if pad != 16:
            content = content[:, :-pad, :]

        """
        a = 0
        batch = []
        max_len = content_lengths.max()
        for c_len in content_lengths:
            c = content[0, a : a + c_len]
            pad = max_len - c_len
            batch.append(F.pad(c.t(), (0, pad), value=-4).t())
            a += c_len
        content = torch.stack(batch)
        """

        return DecoderOutput.copy_from(inputs).set_content(content, content_lengths)
