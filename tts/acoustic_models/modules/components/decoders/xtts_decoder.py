import typing as tp

import torch

from torch import Tensor, nn
from torch.nn import functional as F

from speechflow.utils.tensor_utils import get_mask_from_lengths
from tts.acoustic_models.modules.common.blocks import ConvPrenet
from tts.acoustic_models.modules.common.gpts.gpt_acoustic import GPTA
from tts.acoustic_models.modules.common.gpts.layers.modules import make_pad_mask
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import DecoderOutput, VarianceAdaptorOutput
from tts.acoustic_models.modules.params import DecoderParams

__all__ = [
    "XTTSDecoder",
    "XTTSDecoderParams",
]


class XTTSDecoderParams(DecoderParams):
    audio_feat: str = "spectrogram"
    audio_feat_dim: int = 80
    audio_feat_proj_dim: int = 80
    n_heads: int = 8
    n_layers: int = 12
    n_tokens: int = 1024
    p_dropout: float = 0.1
    use_prenet: bool = True
    decoder_name: str = "gpt"


class XTTSDecoder(Component):
    params: XTTSDecoderParams

    def __init__(self, params: XTTSDecoderParams, input_dim):
        super().__init__(params, input_dim)

        if self.params.audio_feat_dim != self.params.audio_feat_proj_dim:
            self.audio_feat_proj = nn.Linear(
                self.params.audio_feat_dim, self.params.audio_feat_proj_dim
            )
        else:
            self.audio_feat_proj = None

        self.prenet_layer = ConvPrenet(
            in_channels=input_dim,
            out_channels=params.decoder_inner_dim,
        )
        self.gpt = GPTA(
            dim_hidden=params.decoder_inner_dim,
            n_heads=params.n_heads,
            n_layers=params.n_layers,
            dim_prompt_text=params.decoder_inner_dim,
            dim_prompt_audio=params.audio_feat_proj_dim,
            use_prenet=params.use_prenet,
            num_tokens_response=params.n_tokens,
            decoder_name=params.decoder_name,
        )
        self.linear = nn.Linear(params.decoder_inner_dim, params.n_tokens + 1)

        self._ignore_index = -100

    @property
    def output_dim(self):
        return self.params.decoder_output_dim

    def _encode(self, waves: tp.List[Tensor]) -> tp.List[Tensor]:
        """
        return [1, 64, length] * batch_size or [1, 512, length] if
        is_proj is False for audiodec
        """
        with torch.no_grad():
            return self.encoder(waves)

    def _quantize(self, embs: tp.List[Tensor]) -> tp.List[Tensor]:
        """
        embs [1, 64, length] * batch_size
        """
        if not self.encoder.is_use_proj:
            embs = self.encoder._proj(embs)

        return self.encoder.quantize(embs)

    def _qencode(self, waves: tp.List[Tensor]) -> tp.List[Tensor]:
        embs = self._encode(waves)
        quants = self._quantize(embs)

        return quants

    def forward_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        x, x_lens, x_mask = self.get_content_and_mask(inputs)
        x = self.prenet_layer(x.transpose(1, -1)).transpose(1, -1)

        codes = inputs.model_inputs.ac_feat
        codes_lens = inputs.model_inputs.ac_feat_lengths

        _prompt = getattr(inputs.model_inputs, "prompt", None)
        if _prompt is None:
            _prompt = inputs.model_inputs

        _prompt_audio = getattr(_prompt, self.params.audio_feat)
        _prompt_audio_lens = getattr(_prompt, f"{self.params.audio_feat}_lengths")

        output = self.gpt(
            prompt_text=x,
            prompt_text_lens=x_lens,
            prompt_audio=_prompt_audio,
            prompt_audio_lens=_prompt_audio_lens,
            response=codes[:, :, :1].transpose(1, -1),
            response_lens=codes_lens,
        )

        start_resp = output.prompt_lens.max()
        pred = output.emb[:, start_resp:-1]
        logits = self.linear(pred).transpose(1, 2)

        targets = output.target.squeeze(1)
        targets_mask = make_pad_mask(output.target_lens - 1)
        targets.masked_fill_(targets_mask, self._ignore_index)

        loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=self._ignore_index,  # internal cuda error, it's pytorch bug
            reduction="sum",
        ) / torch.sum(codes_lens)

        inputs.additional_losses.update({"loss_gpt": loss})

        return DecoderOutput.copy_from(inputs).set_content(pred, codes_lens + 1)

    def generate_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        content = self.get_content(inputs)[0]
        content_lengths = self.get_content_lengths(inputs)[0]
        content_mask = get_mask_from_lengths(content_lengths)

        if self.params.prior_decoder_type:
            prior = self.prior_decoder.encode(content, content_lengths, inputs)
            prior = self.prior_decoder.proj(prior)
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
