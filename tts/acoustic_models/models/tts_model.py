import typing as tp

from copy import deepcopy as copy

import torch
import numpy.typing as npt

from speechflow.data_pipeline.datasample_processors.spectrogram_processors import (
    MelProcessor,
)
from speechflow.io import Config
from speechflow.training.base_model import BaseTorchModel
from speechflow.training.utils.tensor_utils import get_mask_from_lengths
from speechflow.utils.dictutils import find_field
from tts.acoustic_models.data_types import TTSForwardInput, TTSForwardOutput
from tts.acoustic_models.modules import (
    PARALLEL_ADAPTORS,
    PARALLEL_DECODERS,
    PARALLEL_ENCODERS,
    PARALLEL_POSTNETS,
)
from tts.acoustic_models.modules.additional_modules import (
    AdditionalModules,
    AdditionalModulesParams,
)
from tts.acoustic_models.modules.embedding import EmbeddingComponent
from tts.acoustic_models.modules.modifier import Mode, ModeStage
from tts.acoustic_models.modules.params import *

__all__ = ["ParallelTTSModel", "ParallelTTSParams"]


class ParallelTTSParams(
    ModifierParams,
    EncoderParams,
    VarianceAdaptorParams,
    DecoderParams,
    PostnetParams,
    AdditionalModulesParams,
):
    """Parallel TTS model parameters."""

    def model_post_init(self, __context: tp.Any):
        super(EmbeddingParams, self).model_post_init(__context)
        super(ModifierParams, self).model_post_init(__context)
        super(EncoderParams, self).model_post_init(__context)
        super(VariancePredictorParams, self).model_post_init(__context)
        super(VarianceAdaptorParams, self).model_post_init(__context)
        super(DecoderParams, self).model_post_init(__context)
        super(PostnetParams, self).model_post_init(__context)
        super(AdditionalModulesParams, self).model_post_init(__context)


class ParallelTTSModel(BaseTorchModel):
    params: ParallelTTSParams

    def __init__(
        self,
        params: tp.Union[tp.MutableMapping, ParallelTTSParams],
        strict_init: bool = True,
    ):
        super().__init__(ParallelTTSParams.create(params, strict_init))
        params = self.params

        self.embedding_component = EmbeddingComponent(params)

        self.mode_0 = Mode(
            params,
            input_dim=self.embedding_component.output_dim,
            current_pos=ModeStage.s_0,
        )

        cls, params_cls = PARALLEL_ENCODERS[params.encoder_type]
        encoder_params = params_cls.init_from_parent_params(params, params.encoder_params)
        self.encoder = cls(encoder_params, input_dim=self.mode_0.output_dim)

        self.mode_1 = Mode(
            params, input_dim=self.encoder.output_dim, current_pos=ModeStage.s_1  # type: ignore
        )

        self.va = torch.nn.ModuleList()
        va_variances = list(params.va_variances.values())
        input_dim = self.mode_1.output_dim
        if va_variances:
            for var_names in va_variances:
                if var_names:
                    va_params = copy(params)
                    va_params.va_variances = var_names  # type: ignore
                    adaptor, _ = PARALLEL_ADAPTORS[va_params.va_type]
                    self.va.append(adaptor(va_params, input_dim=input_dim))
                    input_dim = self.va[-1].output_dim
        else:
            adaptor, _ = PARALLEL_ADAPTORS["DummyVarianceAdaptor"]
            self.va.append(adaptor(params, input_dim=input_dim))

        self.mode_2 = Mode(
            params, input_dim=self.va[-1].output_dim[0], current_pos=ModeStage.s_2
        )

        cls, params_cls = PARALLEL_DECODERS[params.decoder_type]
        decoder_params = params_cls.init_from_parent_params(params, params.decoder_params)
        self.decoder = cls(decoder_params, input_dim=self.mode_2.output_dim)

        self.mode_3 = Mode(
            params, input_dim=self.decoder.output_dim, current_pos=ModeStage.s_3  # type: ignore
        )

        if params.postnet_type is not None:
            cls, params_cls = PARALLEL_POSTNETS[params.postnet_type]
            postnet_params = params_cls.init_from_parent_params(
                params, params.postnet_params
            )
            self.postnet = cls(postnet_params, input_dim=self.mode_3.output_dim)  # type: ignore
            output_dim = self.postnet.output_dim
        else:
            self.postnet = None
            output_dim = self.decoder.output_dim

        self.additional_modules = AdditionalModules(params, input_dim=output_dim)

    @property
    def device(self) -> torch.device:
        return self.embedding_component.emb_calculator.embedding.weight.device

    def _encode_inputs(self, inputs, generate: bool = False):
        """Utility method that reduces code duplication."""
        if inputs.additional_inputs is None:
            inputs.additional_inputs = {}

        x = self.embedding_component(inputs)

        if generate:
            x = self.mode_0.generate(x)
            x = self.encoder.generate(x)  # type: ignore
        else:
            x = self.mode_0(x)
            x = self.encoder(x)  # type: ignore

        if generate:
            x = self.mode_1.generate(x)
        else:
            x = self.mode_1(x)

        return x

    def _predict_variances(
        self, x, generate: bool = False, ignored_variance: tp.Set = None
    ):
        """Utility method that reduces code duplication."""
        predictions = {}
        for va in self.va:
            x = (
                va(x)
                if not generate
                else va.generate(x, ignored_variance=ignored_variance)
            )
            predictions.update(x.variance_predictions)
            x.model_inputs.additional_inputs.update(x.additional_content)

        return x, predictions

    def forward(self, inputs: TTSForwardInput) -> TTSForwardOutput:
        x = self._encode_inputs(inputs)

        va_output, variance_predictions = self._predict_variances(x)

        x = self.mode_2(va_output)
        x = self.decoder(x)  # type: ignore

        s_all = x.stack_content()
        decoder_output = x

        if self.postnet is not None:
            x = self.mode_3(x)
            x = self.postnet(x)
            if x.content is not None:
                s_all = torch.cat([s_all, x.content.unsqueeze(0)], dim=0)

        x = self.additional_modules(x)

        output = TTSForwardOutput(
            spectrogram=s_all,
            spectrogram_lengths=x.content_lengths,
            after_postnet_spectrogram=x.content,
            variance_predictions=variance_predictions,
            masks=va_output.masks,
            gate=decoder_output.gate,
            additional_content=x.additional_content,
            additional_losses=x.additional_losses,
            embeddings=x.embeddings,
        )
        return output

    def generate(self, inputs: TTSForwardInput, **kwargs) -> TTSForwardOutput:
        x = self._encode_inputs(inputs, generate=True)

        va_output, variance_predictions = self._predict_variances(
            x, generate=True, ignored_variance=kwargs.get("ignored_variance")
        )

        x = self.mode_2.generate(va_output)
        x = self.decoder.generate(x)  # type: ignore

        decoder_output = x

        if self.postnet is not None:
            x = self.mode_3(x)
            x = self.postnet.generate(x)

        output = TTSForwardOutput(
            spectrogram=x.content,
            spectrogram_lengths=x.content_lengths,
            after_postnet_spectrogram=x.content,
            variance_predictions=variance_predictions,
            masks=va_output.masks,
            gate=decoder_output.gate,
            additional_content=x.additional_content,
            additional_losses=x.additional_losses,
            embeddings=x.embeddings,
        )
        return output

    def get_variance(
        self, inputs, ignored_variance: tp.Set = None
    ) -> tp.Tuple[
        tp.Dict[str, torch.Tensor], tp.Dict[str, torch.Tensor], tp.Dict[str, torch.Tensor]
    ]:
        x = self._encode_inputs(inputs, generate=True)
        va_output, variance_predictions = self._predict_variances(
            x, generate=True, ignored_variance=ignored_variance
        )
        return va_output, variance_predictions, x.additional_content

    def get_speaker_emb(
        self, speaker_id: int, bio_emb: npt.NDArray, mean_bio_emb: npt.NDArray
    ):
        if self.params.use_learnable_speaker_emb and speaker_id is not None:
            sp_id = torch.LongTensor([speaker_id]).to(self.device)
            emb = self.embedding_component.emb_calculator.speaker_emb(sp_id)
        elif self.params.use_dnn_speaker_emb and bio_emb is not None:
            sp_emb = torch.from_numpy(bio_emb).to(self.device)
            emb = self.embedding_component.emb_calculator.speaker_emb_proj(
                sp_emb.unsqueeze(0)
            )
        elif self.params.use_mean_dnn_speaker_emb and mean_bio_emb is not None:
            sp_emb = torch.from_numpy(mean_bio_emb).to(self.device)
            emb = self.embedding_component.emb_calculator.speaker_emb_proj(
                sp_emb
            ).unsqueeze(0)
        else:
            raise NotImplementedError

        return {"speaker": emb.cpu().numpy()}

    def get_style_embedding(
        self,
        bio_embedding: torch.Tensor,
        spectrogram: torch.Tensor,
        ssl_embeddings: torch.Tensor,
    ):
        def get_sample(_sampler, feat_type, input_feat):
            input_feat = torch.from_numpy(input_feat).to(self.device)
            if input_feat.ndim == 1:
                input_feat = input_feat.unsqueeze(0).unsqueeze(0)
            if input_feat.ndim == 2:
                input_feat = input_feat.unsqueeze(0)
            if "ssl" in feat_type:
                input_feat = self.embedding_component.emb_calculator.ssl_proj(input_feat)

            lens = torch.LongTensor([input_feat.shape[1]]).to(input_feat.device)
            try:
                emb = _sampler.encode(input_feat, get_mask_from_lengths(lens))
            except:
                emb = _sampler.sample_latent(input_feat, input_feat, None, None, None)

            if isinstance(emb, tuple):
                emb = emb[0]
            if emb.ndim == 3:
                emb = emb.squeeze(1)
            return emb

        for k, module in self._modules["va"]._modules.items():
            if "speaker_emb_encoder" in module.va_variances:
                break
            if any("style" in name for name in module.va_variances):
                break
        else:
            module = None

        if module is not None:
            sampler = list(module._modules["predictors"].values())[0]
            if sampler.__class__.__name__ == "GMVariationalAutoencoder":
                return {"style_emb": get_sample(sampler, "bio_embedding", bio_embedding)}
            elif "spec" in list(module.predictors.values())[0].params.source:
                return {"style_emb": get_sample(sampler, "spectrogram", spectrogram)}
            elif "ssl" in list(module.predictors.values())[0].params.source:
                return {
                    "style_emb": get_sample(sampler, "ssl_embeddings", ssl_embeddings)
                }
            else:
                return {"style_emb": get_sample(sampler, "spectrogram", spectrogram)}

        return {}

    @classmethod
    def update_and_validate_model_params(cls, cfg_model: Config, cfg_data: Config):
        if "speaker_biometric_model" not in cfg_model["net"]["params"]:
            cfg_model["net"]["params"].speaker_biometric_model = find_field(
                cfg_data["preproc"], "voice_bio.model_type", "resemblyzer"
            )

        if (
            cfg_model["net"]["params"].get("decoder_target", "spectrogram")
            == "spectrogram"
        ):
            cfg_model["net"]["params"].decoder_output_dim = find_field(
                cfg_data["preproc"], "linear_to_mel.n_mels"
            )

        if "SSIM" in cfg_model["loss"]:
            max_abs_value = find_field(cfg_data["preproc"], "normalize.max_abs_value")
            melscale_pipe = find_field(cfg_data["preproc"], "melscale.pipe", [])
            if max_abs_value is None and "normalize" in melscale_pipe:
                max_abs_value = MelProcessor().max_abs_value

            if max_abs_value:
                min_value = cfg_model["loss"]["SSIM"].get("min_value", 0)
                max_value = cfg_model["loss"]["SSIM"].get("max_value", 0)
                if abs(min_value) != max_abs_value or abs(max_value) != max_abs_value:
                    raise ValueError("SSIM configuration is not valid!")

        return cfg_model

    def load_params(self, state_dict: tp.Dict[str, torch.Tensor], *args):
        return super().load_params(state_dict, args)
