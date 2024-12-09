import typing as tp

from copy import deepcopy as copy

from pydantic import Field

from speechflow.data_pipeline.collate_functions.tts_collate import LinguisticFeatures
from speechflow.training.base_model import BaseTorchModelParams

__all__ = [
    "EncoderParams",
    "VarianceParams",
    "VariancePredictorParams",
    "VarianceAdaptorParams",
    "DecoderParams",
    "ModifierParams",
    "EmbeddingParams",
    "PostnetParams",
]

tp_TEXT_FEATURES = tp.Literal["transcription", "lm_feat"]
tp_AUDIO_FEATURES = tp.Literal["waveform", "spectrogram", "ssl_feat", "ac_feat"]
tp_BIOMETRIC_MODELS = tp.Literal["resemblyzer", "speechbrain", "wespeaker"]


class EmbeddingParams(BaseTorchModelParams):
    """Embedding component parameters."""

    input: tp.Union[tp_TEXT_FEATURES, tp_AUDIO_FEATURES] = "transcription"
    target: tp_AUDIO_FEATURES = "spectrogram"

    # Transcription embeddings parameters
    alphabet_size: int = Field(ge=1, default=1)
    n_symbols_per_token: int = Field(ge=1, default=1)
    token_emb_dim: int = 256

    # Speaker embeddings parameters
    n_langs: int = Field(ge=1, default=1)
    n_speakers: int = Field(ge=1, default=1)
    use_onehot_speaker_emb: bool = False
    use_learnable_speaker_emb: bool = False
    use_dnn_speaker_emb: bool = False
    use_mean_dnn_speaker_emb: bool = False
    speaker_emb_dim: int = 256
    speaker_biometric_model: tp_BIOMETRIC_MODELS = "resemblyzer"

    # Linguistic sequences
    num_additional_integer_seqs: int = -1
    num_additional_float_seqs: int = -1

    # LM features parameters
    lm_feat_dim: int = 1024
    lm_feat_proj_dim: int = 256
    plbert_feat_dim: int = 768
    plbert_feat_proj_dim: int = 256

    # spectrogram parameters
    linear_spectrogram_dim: int = 513
    linear_spectrogram_proj_dim: int = 513
    mel_spectrogram_dim: int = 80
    mel_spectrogram_proj_dim: int = 80

    # SSL features parameters
    ssl_feat_dim: int = 1024
    ssl_feat_proj_dim: int = 1024

    # AC features parameters
    ac_feat_dim: int = 1024
    ac_feat_proj_dim: int = 1024

    # Average embeddings parameters
    use_average_emb: bool = False
    average_emb_dim: int = 0
    averages: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})

    # Speech quality embeddings parameters
    speech_quality_emb_dim: int = 4

    max_input_length: int = 0
    max_output_length: int = 0

    def model_post_init(self, __context: tp.Any):
        super().model_post_init(__context)

        if self.input != "transcription":
            self.max_input_length = self.max_output_length

        if (
            self.use_onehot_speaker_emb
            + self.use_learnable_speaker_emb
            + self.use_dnn_speaker_emb
            + self.use_mean_dnn_speaker_emb
        ) > 1:
            raise AttributeError(
                "Cannot perform with both types of embeddings, choose bio/learnable"
            )

        if self.num_additional_integer_seqs < 0:
            self.num_additional_integer_seqs = LinguisticFeatures.num_integer_features()
        if self.num_additional_float_seqs < 0:
            self.num_additional_float_seqs = LinguisticFeatures.num_float_features()

        if self.use_average_emb:
            for name in self.averages.keys():
                self.averages[name].setdefault("n_bins", 64)
                self.averages[name].setdefault("emb_dim", 16)
                self.averages[name].setdefault("log_scale", False)

            if not self.average_emb_dim:
                dim = 0
                for params in self.averages.values():
                    dim += params["emb_dim"]
                self.average_emb_dim = dim

    def get_feat_dim(self, feat_name: str) -> int:
        if hasattr(self, f"{feat_name}_dim"):
            return getattr(self, f"{feat_name}_dim")
        else:
            raise RuntimeError(f"Dim for {feat_name} not found")

    @staticmethod
    def check_deprecated_params(cfg: dict) -> dict:
        if "n_symbols" in cfg:
            cfg["alphabet_size"] = cfg.pop("n_symbols")
        if "input" in cfg:
            if cfg["input"] == "mel_spectrogram":
                cfg["input"] = "spectrogram"
        if "target" in cfg:
            if cfg["target"] == "mel_spectrogram":
                cfg["target"] = "spectrogram"

        return cfg


class EncoderParams(EmbeddingParams):
    """Encoder component parameters."""

    encoder_type: str = "ForwardEncoder"
    encoder_num_layers: int = 2
    encoder_inner_dim: int = 512
    encoder_output_dim: int = 512
    encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})


class DecoderParams(EmbeddingParams):
    """Decoder component parameters."""

    decoder_type: str = "ForwardDecoder"
    decoder_num_layers: int = 2
    decoder_inner_dim: int = 512
    decoder_output_dim: int = 80
    decoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})


class PostnetParams(EmbeddingParams):
    """Postnet component parameters."""

    postnet_type: tp.Optional[str] = "ForwardPostnet"
    postnet_num_layers: int = 1
    postnet_inner_dim: int = 512
    postnet_output_dim: int = 80
    postnet_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})


class ModifierParams(EmbeddingParams):
    """Modifier parameters."""

    mode_add: tp.Dict[int, tp.List[str]] = Field(default_factory=lambda: {})
    mode_cat: tp.Dict[int, tp.List[str]] = Field(default_factory=lambda: {})

    def model_post_init(self, __context: tp.Any):
        super().model_post_init(__context)

        from tts.acoustic_models.modules.modifier import ModeStage

        for att_name in ["mode_add", "mode_cat"]:
            attr = getattr(self, att_name)
            if attr is not None:
                if not isinstance(attr, tp.MutableMapping):
                    raise ValueError(
                        f"Invalid modifier parameter. It should be either dict but got {attr} instead."
                    )
                mode_stages = {item.value for item in ModeStage}
                if not mode_stages > set(attr.keys()):
                    raise ValueError(
                        f"Invalid stage number in ModifierParams {set(attr.keys())}. "
                        f"Only integer values from {min(mode_stages)} to {max(mode_stages)} are allowed."
                    )


class VariancePredictorParams(EmbeddingParams):
    vp_num_layers: int = 1
    vp_inner_dim: int = 256
    vp_output_dim: int = 1
    vp_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})


class VarianceParams(BaseTorchModelParams):
    predictor_type: str = "CNNPredictor"
    predictor_params: VariancePredictorParams = None  # type: ignore
    dim: int = 1
    target: str = None
    input_content: tp.Tuple[int, ...] = (0,)
    input_content_dim: tp.Tuple[int, ...] = None
    detach_input: bool = False
    detach_output: bool = True
    use_target: bool = True
    denormalize: bool = False
    upsample: bool = False
    cat_to_content: tp.Tuple[int, ...] = (0, 1)
    overwrite_content: tp.Tuple[int, ...] = ()
    as_encoder: bool = False
    as_embedding: bool = False
    interval: tp.Tuple[float, float] = (0.0, 1.0)
    log_scale: bool = False
    n_bins: int = 256
    emb_dim: int = 128
    begin_iter: int = 0
    end_iter: int = 1_000_000
    skip: bool = False
    use_loss: bool = False
    loss_type: str = "l1_loss"

    def model_post_init(self, __context: tp.Any):
        super().model_post_init(__context)

        if self.predictor_params is None:
            self.predictor_params = VariancePredictorParams()

        self.input_content = tuple(self.input_content)
        self.cat_to_content = tuple(self.cat_to_content)
        self.overwrite_content = tuple(self.overwrite_content)

        if self.overwrite_content and len(self.cat_to_content) == 2:
            self.cat_to_content = ()

        if self.as_encoder:
            self.dim = self.predictor_params.vp_output_dim
            self.use_target = False
            self.detach_output = False
            self.as_embedding = False
            self.use_loss = False
        else:
            self.predictor_params.vp_output_dim = self.dim


class VarianceAdaptorParams(VariancePredictorParams):
    va_type: str = "VarianceAdaptor"
    va_length_regulator_type: str = "SoftLengthRegulator"  # "LengthRegulator"
    va_variances: tp.Dict[int, tp.Tuple[str, ...]] = None  # type: ignore
    va_variance_params: tp.Dict[str, VarianceParams] = None  # type: ignore

    def model_post_init(self, __context: tp.Any):
        super().model_post_init(__context)

        if self.va_variances is None:
            self.va_variances = {}
            self.va_variance_params = {}
            return

        if isinstance(self.va_variances, (tuple, set, list)):
            self.va_variances = {0: tuple(self.va_variances)}

        all_va_variances = [t for v in self.va_variances.values() for t in v]
        if len(all_va_variances) != len(set(all_va_variances)):
            raise ValueError("Variances at each level must be unique.")

        vp_global_params = {
            k: v
            for k, v in self.to_dict().items()
            if k in VariancePredictorParams().to_dict()
        }
        variance_params: tp.Dict[str, VarianceParams] = {
            name: VarianceParams() for name in all_va_variances
        }
        if self.va_variance_params:
            for name, params in self.va_variance_params.items():
                variance_params[name] = (
                    VarianceParams(**params) if isinstance(params, dict) else params
                )
                vp_custom_params = variance_params[name].predictor_params
                if vp_custom_params is None:
                    variance_params[name].predictor_params = VariancePredictorParams(
                        **vp_global_params
                    )
                elif isinstance(vp_custom_params, dict):
                    vp_custom_params = copy(vp_global_params)
                    vp_custom_params.update(variance_params[name].predictor_params)
                    variance_params[name].predictor_params = VariancePredictorParams(
                        **vp_custom_params
                    )

        self.va_variance_params = variance_params
