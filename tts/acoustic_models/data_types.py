import typing as tp

from dataclasses import dataclass

from torch import Tensor

from speechflow.data_pipeline.collate_functions.tts_collate import LinguisticFeatures
from speechflow.data_pipeline.core.datasample import TrainData
from tts.acoustic_models.models.prosody_reference import ComplexProsodyReference

__all__ = [
    "TTSTarget",
    "TTSForwardInput",
    "TTSForwardInputWithSSML",
    "TTSForwardInputWithPrompt",
    "TTSForwardOutput",
]


@dataclass
class TTSTarget(TrainData):
    transcription: Tensor = None
    spectrogram: Tensor = None
    ssl_feat: Tensor = None
    ac_feat: Tensor = None
    durations: Tensor = None
    energy: Tensor = None
    spectral_flatness: Tensor = None
    spectral_envelope: Tensor = None
    pitch: Tensor = None
    aggregate_mel: Tensor = None
    aggregate_energy: Tensor = None
    aggregate_spectral_flatness: Tensor = None
    aggregate_spectral_envelope: Tensor = None
    aggregate_pitch: Tensor = None
    aggregate_curv_energy: Tensor = None
    aggregate_curv_pitch: Tensor = None
    gate: Tensor = None
    prosody: Tensor = None
    input_lengths: Tensor = None
    output_lengths: Tensor = None
    additional_inputs: tp.Optional[tp.Dict[str, tp.Any]] = None


@dataclass
class TTSForwardInput(TrainData):
    lang_id: Tensor = None
    speaker_id: Tensor = None
    speaker_emb: Tensor = None
    speaker_emb_mean: Tensor = None
    speech_quality_emb: Tensor = None
    waveform: Tensor = None
    waveform_lengths: Tensor = None
    transcription: Tensor = None
    transcription_lengths: Tensor = None
    transcription_by_frames: Tensor = None
    transcription_by_frames_lengths: Tensor = None
    ling_feat: tp.Optional[LinguisticFeatures] = None
    lm_feat: Tensor = None
    linear_spectrogram: Tensor = None
    mel_spectrogram: Tensor = None
    spectrogram_lengths: Tensor = None
    gate: Tensor = None
    ssl_feat: Tensor = None
    ssl_feat_lengths: Tensor = None
    ac_feat: Tensor = None
    ac_feat_lengths: Tensor = None
    durations: Tensor = None
    invert_durations: Tensor = None
    energy: Tensor = None
    spectral_flatness: Tensor = None
    spectral_envelope: Tensor = None
    pitch: Tensor = None
    aggregate_mel: Tensor = None
    aggregate_energy: Tensor = None
    aggregate_spectral_flatness: Tensor = None
    aggregate_spectral_envelope: Tensor = None
    aggregate_pitch: Tensor = None
    aggregate_curv_energy: Tensor = None
    aggregate_curv_pitch: Tensor = None
    averages: tp.Dict[str, Tensor] = None  # type: ignore
    ranges: tp.Dict[str, Tensor] = None  # type: ignore
    synt_lengths: Tensor = None
    num_words: Tensor = None
    word_lengths: Tensor = None
    num_tokens: Tensor = None
    token_lengths: Tensor = None
    prosody: Tensor = None
    input_lengths: Tensor = None
    output_lengths: Tensor = None
    prosody_reference: tp.Optional[ComplexProsodyReference] = None
    additional_inputs: tp.Optional[tp.Dict[str, tp.Any]] = None
    imputer_masks: tp.Optional[tp.Dict[str, Tensor]] = None

    def __post_init__(self):
        if self.additional_inputs is None:
            self.additional_inputs = {}
        if self.imputer_masks is None:
            self.imputer_masks = {}


@dataclass
class TTSForwardInputWithPrompt(TTSForwardInput):
    prompt: TTSForwardInput = None


@dataclass
class TTSForwardInputWithSSML(TTSForwardInput):
    pitch_modifier: Tensor = None
    volume_modifier: Tensor = None
    rate_modifier: Tensor = None


@dataclass
class TTSForwardOutput(TrainData):
    spectrogram: Tensor = None
    spectrogram_lengths: Tensor = None
    after_postnet_spectrogram: Tensor = None
    variance_predictions: tp.Dict[str, Tensor] = None  # type: ignore
    gate: Tensor = None
    attention_weights: Tensor = None
    additional_content: tp.Dict[str, Tensor] = None  # type: ignore
    additional_losses: tp.Dict[str, Tensor] = None  # type: ignore
    embeddings: tp.Dict = None  # type: ignore

    def __post_init__(self):
        if self.variance_predictions is None:
            self.variance_predictions = {}

        if self.additional_content is None:
            self.additional_content = {}

        if self.additional_losses is None:
            self.additional_losses = {}
