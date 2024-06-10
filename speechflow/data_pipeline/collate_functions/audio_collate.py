import typing as tp

from dataclasses import dataclass

import torch

from torch import Tensor

from speechflow.data_pipeline.core import BaseCollate, BaseCollateOutput
from speechflow.data_pipeline.datasample_processors.data_types import AudioDataSample
from speechflow.training.utils.pad_utils import pad_2d

__all__ = [
    "AudioCollate",
    "AudioCollateOutput",
]


@dataclass
class AudioCollateOutput(BaseCollateOutput):
    mu_law_waveform: Tensor = None
    lpc_waveform: Tensor = None
    waveform_lengths: Tensor = None
    lang_id: Tensor = None
    speaker_id: Tensor = None
    speaker_emb: Tensor = None
    speaker_emb_mean: Tensor = None
    speech_quality_emb: Tensor = None
    lpc_feat: Tensor = None
    lpc_feat_lengths: Tensor = None
    ssl_feat: Tensor = None
    ssl_feat_lengths: Tensor = None
    ac_feat: Tensor = None
    ac_feat_lengths: Tensor = None


class AudioCollate(BaseCollate):
    def __call__(self, batch: tp.List[AudioDataSample]) -> AudioCollateOutput:  # type: ignore
        collated = super().__call__(batch)  # type: ignore
        collated = AudioCollateOutput(**collated.to_dict())  # type: ignore
        multiple = self.multiple.get("spec")

        if batch[0].mu_law_waveform is not None:
            if batch[0].mu_law_waveform.ndim == 1:
                mu_law_waveform = [sample.mu_law_waveform.unsqueeze(-1) for sample in batch]  # type: ignore
            else:
                mu_law_waveform = [sample.mu_law_waveform for sample in batch]

            pad_val = self.pad_values.get("mu_law", 0.0)
            mu_law_waveform, waveform_lens = pad_2d(
                mu_law_waveform, pad_val, mu_law_waveform[0].shape[1]
            )
        else:
            mu_law_waveform = None
            waveform_lens = None

        if batch[0].lpc_waveform is not None:
            lpc_waveform = [sample.lpc_waveform for sample in batch]
            pad_val = self.pad_values.get("lpc_feat", 0.0)
            lpc_waveform, waveform_lens = pad_2d(
                lpc_waveform, pad_val, lpc_waveform[0].shape[1]
            )
        else:
            lpc_waveform = None

        if batch[0].lang_id is not None:
            lang_id = [x.lang_id for x in batch]
            lang_id = torch.LongTensor(lang_id)
        else:
            lang_id = None

        if batch[0].speaker_id is not None:
            speaker_id = [x.speaker_id for x in batch]
            speaker_id = torch.LongTensor(speaker_id)
        else:
            speaker_id = None

        if batch[0].speaker_emb is not None:
            speaker_emb = torch.vstack([x.speaker_emb for x in batch])
        else:
            speaker_emb = None

        if batch[0].speaker_emb_mean is not None:
            speaker_emb_mean = torch.vstack([x.speaker_emb_mean for x in batch])
        else:
            speaker_emb_mean = None

        if batch[0].speech_quality_emb is not None:
            speech_quality_emb = torch.vstack([x.speech_quality_emb for x in batch])
        else:
            speech_quality_emb = None

        if batch[0].lpc_feat is not None:
            lpc_feat = [sample.lpc_feat for sample in batch]
            pad_val = self.pad_values.get("lpc_feat", 0.0)
            lpc_feat, lpc_feat_lens = pad_2d(lpc_feat, pad_val, lpc_feat[0].shape[1])
        else:
            lpc_feat = None
            lpc_feat_lens = None

        if batch[0].ssl_feat is not None:
            pad_val = self.pad_values.get("ssl_encode", 0.0)
            embs = [x.ssl_feat.encode for x in batch]
            ssl_feat, ssl_feat_lens = pad_2d(
                embs, pad_val, embs[0].shape[1], multiple=multiple
            )
        else:
            ssl_feat = None
            ssl_feat_lens = None

        if batch[0].ac_feat is not None:
            pad_val = self.pad_values.get("ac_encode", 0.0)
            embs = [x.ac_feat.encode for x in batch]
            ac_feat, ac_feat_lens = pad_2d(
                embs, pad_val, embs[0].shape[1], multiple=multiple
            )
        else:
            ac_feat = None
            ac_feat_lens = None

        collated.mu_law_waveform = mu_law_waveform
        collated.lpc_waveform = lpc_waveform
        collated.waveform_lengths = waveform_lens
        collated.lang_id = lang_id
        collated.speaker_id = speaker_id
        collated.speaker_emb = speaker_emb
        collated.speaker_emb_mean = speaker_emb_mean
        collated.speech_quality_emb = speech_quality_emb
        collated.lpc_feat = lpc_feat
        collated.lpc_feat_lengths = lpc_feat_lens
        collated.ssl_feat = ssl_feat
        collated.ssl_feat_lengths = ssl_feat_lens
        collated.ac_feat = ac_feat
        collated.ac_feat_lengths = ac_feat_lens
        return collated
