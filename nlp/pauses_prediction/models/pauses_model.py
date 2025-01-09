import typing as tp

import torch

from torch import nn

from nlp.pauses_prediction.data_types import PausesPredictionInput, PausesPredictionOutput
from nlp.pauses_prediction.models.params import PausesPredictionParams
from tts.acoustic_models.modules.common.layers import Conv
from tts.acoustic_models.modules.embedding_calculator import EmbeddingCalculator


class SimpleModel(EmbeddingCalculator):
    params: PausesPredictionParams

    def __init__(
        self, params: tp.Union[PausesPredictionParams, dict], strict_init: bool = True
    ):
        super().__init__(PausesPredictionParams.create(params, strict_init))
        params = self.params

        self.encoder = Encoder(params)
        decoder_in_dim = params.encoder_emb_dim * 2
        if params.use_onehot_speaker_emb or params.use_learnable_speaker_emb:
            decoder_in_dim += params.speaker_emb_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, params.decoder_dim),
            nn.ReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.decoder_dim, params.decoder_dim),
            nn.ReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.decoder_dim, 1),
            nn.ReLU(),
        )

    def forward(self, inputs: PausesPredictionInput) -> PausesPredictionOutput:  # type: ignore
        ling_feat = self.get_ling_feat(inputs)  # type: ignore
        speaker_emb = self.get_speaker_embedding(inputs)  # type: ignore

        sil_mask = inputs.sil_mask
        ilens = inputs.input_lengths
        _is_sil = (sil_mask > 0).unsqueeze(-1)

        if ling_feat is not None:
            x = torch.cat([_is_sil, ling_feat], dim=2)

        encoder_output = self.encoder(x, ilens)

        if speaker_emb is not None:
            se_to_concat = speaker_emb.unsqueeze(1).expand(-1, encoder_output.size(1), -1)
            encoder_output = torch.cat([encoder_output, se_to_concat], dim=-1)

        predicted_durations = self.decoder(encoder_output)

        output = PausesPredictionOutput(
            sil_mask=sil_mask,
            durations=predicted_durations,
        )
        return output


class Encoder(EmbeddingCalculator):
    params: PausesPredictionParams

    def __init__(
        self, params: tp.Union[PausesPredictionParams, dict], strict_init: bool = True
    ):
        super().__init__(PausesPredictionParams.create(params, strict_init))
        params = self.params

        emb_dim = params.encoder_emb_dim
        input_dim = params.encoder_emb_dim + 1
        self.use_convolutions = params.use_convolutions
        if self.use_convolutions:
            convolutions = []
            for _ in range(params.encoder_n_convolutions):
                conv_layer = nn.Sequential(
                    Conv(
                        input_dim,
                        input_dim,
                        kernel_size=params.encoder_kernel_size,
                        stride=1,
                        padding=int((params.encoder_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="relu",
                    ),
                    nn.BatchNorm1d(input_dim),
                    nn.ReLU(),
                    nn.Dropout(params.dropout),
                )
                convolutions.append(conv_layer)
            self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(
            emb_dim + 1,
            params.encoder_rnn_dim,
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, lens):

        if self.use_convolutions:
            x = x.transpose(1, 2)
            for layer in self.convolutions:
                x = layer(x)
            x = x.transpose(1, 2)

        input_lens = lens.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lens, batch_first=True, enforce_sorted=False
        )

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs
