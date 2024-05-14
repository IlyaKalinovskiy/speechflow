import re
import typing as tp

import torch

from transformers import AutoModel

from nlp.prosody_prediction.data_types import (
    ProsodyPredictionInput,
    ProsodyPredictionOutput,
)
from nlp.prosody_prediction.models.params import ProsodyPredictionParams
from tts.acoustic_models.modules.embedding_calculator import EmbeddingCalculator


class ProsodyModel(EmbeddingCalculator):
    params: ProsodyPredictionParams

    def __init__(
        self, params: tp.Union[ProsodyPredictionParams, dict], strict_init: bool = True
    ):
        super().__init__(ProsodyPredictionParams.create(params, strict_init))
        params = self.params

        self.bert = AutoModel.from_pretrained(params.model_name, add_pooling_layer=False)
        self.predictors = torch.nn.ModuleDict({})
        self.latent_dim = self.bert.cfg.hidden_size

        if params.classification_task in ["both", "binary"]:
            self.predictors["binary"] = torch.nn.Sequential(
                torch.nn.Dropout(p=params.dropout),
                torch.nn.Linear(self.latent_dim, self.latent_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=params.dropout),
                torch.nn.Linear(self.latent_dim, 2),
            )
        if params.classification_task in ["both", "category"]:
            self.predictors["category"] = torch.nn.Sequential(
                torch.nn.Dropout(p=params.dropout),
                torch.nn.Linear(self.latent_dim, self.latent_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=params.dropout),
                torch.nn.Linear(self.latent_dim, params.n_classes),
            )

        if (
            params.n_layers_tune is not None
            and self.bert.cfg.num_hidden_layers > params.n_layers_tune
        ):
            layers_tune = "|".join(
                [
                    str(self.bert.cfg.num_hidden_layers - i)
                    for i in range(1, params.n_layers_tune)
                ]
            )
            for name, param in self.bert.named_parameters():
                if not re.search(f"pooler|drop|{layers_tune}", name):
                    param.requires_grad = False

    def forward(self, inputs: ProsodyPredictionInput) -> ProsodyPredictionOutput:  # type: ignore
        hidden = self.bert(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
        )[0]

        outputs = {"binary": None, "category": None}
        for name in self.predictors:
            outputs[name] = self.predictors[name](hidden)

        output = ProsodyPredictionOutput(
            binary=outputs["binary"],
            category=outputs["category"],
        )

        return output
