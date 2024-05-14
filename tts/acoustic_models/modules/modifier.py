import enum

import torch

from torch import nn

from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, ComponentOutput
from tts.acoustic_models.modules.params import ModifierParams

__all__ = ["Mode", "ModeStage"]


class ModeStage(enum.Enum):
    s_0 = 0
    s_1 = 1
    s_2 = 2
    s_3 = 3

    @property
    def value(self) -> int:
        return self._value_


class Mode(Component):
    params: ModifierParams

    def __init__(self, params: ModifierParams, input_dim, current_pos: ModeStage):
        super().__init__(params, input_dim)
        self._current_pos = current_pos

        self.cat = params.mode_cat.get(self._current_pos.value)
        self.add = params.mode_add.get(self._current_pos.value)

        if self.add:
            if isinstance(self.add, dict):
                raise ValueError(
                    "Modifier mode_add parameter does not support custom projection shapes. \
                                  Use mode_cat instead."
                )
            self.add = (self.add,) if isinstance(self.add, str) else self.add
            self.add_projections = nn.ModuleDict({})
            for emb_type in self.add:
                emb_size = getattr(params, f"{emb_type}_embedding_dim")

                if emb_size != self.input_dim:
                    proj = nn.Linear(emb_size, self.input_dim)
                    self.add_projections[emb_type] = proj
                else:
                    self.add_projections[emb_type] = None

        self._dim_increment = 0
        if self.cat:
            self.cat = (self.cat,) if isinstance(self.cat, str) else self.cat
            if not isinstance(self.cat, dict):
                self.cat = {emb_type: None for emb_type in self.cat}

            self.cat_projections = nn.ModuleDict({})
            for emb_type in self.cat:
                if emb_type.startswith("average") and emb_type != "average_emb":
                    avg_params = self.params.averages[emb_type.split("_", 1)[1]]
                    emb_size = avg_params["emb_dim"]
                else:
                    emb_size = getattr(params, f"{emb_type}_dim")
                proj_size = self.cat[emb_type]
                if proj_size is not None and proj_size != emb_size:
                    proj = nn.Sequential(nn.Linear(emb_size, proj_size), nn.ReLU())
                    self.cat_projections[emb_type] = proj
                    self._dim_increment += proj_size
                else:
                    self.cat_projections[emb_type] = None
                    self._dim_increment += emb_size

    @property
    def output_dim(self):
        if isinstance(self.input_dim, list):
            return [d + self._dim_increment for d in self.input_dim]
        else:
            return self.input_dim + self._dim_increment

    def forward_step(self, inputs: ComponentInput) -> ComponentOutput:  # type: ignore
        content = (
            [inputs.content] if not isinstance(inputs.content, list) else inputs.content
        )

        embeddings_dict = inputs.embeddings

        if self.add:
            for emb_type in self.add:
                embedding = embeddings_dict.get(emb_type)
                if embedding is None:
                    raise ValueError(f"{emb_type} embedding not found.")
                projection = self.add_projections[emb_type]  # type: ignore
                if projection is not None:
                    embedding = projection(embedding)
                if embedding.dim() == 2:
                    embedding = embedding.unsqueeze(1)
                if isinstance(content, list):
                    for i, tensor in enumerate(content):
                        content[i] = tensor + embedding
                else:
                    content = content + embedding

        if self.cat:
            for emb_type in self.cat:
                embedding = embeddings_dict.get(emb_type)
                if embedding is None:
                    raise ValueError(f"'{emb_type}' tensor not found.")
                projection = self.cat_projections[emb_type]
                if projection is not None:
                    embedding = projection(embedding)
                if embedding.dim() == 2:
                    embedding = embedding.unsqueeze(1)
                if isinstance(content, list):
                    for i, tensor in enumerate(content):
                        temp_tensor = embedding.expand(-1, tensor.size(1), -1)
                        content[i] = torch.cat([tensor, temp_tensor], dim=-1)
                else:
                    temp_tensor = embedding.expand(-1, content.size(1), -1)
                    content = torch.cat([content, temp_tensor], dim=-1)

        if not isinstance(inputs.content, list):
            inputs.content = content[0]
        else:
            inputs.content = content

        return inputs
