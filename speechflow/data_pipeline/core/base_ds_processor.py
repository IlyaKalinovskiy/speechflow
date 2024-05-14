import os
import enum
import typing as tp

from copy import deepcopy as copy

from speechflow.data_pipeline.core.datasample import DataSample
from speechflow.io import Config
from speechflow.utils.init import get_default_args, init_method_from_config

__all__ = ["BaseDSProcessor", "ComputeBackend"]


class ComputeBackend(enum.Enum):
    notset = 0
    numpy = 1
    torch = 2
    librosa = 3
    torchaudio = 4
    nvidia = 5


class BaseDSProcessor:
    def __init__(
        self,
        pipe: tp.Tuple[str, ...] = (),
        pipe_cfg: Config = Config.empty(),
        backend: ComputeBackend = ComputeBackend.notset,
        device: str = "cpu",
    ):
        self.pipe = pipe
        self.pipe_cfg = pipe_cfg
        self.backend = backend
        self.device = device

        self.components = {}
        self.transform_params = {}
        for step_name in self.pipe:
            if step_name != self.__class__.__name__:
                method = getattr(self, step_name)
                method_params = self.pipe_cfg.get(step_name, {})
            else:
                method = getattr(self, "__call__")
                method_params = {}

            handler = init_method_from_config(method, method_params)
            self.components[step_name] = handler

            if method.__name__ != "__call__":
                params = copy(handler.keywords)
            else:
                params = {}

            params.update(method_params)
            self.transform_params[step_name] = copy(params)  # type: ignore

    @staticmethod
    def get_config_from_locals(local: dict) -> Config:
        child_class = local.pop("self")
        if not hasattr(child_class, "__call__"):
            raise RuntimeError("Processor must have method __call__.")
        if get_default_args(child_class.__call__):
            raise RuntimeError("Method __call__ should not default arguments.")

        args = {k: v for k, v in local.items() if not k.startswith("_")}
        return Config({child_class.__class__.__name__: args})

    def init(self):
        if "DEVICE" in os.environ:
            self.device = os.environ.get("DEVICE")

    def process(self, ds: DataSample):
        ds.transform_params.update(self.transform_params)

        if self.pipe:
            for handler in self.components.values():
                ds = handler(ds)  # type: ignore
                if ds is None:
                    raise RuntimeError(
                        f"Handler {handler} should return DataSample object."
                    )

        return ds.to_numpy()
