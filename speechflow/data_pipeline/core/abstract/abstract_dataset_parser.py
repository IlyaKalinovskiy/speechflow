import abc
import typing as tp

from pathlib import Path

from speechflow.data_pipeline.core.dataset import Dataset


class AbstractDatasetParser:
    """Base class for data loaders for pyTorch nets."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """Everything that we need to init."""

    @staticmethod
    @abc.abstractmethod
    def do_preprocessing(
        all_metadata: tp.Union[tp.List[tp.Any], Dataset],
        preproc_fn: tp.Sequence[tp.Callable],
    ) -> Dataset:
        """Apply processing functions."""
        pass

    @abc.abstractmethod
    def to_datasample(self, all_metadata: tp.Union[tp.List[tp.Any], Dataset]) -> Dataset:
        """Convert metadata to datasample."""
        pass

    @abc.abstractmethod
    def read_datasamples(
        self,
        file_list: tp.Union[tp.List[str], tp.List[Path]],
        data_root: tp.Optional[tp.Union[str, Path]] = None,
        n_processes: tp.Optional[int] = None,
        post_read_hooks: tp.Optional[tp.Sequence[tp.Callable]] = None,
    ) -> Dataset:
        """Read all datasamples from file_list."""
        pass
