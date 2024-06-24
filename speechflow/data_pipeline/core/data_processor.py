import io
import codecs
import pickle
import typing as tp
import hashlib
import logging
import argparse

from functools import partial
from os import environ as env
from pathlib import Path

import torch

from speechflow.data_pipeline.core import Batch, DataSample, PipeRegistry
from speechflow.data_pipeline.core.abstract import AbstractDataProcessor
from speechflow.data_pipeline.core.dataset import DatasetItem
from speechflow.io import Config, check_path, tp_PATH
from speechflow.logging import log_to_file, trace
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import ProfilerManager

__all__ = ["DataProcessor", "DataProcessingError"]

LOGGER = logging.getLogger("root")


class TensorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


class DataProcessingError(RuntimeError):
    def __init__(
        self,
        message: str,
        corrupted_sample: tp.Optional[DataSample] = None,
        fn: tp.Optional[str] = None,
    ):
        self.message = message
        self.corrupted_sample = str(corrupted_sample) if corrupted_sample else None
        self.fn = fn

    def __str__(self):
        return self.message


class DumpProcessor:
    @check_path(assert_file_exists=True)
    def __init__(
        self,
        data_root: tp_PATH,
        folder_path: tp_PATH,
        mode: str = "file_path",
        fields: tp.Optional[tp.Union[str, tp.List[str]]] = None,
        functions: tp.Optional[tp.Union[str, tp.List[str]]] = None,
        skip_samples_without_dump: bool = False,
        update_functions: tp.Optional[tp.Union[str, tp.List[str]]] = None,
    ):
        """
        :param update_functions: functions that will be updated in the dump, remaining functions will not be
        recalculated.
        """
        self._verbose_logging = bool(env.get("VERBOSE", False))

        self.data_root = data_root
        self.folder_path = folder_path
        self.mode = mode
        self.skip_samples_without_dump = skip_samples_without_dump

        (self.folder_path / "files").mkdir(parents=True, exist_ok=True)

        if fields is not None:
            self.fields = fields if isinstance(fields, tp.MutableSequence) else [fields]
        else:
            self.fields = []

        if functions is not None:
            self.preproc_functions = (
                functions if isinstance(functions, tp.MutableSequence) else [functions]
            )
        else:
            self.preproc_functions = []

        if update_functions is not None:
            self.update_functions = (
                update_functions
                if isinstance(update_functions, tp.MutableSequence)
                else [update_functions]
            )
        else:
            self.update_functions = []

        for func in self.update_functions:
            if func not in self.preproc_functions:
                self.preproc_functions.append(func)

        self.skip_flist_path = self.folder_path / "skip_samples.txt"
        self.skip_samples = self._load_skip_samples(self.skip_flist_path)

        self.preproc_functions_storage: tp.Dict = {}

    @staticmethod
    def _load_skip_samples(path: Path) -> tp.List[str]:
        if path.exists():
            return list(set(path.read_text(encoding="utf-8").split("\n")))
        else:
            return []

    def _get_sample_path(self, sample: DataSample) -> str:
        path = sample.file_path.as_posix()
        path = path.replace(self.data_root.as_posix(), "")
        return path

    def _get_filename(self, sample: DataSample) -> Path:
        if self.mode == "uid":
            name = sample.uid
        elif self.mode == "file_path":
            path = self._get_sample_path(sample)
            name = hashlib.sha256(path.encode("utf-8")).hexdigest()
        else:
            raise NotImplementedError

        file_path = self.folder_path / "files" / f"{name}.pkl"
        return file_path

    @staticmethod
    def get_name_and_fields(function) -> tp.Tuple[str, tp.List[str], str]:
        """Returns function name and a list of modified outputs."""
        init_params: Config = getattr(function, "init_params", Config.empty())

        while isinstance(function, partial):
            function = function.func

        fields = getattr(function, "_io", dict()).get("outputs")
        fields = [fields] if isinstance(fields, str) else fields
        name_attr = "_classname" if hasattr(function, "_classname") else "__name__"
        func_name = getattr(function, name_attr)
        return func_name, fields, init_params.hash

    def _load_preproc_data(
        self,
        sample: DataSample,
        func_name: str,
        func_fields: tp.List[str],
        hash_params: str,
    ):
        file_path = self._get_filename(sample)
        preloaded_data = self.preproc_functions_storage.get(file_path)
        preloaded_data_key = f"{func_name}|{hash_params}"
        if (
            isinstance(preloaded_data, tp.Mapping)
            and preloaded_data.get(preloaded_data_key) is not None
        ):
            saved_fields = preloaded_data.get(preloaded_data_key, {})
            if all(field in saved_fields for field in func_fields):
                sample.update(saved_fields)
                return True

        return False

    def apply_or_not(
        self,
        sample: DataSample,
        fn: tp.Callable,
    ) -> bool:
        func_name, func_fields, hash_params = self.get_name_and_fields(fn)

        if func_name in self.preproc_functions and self._load_preproc_data(
            sample, func_name, func_fields, hash_params
        ):
            return False

        if func_fields and all(
            name in self.fields and getattr(sample, name) is not None
            for name in func_fields
        ):
            return False

        return True

    def load_samples(self, samples: tp.List[DataSample]) -> tp.List[DataSample]:
        if not self._verbose_logging:
            num_samples = len(samples)
            samples = [
                s for s in samples if self._get_sample_path(s) not in self.skip_samples
            ]
            if num_samples != len(samples):
                message = (
                    f"{num_samples - len(samples)} samples thrown out as blacklisted!"
                )
                log_to_file(trace(self, message=message))

        if self.skip_samples_without_dump:
            num_samples = len(samples)
            samples = [s for s in samples if self._get_filename(s).exists()]
            if num_samples != len(samples):
                message = f"{num_samples - len(samples)} samples thrown out because no dump was found for them!"
                log_to_file(trace(self, message=message))

        for sample in samples:
            file_path = self._get_filename(sample)
            if not file_path.exists():
                if self._verbose_logging:
                    message = f"Dump for {sample.file_path.as_posix()} not found."
                    LOGGER.info(trace(self, message=message))
                continue

            try:
                with open(file_path.as_posix(), "rb") as f:
                    dump_data: tp.Dict[str, tp.Any] = TensorUnpickler(f).load()
            except (EOFError, pickle.UnpicklingError):
                file_path.unlink()
                continue
            except Exception as e:
                LOGGER.error(trace(self, e))
                continue

            dumped_fields = dump_data["fields"]
            if len(dumped_fields) != len(self.fields):
                if self._verbose_logging:
                    message = f"Not all fields are calculated for {sample.file_path.as_posix()}!"
                    LOGGER.warning(trace(self, message=message))

            sample.update(dumped_fields)

            dumped_functions = dump_data.get("functions", None)
            if dumped_functions:
                for func_name, func_dump_fields in dumped_functions.items():
                    if "|" in func_name:
                        func_name, hash_params = func_name.split("|")
                    else:
                        hash_params = None

                    if func_name in self.update_functions:
                        continue
                    file_path_storage = self.preproc_functions_storage.setdefault(
                        file_path, {}
                    )
                    file_path_storage[f"{func_name}|{hash_params}"] = func_dump_fields

        return samples

    def dump_samples(self, samples: tp.List[DataSample]):
        for sample in samples:
            file_path = self._get_filename(sample)

            if file_path.exists() and not self.update_functions:
                continue

            dump_data = {
                k: v
                for k, v in sample.to_dict().items()
                if k in self.fields and v is not None
            }
            if len(dump_data) != len(self.fields):
                if self._verbose_logging:
                    message = f"Not all fields are calculated for {sample.file_path.as_posix()}!"
                    LOGGER.warning(trace(self, message=message))

            all_dump_data = {"fields": dump_data}
            if self.preproc_functions_storage:
                all_dump_data["functions"] = self.preproc_functions_storage[file_path]
                if self._verbose_logging:
                    message = f"dump functions with keys: {list(all_dump_data['functions'].keys())}"
                    LOGGER.info(trace(self, message=message))

            file_path.write_bytes(pickle.dumps(all_dump_data))

        self.clear_storage()

    def skip(self, sample: DataSample):
        path = self._get_sample_path(sample)
        with codecs.open(self.skip_flist_path.as_posix(), "a", "utf-8") as f:
            f.write(f"{path}\n")
        self.skip_samples = self._load_skip_samples(self.skip_flist_path)

    def update_storage(
        self,
        samples: tp.List[DataSample],
        func_name: str,
        func_fields: tp.List[str],
        hash_params: str,
    ):
        for sample in samples:
            file_path = self._get_filename(sample)
            dump_data = {
                k: v
                for k, v in sample.to_dict().items()
                if k in func_fields and v is not None
            }
            file_path_storage = self.preproc_functions_storage.setdefault(file_path, {})
            file_path_storage[f"{func_name}|{hash_params}"] = dump_data

    def clear_storage(self):
        self.preproc_functions_storage = {}


class DataProcessor(AbstractDataProcessor):
    def __init__(
        self,
        preproc_fn: tp.Sequence[tp.Callable],
        collate_fn: tp.Optional[tp.Callable] = None,
        output_collated_only: bool = False,
        dump: tp.Optional[tp.Dict] = None,
    ):
        super().__init__()

        if preproc_fn:
            PipeRegistry.check(preproc_fn, input_fields=DataSample.all_keys())
            self._preproc_fn = preproc_fn
        else:
            self._preproc_fn = []

        self._collate_fn = collate_fn
        self._output_collated_only = output_collated_only

        if dump:
            self._dump_proc = init_class_from_config(DumpProcessor, dump)()
        else:
            self._dump_proc = None  # type: ignore

        self._use_profiler = bool(env.get("DATAPIPE_PROFILING", False))

    @staticmethod
    def apply(
        sample: DataSample,
        preproc_fn: tp.Sequence[tp.Callable],
        dump_proc: tp.Optional[DumpProcessor] = None,
        use_profiler: bool = False,
    ) -> tp.List[DataSample]:
        for fn in preproc_fn:
            if dump_proc and not dump_proc.apply_or_not(sample, fn):
                continue

            func_name, fields, hash_params = DumpProcessor.get_name_and_fields(fn)

            with ProfilerManager(func_name, group="PIPE", enable=use_profiler):
                sample = fn(sample)

            if dump_proc:
                if func_name in dump_proc.preproc_functions:
                    if (
                        dump_proc.update_functions
                        and func_name not in dump_proc.update_functions
                    ):
                        continue
                    dump_proc.update_storage([sample], func_name, fields, hash_params)

        return [sample]

    def do_preprocessing(
        self,
        in_samples: tp.List[DataSample],
        preproc_fn: tp.Sequence[tp.Callable],
        skip_corrupted_samples: bool = True,
    ) -> tp.List[DataSample]:
        if preproc_fn:
            out_samples = []
            for sample in in_samples:
                try:
                    processed_samples = DataProcessor.apply(
                        sample, preproc_fn, self._dump_proc, self._use_profiler
                    )
                    out_samples.extend(processed_samples)
                except Exception as e:
                    trace_message = trace(
                        DataProcessor,
                        exception=e,
                        message=f"Filepath: {sample.file_path}",
                    )
                    LOGGER.error(trace_message)
                    if not skip_corrupted_samples:
                        raise DataProcessingError(
                            message=f"Error occured while processing datasamples. Traceback: \n{trace_message}",
                            corrupted_sample=sample,
                        )
                    if self._dump_proc:
                        self._dump_proc.skip(sample)

            out_samples = out_samples
        else:
            out_samples = in_samples

        return out_samples

    def process(self, in_samples: tp.List[DataSample]) -> tp.Union[Batch, None]:
        if len(in_samples) == 0:
            LOGGER.warning(trace(self, message="Input samples list is empty!"))

        try:
            is_last_batch = in_samples[-1] is None
            if is_last_batch:
                in_samples.pop()

            in_samples = [
                item.get() if isinstance(item, DatasetItem) else item
                for item in in_samples
            ]

            if self._dump_proc:
                with ProfilerManager(
                    "load_samples", group="DUMP", enable=self._use_profiler
                ):
                    in_samples = self._dump_proc.load_samples(in_samples)
                    if len(in_samples) == 0:
                        LOGGER.warning(
                            trace(
                                self,
                                message="Samples list after loading dump is empty!",
                            )
                        )

            out_samples = self.do_preprocessing(in_samples, self._preproc_fn)

            if self._dump_proc:
                with ProfilerManager(
                    "dump_samples", group="DUMP", enable=self._use_profiler
                ):
                    self._dump_proc.dump_samples(out_samples)

            with ProfilerManager("collated", group="PIPE", enable=self._use_profiler):
                collated_samples = (
                    self._collate_fn(out_samples) if self._collate_fn else None
                )

            batch = Batch(
                size=len(out_samples),
                is_last=is_last_batch,
                data_samples=None if self._output_collated_only else out_samples,
                collated_samples=collated_samples,
            )
            return batch

        except Exception as e:
            LOGGER.error(trace(self, e))
            log_to_file(e)
            return None


if __name__ == "__main__":
    from speechflow.io import construct_file_list

    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-d", "--data_root", help="path to dataset", type=Path, required=True
    )
    arguments_parser.add_argument(
        "-f", "--folder_path", help="path to dump folder", type=Path, required=True
    )
    arguments_parser.add_argument(
        "-ext",
        "--file_extension",
        help="dataset files extension",
        type=str,
        default=".TextGridStage2",
    )
    args = arguments_parser.parse_args()

    flist = construct_file_list(args.data_root, ext=args.file_extension)
    dlist = construct_file_list(args.folder_path, ext=".pkl")

    dump_proc = DumpProcessor(folder_path=args.folder_path, data_root=args.data_root)

    for file_path_ in flist[:100]:
        fname_ = dump_proc._get_filename(DataSample(file_path=Path(file_path_))).name
        assert any(fname_ in path for path in dlist)

    print("DUMP IS OK!")
