import typing as tp

import torch

from torch.nn import functional as F

from speechflow.training.utils.tensor_utils import (
    get_lengths_from_durations,
    get_mask_from_lengths,
    merge_additional_outputs,
)
from tts.acoustic_models.modules.common import VarianceEmbedding
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import EncoderOutput, VarianceAdaptorOutput
from tts.acoustic_models.modules.params import VarianceAdaptorParams

__all__ = ["ForwardVarianceAdaptor"]


class ForwardVarianceAdaptor(Component):
    params: VarianceAdaptorParams

    def __init__(
        self, params: VarianceAdaptorParams, input_dim: tp.Union[int, tp.Tuple[int, ...]]
    ):
        from tts.acoustic_models.modules import LENGTH_REGULATORS, VARIANCE_PREDICTORS

        super().__init__(params, input_dim)
        self.input_dim = (
            (input_dim, input_dim) if isinstance(input_dim, int) else input_dim
        )
        self.va_variances: tp.Tuple[str, ...] = params.va_variances  # type: ignore
        assert isinstance(self.va_variances, (tuple, list))
        self.va_variance_params = params.va_variance_params
        assert isinstance(self.va_variance_params, dict)

        self.predictors = torch.nn.ModuleDict()
        for name in self.va_variances:
            variance_params = self.va_variance_params[name]
            predictor_cls = VARIANCE_PREDICTORS[variance_params.predictor_type]
            predictor_params = variance_params.predictor_params

            if isinstance(predictor_cls, tuple):
                predictor_cls, predictor_param_cls = predictor_cls
            else:
                predictor_param_cls = None

            if variance_params.input_content_dim is None:
                input_content_dim = [None] * len(variance_params.input_content)
            else:
                input_content_dim = variance_params.input_content_dim

            # TODO bugs here vp.tag is None
            content_dim = []
            for p, index in enumerate(variance_params.input_content):
                if str(index).isdigit():
                    content_dim.append(self.input_dim[index])
                elif input_content_dim[p] not in [None, "None"]:
                    content_dim.append(input_content_dim[p])
                else:
                    if index == "transcription":
                        content_dim.append(getattr(params, "token_emb_dim"))
                    elif hasattr(params, f"{index}_proj_dim"):
                        content_dim.append(getattr(params, f"{index}_proj_dim"))
                    else:
                        for vp_name, vp in self.va_variance_params.items():
                            if vp.tag == index:
                                content_dim.append(vp.predictor_params.vp_latent_dim)
                            elif vp_name == index:
                                content_dim.append(vp.dim)

            if variance_params.input_content_cat or len(content_dim) == 1:
                content_dim = sum(d for d in content_dim)

            vp_params = predictor_param_cls.init_from_parent_params(
                params, predictor_params.vp_params
            )
            for key in vp_params.to_dict():
                if key in predictor_params.to_dict():
                    setattr(vp_params, key, getattr(predictor_params, key))
            vp_params.vp_output_dim = variance_params.dim
            self.predictors[name] = predictor_cls(vp_params, content_dim)

        self.embeddings = torch.nn.ModuleDict()
        for name in self.va_variances:
            variance_params = self.va_variance_params[name]
            if variance_params.as_embedding:
                self.embeddings[name] = VarianceEmbedding(
                    interval=variance_params.interval,
                    n_bins=variance_params.n_bins,
                    emb_dim=variance_params.emb_dim,
                    log_scale=variance_params.log_scale,
                )

        self.length_regulators = torch.nn.ModuleDict()
        for name in self.va_variances:
            variance_params = self.va_variance_params[name]
            if not variance_params.aggregate_by_tokens:
                length_regulator_cls = LENGTH_REGULATORS[params.va_length_regulator_type]
                self.length_regulators[name] = length_regulator_cls()

        self._length_regulator_ssml = None

    @property
    def output_dim(self):
        out_dim = list(self.input_dim)
        for name in self.va_variances:
            if name == "durations":
                continue

            variance_params = self.va_variance_params[name]
            if isinstance(self.predictors[name], Component):
                dim = self.predictors[name].output_dim
                if variance_params.as_embedding:
                    dim = variance_params.emb_dim * dim
            else:
                if variance_params.as_embedding:
                    dim = variance_params.emb_dim * variance_params.dim
                else:
                    dim = variance_params.dim

            for i in variance_params.cat_to_content:
                out_dim[i] += dim

            for i in variance_params.overwrite_content:
                out_dim[i] = dim

        return tuple(out_dim)

    @property
    def length_regulator_ssml(self):
        from tts.acoustic_models.modules import LENGTH_REGULATORS

        if self._length_regulator_ssml is None:
            length_regulator_cls = LENGTH_REGULATORS[self.params.va_length_regulator_type]
            self._length_regulator_ssml = length_regulator_cls()
            self._length_regulator_ssml.sigma = torch.nn.Parameter(
                torch.tensor(999999.0), requires_grad=False
            )
        return self._length_regulator_ssml

    @staticmethod
    def _get_x(inputs: EncoderOutput) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        content = inputs.content

        if isinstance(content, list):
            assert 1 <= len(content) <= 2, "unsupported content format"
            x = content[0]
            x_adaptor = content[-1]
        else:
            x = content
            x_adaptor = content

        return x, x_adaptor

    @staticmethod
    def _get_x_lengths(inputs: EncoderOutput) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        content_lengths = inputs.content_lengths

        if isinstance(content_lengths, list):
            assert 1 <= len(content_lengths) <= 2, "unsupported content format"
            x_lens = content_lengths[0]
            x_adaptor_lens = content_lengths[-1]
        else:
            x_lens = content_lengths
            x_adaptor_lens = content_lengths

        return x_lens, x_adaptor_lens

    @staticmethod
    def _get_targets(
        inputs: EncoderOutput, variance_names, variance_params
    ) -> tp.Dict[str, torch.Tensor]:
        targets = {}
        content = ForwardVarianceAdaptor._get_x(inputs)
        for var_name in variance_names:
            target_name = var_name
            if variance_params[var_name].target is not None:
                target_name = variance_params[var_name].target

            if "content" in target_name:
                targets[var_name] = content[int(target_name.replace("content_", ""))]
            else:
                target = getattr(
                    inputs.model_inputs, target_name.replace("_encoder", ""), None
                )
                if target is None:
                    if "dummy" not in var_name:
                        target = inputs.model_inputs.additional_inputs.get(target_name)
                if target is not None:
                    if not isinstance(target, list):
                        targets[var_name] = target.float()  # type: ignore
                    else:
                        targets[var_name] = target

        return targets

    @staticmethod
    def _get_spec_len_from_duration(durations: torch.Tensor):
        spec_lengths = get_lengths_from_durations(durations)
        max_spec_len = spec_lengths.max()
        mask = get_mask_from_lengths(spec_lengths)
        return spec_lengths, max_spec_len, mask.to(durations.device)

    @staticmethod
    def _get_modifier_name(var_name: str) -> str:
        if var_name == "energy":
            return "volume_modifier"
        elif var_name == "aggregate_pitch":
            return "pitch_modifier"
        else:
            return ""

    @staticmethod
    def _get_input_content(x_duration, x_adaptor, variance_params, model_inputs):
        content = []
        for name in variance_params.input_content:
            if str(name).isdigit():
                if name == 0:
                    content.append(x_duration)
                elif name == 1:
                    content.append(x_adaptor)
            else:
                feat = model_inputs.additional_inputs.get(name)
                if feat is None:
                    feat = getattr(model_inputs, name, None)
                if feat is not None and not isinstance(feat, list):
                    assert feat is not None
                    if feat.ndim == 2:
                        feat = feat.unsqueeze(1)
                    if len(content) > 0:
                        t_dim = content[-1].shape[1]
                    else:
                        t_dim = x_duration.shape[1]
                    if feat.shape[1] == 1:
                        if x_duration.shape[1] == x_adaptor.shape[1]:
                            feat = feat.expand(-1, t_dim, -1)
                        elif len(content) > 0 and content[0].shape[1] != 1:
                            feat = feat.expand(-1, content[0].shape[1], -1)
                content.append(feat)

        if not variance_params.input_content_cat and not any(
            isinstance(c, list) for c in content
        ):
            if len(content) > 1:
                content = [content]

        if not any(isinstance(c, list) for c in content):
            content = torch.cat(content, dim=2)
            return content.detach() if variance_params.detach_input else content
        else:
            content = content[0]
            if variance_params.detach_input:
                return [c.detach() for c in content]
            else:
                return content

    def _predict_durations(
        self,
        x_duration,
        x_adaptor,
        x_duration_mask,
        x_adaptor_mask,
        targets,
        model_inputs,
    ):
        variance_params = self.va_variance_params["durations"]

        content = self._get_input_content(
            x_duration, x_adaptor, variance_params, model_inputs
        )

        if content.shape[1] == x_duration.shape[1]:
            src_mask = x_duration_mask
        elif content.shape[1] == x_adaptor.shape[1]:
            src_mask = x_adaptor_mask
        else:
            raise RuntimeError

        prediction, content, loss = self.predictors["durations"](
            content,
            src_mask,
            target=targets.get("durations"),
            model_inputs=model_inputs,
            name="durations",
        )

        if (
            self.training
            and variance_params.use_target
            and targets.get("durations") is not None
        ):
            durations = targets["durations"]
        else:
            durations = (
                prediction.detach() if variance_params.detach_output else prediction
            )

        return durations, prediction, content, loss

    def _process_durations(
        self,
        x_duration,
        x_adaptor,
        x_duration_mask,
        x_adaptor_mask,
        targets,
        model_inputs,
        **kwargs,
    ):
        variance_params = self.va_variance_params.get("durations")
        ignored_variance = kwargs.get("ignored_variance")
        if ignored_variance and "durations" in ignored_variance:
            return None, None, {}, {}

        if "durations" not in targets and "durations" not in self.predictors:
            return None, None, {}, {}

        if "fa_postprocessed" in model_inputs.additional_inputs:
            durations = model_inputs.additional_inputs["fa_postprocessed"]
            durations = durations.squeeze(1).squeeze(-1)
            return durations, durations, {"durations_postprocessed": durations}, {}

        if "durations" in self.predictors:
            (
                durations,
                durations_prediction,
                durations_content,
                durations_loss,
            ) = self._predict_durations(
                x_duration,
                x_adaptor,
                x_duration_mask,
                x_adaptor_mask,
                targets,
                model_inputs,
            )
        else:
            durations = durations_prediction = targets.get("durations")
            durations_content = {}
            durations_loss = {}

        if self.training and not (
            variance_params.begin_iter
            <= model_inputs.batch_idx
            < variance_params.end_iter
        ):
            durations = durations_prediction = targets.get("durations")
            durations_content = {}
            durations_loss = {}

        if variance_params is not None:
            durations = torch.expm1(durations)

        if variance_params.denormalize:
            _range = model_inputs.ranges["durations"]
            durations = durations * _range[:, 2:3] + _range[:, 0:1]
            if model_inputs.durations is not None:
                model_inputs.durations = (
                    model_inputs.durations * _range[:, 2:3] + _range[:, 0:1]
                )

        durations_content.update({"durations_postprocessed": durations})
        return durations, durations_prediction, durations_content, durations_loss

    def _predict_variance(
        self,
        name,
        x_duration,
        x_adaptor,
        duration,
        target,
        x_duration_mask,
        x_adaptor_mask,
        spec_mask,
        max_spec_len,
        model_inputs,
        modifier=None,
        **kwargs,
    ):
        variance_params = self.va_variance_params[name]

        content = self._get_input_content(
            x_duration, x_adaptor, variance_params, model_inputs
        )

        if not isinstance(content, list):
            content = [content]

        mask = []
        for c in content:
            if c is None:
                m = None
            elif c.shape[1] == x_duration.shape[1]:
                m = x_duration_mask
            elif c.shape[1] == x_adaptor.shape[1]:
                m = x_adaptor_mask
            else:
                m = get_mask_from_lengths(model_inputs.transcription_lengths)
            mask.append(m)

        if len(content) == 1:
            content = content[0]
            mask = mask[0]

        if not variance_params.aggregate_by_tokens:
            if duration is not None:
                length_regulator = self.length_regulators[name]
                content, _ = length_regulator(content, duration, max_spec_len)
                mask = spec_mask

            if modifier is not None and duration is not None:
                modifier, _ = self.length_regulator_ssml(
                    modifier.unsqueeze(2), duration, max_spec_len
                )
                modifier = modifier.squeeze(-1)

        predictor = self.predictors[name]
        try:
            prediction, additional_content, additional_losses = predictor(
                content,
                mask,
                target=target,
                model_inputs=model_inputs,
                name=name,
                **kwargs,
            )
        except Exception as e:
            print(f"{name} prediction failed: {e}")
            raise e

        if modifier is not None and prediction.shape[1] == modifier.shape[1]:
            prediction *= modifier.unsqueeze(-1) if prediction.ndim > 2 else modifier

        if (
            not isinstance(predictor, Component)
            and variance_params.predictor_params.tag != "default"
        ):
            additional_content[variance_params.predictor_params.tag] = prediction[:, 0, :]

        return prediction, additional_content, additional_losses

    def _postprocessing_variance(
        self,
        name,
        prediction,
        target,
        ranges,
    ):
        variance_params = self.va_variance_params
        if self.training and variance_params[name].use_target and target is not None:
            variance = target
        else:
            variance = (
                prediction.detach() if variance_params[name].detach_output else prediction
            )

        if variance_params[name].denormalize:
            range_name = name
            if variance_params[name].target is not None:
                range_name = variance_params[name].target

            range_name = range_name.replace("aggregate_", "")
            variance = variance * ranges[range_name][:, 2:3] + ranges[range_name][:, 0:1]

        if variance_params[name].as_embedding:
            embedding_builder = self.embeddings[name]
            variance_emb = embedding_builder(variance)
        else:
            variance_emb = variance

        if variance_emb.ndim == 2:
            variance_emb = variance_emb.unsqueeze(-1)
            variance = variance.unsqueeze(-1)

        return variance_emb, variance

    def _process_variance(
        self,
        x_duration,
        x_adaptor,
        durations,
        targets,
        x_duration_mask,
        x_adaptor_mask,
        spec_mask,
        max_spec_len,
        model_inputs,
        **kwargs,
    ) -> tp.Tuple[tp.Dict[str, torch.Tensor], ...]:
        variance_embeddings = {}
        variance_predictions = {}
        variance_content = {}
        variance_losses = {}
        current_iter = model_inputs.batch_idx

        for name in self.va_variances:
            if name == "durations":
                continue

            if kwargs.get("ignored_variance") and name in kwargs.get("ignored_variance"):
                continue

            if not (
                self.va_variance_params[name].begin_iter
                <= current_iter
                < self.va_variance_params[name].end_iter
            ):
                if self.training and self.va_variance_params[name].skip:
                    continue

            (
                variance_predictions[name],
                variance_content[name],
                variance_losses[name],
            ) = self._predict_variance(
                name,
                x_duration,
                x_adaptor,
                durations,
                targets.get(name),
                x_duration_mask,
                x_adaptor_mask,
                spec_mask,
                max_spec_len,
                model_inputs,
                getattr(model_inputs, self._get_modifier_name(name), None),
                **kwargs,
            )

            if not (
                self.va_variance_params[name].begin_iter
                <= current_iter
                < self.va_variance_params[name].end_iter
            ):
                variance_losses.pop(name)
                variance_predictions[name] = variance_predictions[name].detach()

        for name, prediction in variance_predictions.items():
            embeddings, variance = self._postprocessing_variance(
                name, prediction, targets.get(name), model_inputs.ranges
            )
            variance_embeddings[name] = embeddings
            variance_content[name].update({f"{name}_postprocessed": variance})

            if (
                self.training
                and self.va_variance_params[name].with_loss
                and name in variance_losses
            ):
                loss = getattr(F, self.va_variance_params[name].loss_type)(
                    prediction, targets.get(name).detach()
                )
                variance_losses[name].update({f"{name}_l1": loss})

        return (
            variance_embeddings,
            variance_predictions,
            variance_content,
            variance_losses,
        )

    def _process_content(
        self,
        content,
        content_lengths,
        variance_embeddings,
        durations,
        text_lengths,
        spec_lengths,
        max_spec_len,
    ):
        def cat_tensor(content_, embedding_):
            if embedding_.shape[1] == 1:
                embedding_ = embedding_.expand(-1, content_.shape[1], -1)
            delta = content_.shape[1] - embedding_.shape[1]
            if abs(delta) > 1:
                print(
                    f"delta: {delta}, content.shape: {content_.shape}, embedding.shape: {embedding_.shape}"
                )
            if delta > 0:
                shape = list(embedding_.shape)
                shape[1] = delta
                embedding_ = torch.cat(
                    [embedding_, torch.zeros(shape).to(embedding_.device)], dim=1
                )
            if delta < 0:
                embedding_ = embedding_[:, :delta, :]
            return torch.cat([content_, embedding_], dim=-1)

        variance_params = self.va_variance_params
        attention_weights = None

        for name, embedding in variance_embeddings.items():
            if name not in variance_params:
                continue
            if variance_params[name].aggregate_by_tokens:
                for i in variance_params[name].cat_to_content:
                    content[i] = cat_tensor(content[i], embedding)
                for i in variance_params[name].overwrite_content:
                    content[i] = embedding
                    if variance_params[name].content_length_by == "spectrogram":
                        content_lengths[i] = spec_lengths
                    if variance_params[name].content_length_by == "text":
                        content_lengths[i] = text_lengths

        if durations is not None and durations.shape[1] > 1:
            length_regulator = self.length_regulators["durations"]

            for i in variance_params["durations"].cat_to_content:
                if content[i].shape[1] == 1:
                    continue

                content[i], attention_weights = length_regulator(
                    content[i],
                    durations,
                    max_spec_len,
                    upsample_x2=True,
                )
                content_lengths[i] = spec_lengths

        for name, embedding in variance_embeddings.items():
            if name not in variance_params:
                continue
            if not variance_params[name].aggregate_by_tokens:
                for i in variance_params[name].cat_to_content:
                    content[i] = cat_tensor(content[i], embedding)
                for i in variance_params[name].overwrite_content:
                    content[i] = embedding
                    if embedding.shape[1] != max_spec_len:
                        content_lengths[i] = text_lengths
                    else:
                        content_lengths[i] = spec_lengths

        return content, content_lengths, attention_weights

    def forward_step(self, inputs: EncoderOutput) -> VarianceAdaptorOutput:  # type: ignore
        x_duration, x_adaptor = self._get_x(inputs)
        x_duration_lengths, x_adaptor_lengths = self._get_x_lengths(inputs)

        model_inputs = inputs.model_inputs
        text_lengths = model_inputs.transcription_lengths
        spec_lengths = model_inputs.spectrogram_lengths
        max_spec_len = torch.max(spec_lengths)
        if text_lengths is None:
            text_lengths = spec_lengths

        x_duration_mask = get_mask_from_lengths(x_duration_lengths)
        x_adaptor_mask = get_mask_from_lengths(x_adaptor_lengths)
        text_mask = get_mask_from_lengths(text_lengths)
        spec_mask = get_mask_from_lengths(spec_lengths)

        variance_names = self.va_variances
        variance_params = self.va_variance_params
        targets = self._get_targets(inputs, variance_names, variance_params)

        (
            durations,
            durations_prediction,
            durations_content,
            durations_loss,
        ) = self._process_durations(
            x_duration, x_adaptor, x_duration_mask, x_adaptor_mask, targets, model_inputs
        )

        (
            variance_embeddings,
            variance_predictions,
            variance_content,
            variance_losses,
        ) = self._process_variance(
            x_duration,
            x_adaptor,
            durations,
            targets,
            x_duration_mask,
            x_adaptor_mask,
            spec_mask,
            max_spec_len,
            model_inputs,
        )

        if durations_prediction is not None:
            variance_predictions["durations"] = durations_prediction
            variance_predictions["durations_denormalize"] = durations
            variance_content["durations"] = durations_content
            variance_losses["durations"] = durations_loss

        content, content_lengths, attention_weights = self._process_content(
            [x_duration, x_adaptor],
            [x_duration_lengths, x_adaptor_lengths],
            variance_embeddings,
            durations,
            text_lengths,
            spec_lengths,
            max_spec_len,
        )

        additional_content = merge_additional_outputs(
            inputs.additional_content, variance_embeddings
        )
        additional_content = merge_additional_outputs(
            additional_content, tuple(variance_content.values())
        )
        additional_losses = merge_additional_outputs(
            inputs.additional_losses, tuple(variance_losses.values())
        )

        masks = {
            "src": x_adaptor_mask,
            "text": text_mask,
            "spec": spec_mask,
        }

        return VarianceAdaptorOutput(
            content=content,
            content_lengths=content_lengths,
            masks=masks,
            attention_weights=attention_weights,
            variance_predictions=variance_predictions,
            embeddings=inputs.embeddings,
            model_inputs=inputs.model_inputs,
            additional_content=additional_content,
            additional_losses=additional_losses,
        )

    def get_spec_mask(self, durations, inputs):
        if durations is not None:
            if durations.shape[1] > 1:
                if getattr(inputs.model_inputs, "rate_modifier", None) is not None:
                    durations *= inputs.model_inputs.rate_modifier  # type: ignore
                    inputs.model_inputs.rate_modifier = None
        else:
            variance_predictions = getattr(inputs, "variance_predictions", None)
            if variance_predictions is not None and "durations" in variance_predictions:
                durations = torch.expm1(variance_predictions["durations"])

        if durations is not None:
            (
                spec_lengths,
                max_spec_len,
                spec_mask,
            ) = self._get_spec_len_from_duration(durations)
        else:
            spec_lengths = max_spec_len = spec_mask = None

        if spec_lengths is None:
            spec_lengths = inputs.model_inputs.output_lengths
            max_spec_len = spec_lengths.max()
            spec_mask = get_mask_from_lengths(spec_lengths)

        return spec_lengths, max_spec_len, spec_mask

    def generate_step(self, inputs: EncoderOutput, **kwargs) -> VarianceAdaptorOutput:  # type: ignore
        x_duration, x_adaptor = self._get_x(inputs)
        x_duration_lengths, x_adaptor_lengths = self._get_x_lengths(inputs)

        model_inputs = inputs.model_inputs
        src_lengths = x_duration_lengths
        text_lengths = model_inputs.transcription_lengths
        if text_lengths is None:
            text_lengths = model_inputs.output_lengths

        x_duration_mask = get_mask_from_lengths(x_duration_lengths)
        x_adaptor_mask = get_mask_from_lengths(x_adaptor_lengths)
        src_mask = get_mask_from_lengths(src_lengths)
        text_mask = get_mask_from_lengths(text_lengths)

        variance_names = self.va_variances
        variance_params = self.va_variance_params
        targets = self._get_targets(inputs, variance_names, variance_params)

        (
            durations,
            durations_prediction,
            durations_content,
            durations_loss,
        ) = self._process_durations(
            x_duration,
            x_adaptor,
            x_duration_mask,
            x_adaptor_mask,
            targets,
            model_inputs,
            **kwargs,
        )

        dura = inputs.additional_content.get("durations_postprocessed", durations)
        spec_lengths, max_spec_len, spec_mask = self.get_spec_mask(dura, inputs)

        (
            variance_embeddings,
            variance_predictions,
            variance_content,
            variance_losses,
        ) = self._process_variance(
            x_duration,
            x_adaptor,
            durations,
            targets,
            x_duration_mask,
            x_adaptor_mask,
            spec_mask,
            max_spec_len,
            model_inputs,
            **kwargs,
        )

        if durations_prediction is not None:
            variance_predictions["durations"] = durations_prediction
            variance_content["durations"] = durations_content
            variance_losses["durations"] = durations_loss

        content, content_lengths, attention_weights = self._process_content(
            [x_duration, x_adaptor],
            [x_duration_lengths, x_adaptor_lengths],
            variance_embeddings,
            durations,
            text_lengths,
            spec_lengths,
            max_spec_len,
        )

        additional_content = merge_additional_outputs(
            inputs.additional_content, variance_embeddings
        )
        additional_content = merge_additional_outputs(
            additional_content, tuple(variance_content.values())
        )
        additional_losses = merge_additional_outputs(
            inputs.additional_losses, tuple(variance_losses.values())
        )

        masks = {
            "src": x_adaptor_mask,
            "text": text_mask,
            "spec": spec_mask,
        }

        return VarianceAdaptorOutput(
            content=content,
            content_lengths=content_lengths,
            masks=masks,
            attention_weights=attention_weights,
            variance_predictions=variance_predictions,
            embeddings=inputs.embeddings,
            model_inputs=inputs.model_inputs,
            additional_content=additional_content,
            additional_losses=additional_losses,
        )
