from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import (
    ComponentOutput,
    DecoderOutput,
    VarianceAdaptorOutput,
)
from tts.acoustic_models.modules.params import DecoderParams

__all__ = [
    "WrapperDecoder",
    "WrapperDecoderParams",
]


class WrapperDecoderParams(DecoderParams):
    base_decoder_type: str = "RNNEncoder"  # type: ignore
    base_decoder_params: dict = None  # type: ignore


class WrapperDecoder(Component):
    """WrapperDecoder."""

    params: WrapperDecoderParams

    def __init__(self, params: WrapperDecoderParams, input_dim):
        super().__init__(params, input_dim)

        from tts.acoustic_models.modules import PARALLEL_DECODERS, PARALLEL_ENCODERS

        if params.base_decoder_type in PARALLEL_ENCODERS:
            components = PARALLEL_ENCODERS
        elif params.base_decoder_type in PARALLEL_DECODERS:
            components = PARALLEL_DECODERS
        else:
            raise RuntimeError(f"Component '{params.base_decoder_type}' not found")

        dec_cls, dec_params_cls = components[params.base_decoder_type]
        dec_params = dec_params_cls.init_from_parent_params(
            params, params.base_decoder_params
        )

        if components == PARALLEL_ENCODERS:
            dec_params.encoder_num_blocks = params.decoder_num_blocks
            dec_params.encoder_num_layers = params.decoder_num_layers
            dec_params.encoder_inner_dim = params.decoder_inner_dim
            dec_params.encoder_output_dim = params.decoder_output_dim
            dec_params.max_input_length = params.max_output_length

        self.decoder = dec_cls(dec_params, input_dim)

    @property
    def output_dim(self):
        return self.decoder.output_dim

    def forward_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        outputs: ComponentOutput = self.decoder(inputs)
        return DecoderOutput.copy_from(outputs)
