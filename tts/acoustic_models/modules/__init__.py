import tts.acoustic_models.modules.components.variance_adaptors as va

from tts.acoustic_models.modules import ada_speech, common, forward_tacotron, tacotron2
from tts.acoustic_models.modules.collection import ComponentCollection
from tts.acoustic_models.modules.components import (
    decoders,
    duration_predictors,
    encoders,
    forced_alignment,
    postnet,
    style_encoders,
    variance_predictors,
)
from tts.acoustic_models.modules.params import VarianceAdaptorParams
from tts.acoustic_models.modules.prosody import ProsodyEncoder, ProsodyEncoderParams

PARALLEL_ENCODERS = ComponentCollection()
PARALLEL_ENCODERS.registry_module(encoders, lambda x: "Encoder" in x)
PARALLEL_ENCODERS.registry_module(tacotron2, lambda x: "Encoder" in x)
PARALLEL_ENCODERS.registry_module(forward_tacotron, lambda x: "Encoder" in x)
PARALLEL_ENCODERS.registry_module(ada_speech, lambda x: "Encoder" in x)
PARALLEL_ENCODERS.registry_component(ProsodyEncoder, ProsodyEncoderParams)

PARALLEL_DECODERS = ComponentCollection()
PARALLEL_DECODERS.registry_module(decoders, lambda x: "Decoder" in x)
PARALLEL_DECODERS.registry_module(tacotron2, lambda x: "Decoder" in x)
PARALLEL_DECODERS.registry_module(forward_tacotron, lambda x: "Decoder" in x)
PARALLEL_DECODERS.registry_module(ada_speech, lambda x: "Decoder" in x)

PARALLEL_POSTNETS = ComponentCollection()
PARALLEL_POSTNETS.registry_module(postnet, lambda x: "Postnet" in x)
PARALLEL_POSTNETS.registry_module(tacotron2, lambda x: "Postnet" in x)
PARALLEL_POSTNETS.registry_module(forward_tacotron, lambda x: "Postnet" in x)

PARALLEL_ADAPTORS = ComponentCollection()
PARALLEL_ADAPTORS.registry_component(va.DummyVarianceAdaptor, VarianceAdaptorParams)
PARALLEL_ADAPTORS.registry_component(
    forward_tacotron.ForwardVarianceAdaptor, VarianceAdaptorParams
)

VARIANCE_PREDICTORS = ComponentCollection()
VARIANCE_PREDICTORS.registry_module(variance_predictors)
VARIANCE_PREDICTORS.registry_module(duration_predictors)
VARIANCE_PREDICTORS.registry_module(style_encoders)
VARIANCE_PREDICTORS.registry_component(
    forced_alignment.GradTTSFA, forced_alignment.GradTTSFAParams
)

LENGTH_REGULATORS = ComponentCollection()
LENGTH_REGULATORS.registry_component(common.LengthRegulator)
LENGTH_REGULATORS.registry_component(common.SoftLengthRegulator)
