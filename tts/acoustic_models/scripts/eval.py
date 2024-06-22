import pickle
import typing as tp

from pathlib import Path

import torch
import numpy.typing as npt

from speechflow.io import AudioChunk
from speechflow.utils.plotting import plot_spectrogram as plot_spectrogram_with_phonemes
from speechflow.utils.plotting import plot_tensor
from speechflow.utils.profiler import Profiler
from speechflow.utils.seed import get_seed
from tts.acoustic_models.data_types import TTSForwardInput, TTSForwardOutput
from tts.acoustic_models.interface.eval_interface import (
    TTSContext,
    TTSEvaluationInterface,
    TTSOptions,
)
from tts.acoustic_models.models.prosody_reference import REFERENECE_TYPE
from tts.vocoders.data_types import VocoderInferenceOutput
from tts.vocoders.eval_interface import VocoderEvaluationInterface, VocoderOptions


def _plot_spectrogram(
    spec: tp.Union[npt.NDArray, torch.Tensor], dura=None, symbols=None, pitch=None
):
    import matplotlib

    matplotlib.use("TkAgg")

    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()[0].transpose()

    if dura is not None:
        dura = dura.cpu().cumsum(0).long().numpy().tolist()
        symbols = list(symbols[1:]) + [symbols[0]]
        pitch = pitch.cpu().numpy()[: spec.shape[1]]
        pitch = pitch / pitch.max() * (spec.shape[0] // 2)
        plot_spectrogram_with_phonemes(spec, symbols, dura, signal=pitch, dont_close=True)
    else:
        plot_spectrogram_with_phonemes(spec, dont_close=True)


def prepare_output(
    tts_in: TTSForwardInput,
    tts_out: TTSForwardOutput,
    voc_out: tp.Union[VocoderInferenceOutput, tp.List[VocoderInferenceOutput]],
    sample_rate: int,
):
    voc_out = voc_out if isinstance(voc_out, list) else [voc_out]  # type: ignore
    try:
        plot_tensor(tts_out.spectrogram)
        pass
    except Exception as e:
        print(e)

    waveform = []
    for out in voc_out[0][0]:
        waveform.append(out.cpu())
    waveform = torch.cat(waveform)

    return AudioChunk(data=waveform.numpy(), sr=sample_rate), tts_in, tts_out, voc_out


def synthesize(
    text: tp.Union[str, Path],
    lang: str,
    tts_interface: TTSEvaluationInterface,
    voc_interface: VocoderEvaluationInterface,
    speaker_name: tp.Optional[tp.Union[str, tp.Dict[str, str]]] = None,
    speaker_reference: tp.Optional[
        tp.Union[REFERENECE_TYPE, tp.Dict[str, REFERENECE_TYPE]]
    ] = None,
    style_reference: tp.Optional[
        tp.Union[REFERENECE_TYPE, tp.Dict[str, REFERENECE_TYPE]]
    ] = None,
    tts_ctx: tp.Optional[TTSContext] = None,
    tts_opt: tp.Optional[TTSOptions] = None,
    voc_opt: tp.Optional[VocoderOptions] = None,
    use_profiler: bool = False,
    seed: int = 0,
):
    print("seed", seed)
    if tts_ctx is None:
        tts_ctx = TTSContext.create(
            speaker_name, speaker_reference, style_reference, seed
        )
    if tts_opt is None:
        tts_opt = TTSOptions()
    if voc_opt is None:
        voc_opt = VocoderOptions()

    voc_opt.lang = lang
    voc_opt.speaker_name = tts_ctx.prosody_reference.default.speaker_name

    tts_ctx = tts_interface.prepare_embeddings(
        lang,
        tts_ctx,
        tts_opt,
    )

    doc = tts_interface.prepare_text(text, lang, tts_opt)
    doc = tts_interface.predict_prosody_by_text(doc, tts_ctx, tts_opt)
    tts_in = tts_interface.prepare_batch(
        tts_interface.split_sentences(doc)[0],
        tts_ctx,
        tts_opt,
    )

    with Profiler(enable=use_profiler):
        tts_out = tts_interface.evaluate(tts_in, tts_ctx, tts_opt)
        # _plot_spectrogram(tts_out.after_postnet_spectrogram)
        tts_in.speaker_emb = (
            torch.from_numpy(tts_ctx.prosody_reference.default.speaker_bio_emb)
            .unsqueeze(0)
            .expand((2, -1))
        )
        voc_out = voc_interface.synthesize(
            tts_in,
            tts_out,
            voc_opt,
        )

    return prepare_output(
        tts_in, tts_out, voc_out, 24000
    )  # voc_interface.output_sample_rate)


if __name__ == "__main__":
    device = "cpu"

    tts_model_path = Path("C:\\SRS\\data\\tts\\epoch=59-step=125040.ckpt")
    voc_model_path = Path(
        "C:\\SRS\\data\\tts\\vocos_checkpoint_epoch=2_step=75000_val_loss=8.7801.ckpt"
    )
    prosody_model_path = None  # Path("P:\\cfm\\prosody_ru\\epoch=14-step=7034.ckpt")

    voc = VocoderEvaluationInterface(
        ckpt_path=voc_model_path,
        device=device,
    )
    tts = TTSEvaluationInterface(
        tts_ckpt_path=tts_model_path,
        prosody_ckpt_path=prosody_model_path,
        device=device,
    )

    print(tts.get_languages())
    print(tts.get_speakers())

    tests = [
        {
            "lang": "RU",
            "speaker_name": "Kontur",  # "Madina",
            "source_speaker_name": "Kontur",  # "Madina",
            "utterances": """
«Норильский никель» — диверсифицированная го+рно металлургическая компания, являющаяся крупнейшим в мире производителем палладия и высокосортного никеля!
Производственные подразделения группы компаний «Норильский никель» расположены в России в Норильском промышленном районе, на Кольском полуострове и в Забайкальском крае, а также в Финляндии.
            """,
        },
    ]

    for test in tests:
        for i in range(1):
            if test["lang"] not in tts.get_languages():
                continue

            # test["speaker_name"] = random.choice(tts.get_speakers())
            # print(test["speaker_name"])

            wave_chunk, tts_in, tts_out, voc_out = synthesize(
                test["utterances"],
                test["lang"],
                tts,
                voc,
                speaker_name={
                    "default": test["speaker_name"],
                    "vq_encoder": test["source_speaker_name"],
                },
                style_reference=Path("C:\\SRS\\5.wav"),
                seed=get_seed(),
            )
            wave_chunk.save(
                f"vc_rnd_ru_v3_{i}.wav",
                overwrite=True,
            )
