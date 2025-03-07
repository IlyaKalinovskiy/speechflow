import typing as tp

from pathlib import Path

from speechflow.utils.plotting import plot_durations_and_signals, plot_tensor
from speechflow.utils.profiler import Profiler
from speechflow.utils.seed import get_seed
from tts.acoustic_models.data_types import TTSForwardInput, TTSForwardOutput
from tts.acoustic_models.interface.eval_interface import (
    TTSContext,
    TTSEvaluationInterface,
)
from tts.acoustic_models.models.prosody_reference import REFERENECE_TYPE
from tts.acoustic_models.modules.common.length_regulators import SoftLengthRegulator
from tts.vocoders.eval_interface import VocoderEvaluationInterface


def plotting(tts_in: TTSForwardInput, tts_out: TTSForwardOutput, doc, signals=("pitch",)):
    try:
        dura = tts_out.variance_predictions["durations"]
        max_len = tts_out.spectrogram[0].shape[0]

        if tts_out.spectrogram.shape[0] == 1:
            lr = SoftLengthRegulator(sigma=999999)

            signal = {}
            for name in signals:
                if name not in tts_out.variance_predictions:
                    continue

                signal[name] = tts_out.variance_predictions[name][0]

                name = f"aggregate_{name}"
                if name in tts_out.variance_predictions:
                    val = tts_out.variance_predictions[name]
                    val, _ = lr(val.unsqueeze(-1), dura, max_len)
                    signal[name] = val[0, :, 0]

            val = tts_in.ling_feat.breath_mask * (-1)
            val, _ = lr(val.unsqueeze(-1), dura, max_len)
            signal["breath_mask"] = val[0, :, 0]

            plot_durations_and_signals(
                tts_out.spectrogram[0],
                dura[0],
                doc.sents[0].get_phonemes(as_tuple=True),
                signal,
            )
        else:
            plot_tensor(tts_out.spectrogram)
    except Exception as e:
        print(e)
    finally:
        Profiler.sleep(1)


def synthesize(
    tts_interface: TTSEvaluationInterface,
    voc_interface: VocoderEvaluationInterface,
    text: tp.Union[str, Path],
    lang: str,
    speaker_name: tp.Optional[tp.Union[str, tp.Dict[str, str]]] = None,
    speaker_reference: tp.Optional[
        tp.Union[REFERENECE_TYPE, tp.Dict[str, REFERENECE_TYPE]]
    ] = None,
    style_reference: tp.Optional[
        tp.Union[REFERENECE_TYPE, tp.Dict[str, REFERENECE_TYPE]]
    ] = None,
    use_profiler: bool = False,
    seed: int = 0,
):
    tts_ctx = TTSContext.create(
        lang, speaker_name, speaker_reference, style_reference, seed
    )
    tts_ctx = tts_interface.prepare_embeddings(tts_ctx)

    doc = tts_interface.prepare_text(text, lang)
    doc = tts_interface.predict_prosody_by_text(doc, tts_ctx)

    tts_in = tts_interface.prepare_batch(
        tts_interface.split_sentences(doc)[0],
        tts_ctx,
    )

    with Profiler(enable=use_profiler):
        tts_out = tts_interface.evaluate(tts_in, tts_ctx)

        if voc_interface is not None:
            voc_out = voc_interface.synthesize(
                tts_in,
                tts_out,
                lang=tts_ctx.prosody_reference.default.lang,
                speaker_name=tts_ctx.prosody_reference.default.speaker_name,
            )
        else:
            voc_out = None

    plotting(tts_in, tts_out, doc)
    return voc_out.audio_chunk


if __name__ == "__main__":
    device = "cpu"

    tts_model_path = "M:\\Илья\\JustAI\\epoch=19-step=125000.ckpt"
    voc_model_path = (
        "M:\\Илья\\JustAI\\vocos_checkpoint_epoch=1_step=16668_val_loss=5.8527.ckpt"
    )
    prosody_model_path = None

    tts = TTSEvaluationInterface(
        tts_ckpt_path=tts_model_path,
        prosody_ckpt_path=prosody_model_path,
        device_model=device,
    )
    voc = VocoderEvaluationInterface(
        ckpt_path=voc_model_path,
        device=device,
    )

    print(tts.get_languages())
    print(tts.get_speakers())

    tests = [
        {
            "lang": "RU",
            "speaker_name": "Ksyusha",
            "style_reference": Path("M:/Илья/JustAI/30872.wav"),
            "utterances": """

            С нача+ла следующего учебного года в школах Петербурга планируют увеличить стоимость школьных обедов, изменение цен затронет все категории школьников.

            """,
        },
    ]

    for idx, test in enumerate(tests):
        audio_chunk = synthesize(
            tts,
            voc,
            test["utterances"],
            test["lang"],
            speaker_name=test["speaker_name"],
            speaker_reference=test["style_reference"],
            style_reference=test["style_reference"],
            seed=get_seed(),
        )
        audio_chunk.save(
            f"tts_result_{test['speaker_name']}_{idx}.wav",
            overwrite=True,
        )
