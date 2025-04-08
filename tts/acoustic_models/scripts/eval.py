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
from tts.acoustic_models.interface.prosody_reference import REFERENECE_TYPE
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
    # vocos_checkpoint_epoch=40_step=370856_val_loss=5.7798.ckpt
    tts_model_path = "M:\\Ilya\\JustAI\\epoch=104-step=250050.ckpt"  # "M:\\Ilya\\JustAI\\vocos_checkpoint_epoch=40_step=370856_val_loss=5.7798.ckpt"
    tts_model_path = (
        "M:\\Ilya\\JustAI\\vocos_checkpoint_epoch=40_step=370856_val_loss=5.7798.ckpt"
    )
    voc_model_path = (
        "M:\\Ilya\\JustAI\\vocos_checkpoint_epoch=40_step=370856_val_loss=5.7798.ckpt"
    )
    # voc_model_path = "M:\\Ilya\\JustAI\\vocos_checkpoint_epoch=12_step=108342_val_loss=5.8525.ckpt"

    tts_model_path = (
        "M:\\Ilya\\JustAI\\vocos_checkpoint_epoch=18_step=395000_val_loss=6.4311.ckpt"
    )
    voc_model_path = (
        "M:\\Ilya\\JustAI\\vocos_checkpoint_epoch=18_step=395000_val_loss=6.4311.ckpt"
    )

    prosody_model_path = "M:\\Ilya\\JustAI\\07_Apr_2025_16_43_34_prosody_predictor_epoch=20_step=131250_category_EER=0.3664.ckpt"

    tts = TTSEvaluationInterface(
        tts_ckpt_path=tts_model_path,
        prosody_ckpt_path=prosody_model_path,
        device=device,
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
            "style_reference": Path("M:/Ilya/JustAI/30872.wav"),
            "utterances": """

    "Мы идем сейчас в парк?",
    "Не пойти ли нам погулять?",
    "Неужели нельзя было подготовиться к занятию лучше?",
    "Разве ты об этом ничего не читал?",
    "Разве можно останавливаться на полпути?",
    "Кто знает ответ на заданный вопрос?",
    "О чем вы тут шушукаетесь?",
    "Для чего используется этот прибор?",


            """,
        },
    ]

    tests[0][
        "utterances"
    ] = """
    Главная #особенность этой технологии —, создание замкнутого цикла обучения, где искусственный интеллект сам выступает и учеником, и учителем.
    Система работает по принципу внутренней обратной свя+зи: одна часть модели генерирует ответы, а другая выступает «судьей», оценивая их качество и соответствие заданным критэ+риям.
    Если ответ удовлетворяет требованиям, модель получает «вознаграждение» и запоминает успешную стратегию.
            """

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
