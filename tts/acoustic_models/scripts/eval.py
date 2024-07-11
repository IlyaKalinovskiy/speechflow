import typing as tp

from pathlib import Path

import torch
import numpy.typing as npt

from speechflow.utils.plotting import plot_spectrogram as plot_spectrogram_with_phonemes
from speechflow.utils.plotting import plot_tensor
from speechflow.utils.profiler import Profiler
from speechflow.utils.seed import get_seed
from tts.acoustic_models.interface.eval_interface import (
    TTSContext,
    TTSEvaluationInterface,
)
from tts.acoustic_models.models.prosody_reference import REFERENECE_TYPE
from tts.vocoders.eval_interface import VocoderEvaluationInterface


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
        voc_out = voc_interface.synthesize(
            tts_in,
            tts_out,
            lang=tts_ctx.prosody_reference.default.lang,
            speaker_name=tts_ctx.prosody_reference.default.speaker_name,
        )

    try:
        plot_tensor(tts_out.spectrogram)
        Profiler.sleep(1)
    except Exception as e:
        print(e)

    return voc_out.audio_chunk


if __name__ == "__main__":
    device = "cpu"

    tts_model_path = "epoch=24-step=104175.ckpt"
    voc_model_path = "vocos_checkpoint_epoch=59_step=1500000_val_loss=7.5391.ckpt"
    prosody_model_path = None

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
            "speaker_name": "Kontur",
            "style_reference": Path("374.wav"),
            "utterances": """

        Директор департамента финансовой стабильности ЦБ - Елизавета Данилова заявила, что в ноябре выдача льготной ипотеки, к примеру, сопоставима по объемам с октябрем, несмотря на действующие ограничения.
        В таких условиях ЦБ ничего не оставалось как ввести дестимулирующие меры. В документе регулятор пишет, что прибегнул к фактически запретительным мерам.
        Помимо роста доли закредитованных заемщиков рынок столкнулся с ценовым расслоением — разрыв цен на первичном и вторичном рынках недвижимости достиг 42%.
        Однако на фоне повышения ключевой ставки, ипотечное кредитование на вторичном рынке замедляется, что будет приводить к сокращению спроса и на первичном рынке.
        ЦБ не раз указывал на риски, связанные с перегревом рынка ипотечного кредитования, а также выступал с критикой льготных ипотечных программ, которые, по его мнению, «уместны только как антикризисная мера».
        Также именно с ипотекой регулятор связывал один из дисбалансов в экономике, так как «она накачана льготными и псевдольготными программами».
        В 2023 году Банк России всерьез взялся за охлаждение рынка ипотеки: регулятор повысил макронадбавки по ипотеке с низким первоначальным взносом и высокой долговой нагрузкой заемщиков.

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
            style_reference=test["style_reference"],
            seed=get_seed(),
        )
        audio_chunk.save(
            f"tts_result_{test['speaker_name']}_{idx}.wav",
            overwrite=True,
        )
