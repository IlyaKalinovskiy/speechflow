from pathlib import Path

from tqdm import tqdm

from speechflow.data_pipeline.datasample_processors.algorithms.audio_processing.audio_codecs import (
    DAC,
)
from speechflow.io import AudioChunk
from speechflow.logging import trace
from speechflow.utils.plotting import plot_tensor
from tts.acoustic_models.interface.eval_interface import TTSEvaluationInterface

if __name__ == "__main__":
    device = "cpu"

    vc_model_path = "C:/FluentaAI/epoch=34-step=350000.ckpt"
    ref_wav_path = "C:/FluentaAI/642111.wav"

    hubert_model_path = "C:/FluentaAI/hubert_phoneme_en/epoch=0-step=4500.ckpt"
    hubert_vocab_path = "C:/FluentaAI/hubert_phoneme_en/vocab.json"

    tts = TTSEvaluationInterface(
        tts_ckpt_path=vc_model_path,
        device_model=device,
        hubert_model_path=hubert_model_path,
        hubert_vocab_path=hubert_vocab_path,
    )
    # voc = VocoderEvaluationInterface(
    #     ckpt_path=voc_model_path,
    #     device=device,
    # )

    print(tts.get_languages())
    print(tts.get_speakers())

    test_files = "C:/FluentaAI/eng_spontan/wav_16k"
    result_path = "C:/FluentaAI/eng_spontan/result_xtts"

    file_list = list(Path(test_files).glob("*.wav"))
    Path(result_path).mkdir(parents=True, exist_ok=True)

    dac = DAC()

    for wav_path in tqdm(file_list):
        result_file_name = Path(result_path) / wav_path.name
        if result_file_name.exists():
            continue

        try:
            tts_out = tts.resynthesize(ref_wav_path, ref_wav_path, lang="EN")
            codes = tts_out.spectrogram.cpu()
            codes = codes.reshape(codes.shape[0], -1, 4)
            waveform = dac.decode_from_codes(codes.transpose(1, -1))
            AudioChunk(data=waveform[0].numpy(), sr=dac.sample_rate).save(
                "test.wav", overwrite=True
            )
            # plot_tensor(tts_out.spectrogram)
            break
        except Exception as e:
            print(trace("eval_interface", e))
