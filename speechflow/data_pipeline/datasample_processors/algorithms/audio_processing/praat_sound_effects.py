import os
import sys
import uuid
import logging
import tempfile
import subprocess

from pathlib import Path

from speechflow.io.audio_io import AudioChunk
from speechflow.utils.fs import get_root_dir

__all__ = ["PraatSoundEffects"]

LOGGER = logging.getLogger("root")


class PraatSoundEffects:
    def __init__(self):
        self._praat_root = get_root_dir() / "speechflow/data/praat"
        self._plugin_path = self._praat_root / "plugin_VocalToolkit"
        if sys.platform == "win32":
            self._praat_path = self._praat_root / "win32" / "Praat.exe"
        else:
            self._praat_path = self._praat_root / "linux" / "praat_nogui"
            os.chmod(self._praat_path, 0o777)

    def whisper(self, audio_chunk: AudioChunk) -> AudioChunk:
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                wav_path = Path(tmp_dir) / f"{uuid.uuid4()}.wav"
                audio_chunk.save(wav_path)

                praat_script = f"""

                filename$ = "{wav_path.as_posix()}"
                sounds = Read from file: filename$
                plusObject: sounds

                # Run your script
                runScript: "{self._plugin_path.as_posix()}/whisper.praat"
                # where the ... is the list of arguments your script expects

                Write to WAV file... 'filename$'_whsp.wav

                """

                script_path = Path(tmp_dir) / "convert.praat"
                script_path.write_text(praat_script, encoding="utf-8")
                subprocess.call(
                    [self._praat_path.as_posix(), "--run", script_path.as_posix()]
                )
                whisper_sound = AudioChunk(f"{wav_path.as_posix()}_whsp.wav").load()
                audio_chunk.data = whisper_sound.data
            except Exception as e:
                LOGGER.error(e)

        return audio_chunk


if __name__ == "__main__":
    from speechflow.utils.profiler import Profiler

    sound_effects = PraatSoundEffects()

    _wav_path = get_root_dir() / "tests/data/test_audio.wav"
    _audio_chunk = AudioChunk(_wav_path).load()

    with Profiler():
        _audio_chunk = sound_effects.whisper(_audio_chunk)

    _audio_chunk.save("test_whisper.wav", overwrite=True)
