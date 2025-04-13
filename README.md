# SpeechFlow

A speech processing toolkit designed for easy configuration of complex speech data preparation pipelines and rapid prototyping of text-to-speech (TTS) models.

## Overview

This project provides a comprehensive solution for TTS development, featuring:
- [Multilingual text processing frontend](https://github.com/just-ai/multilingual-text-parser)
- Forced alignment models
- Modular framework for building TTS systems from reusable components

## News

- **April 2024 (v1.0):**
  - 🔥 Initial release of SpeechFlow 1.0!

## Installation

### Prerequisites
1. Install [Anaconda](https://www.anaconda.com/)
2. Clone a repository and update submodules

```
git clone https://github.com/just-ai/speechflow
cd speechflow
git submodule update --init --recursive -f
```

### On Ubuntu:

3. Install system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y libssl1.1 g++ wget sox ffmpeg
```

4. Configure Python environment:

```bash
conda create -n py310 python=3.10
conda activate py310
pip install -r requirements.txt
```

5. Install [multilingual frontend](https://github.com/just-ai/multilingual-text-parser) dependencies:

```bash
# Install .NET SDK
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb

sudo apt-get install -y apt-transport-https && apt-get update
sudo apt-get install -y dotnet-sdk-5.0 aspnetcore-runtime-5.0 dotnet-runtime-5.0 nuget

# install eSpeak
sudo apt-get install -y espeak-ng
```

6. Complete installation:

```bash
sh libs/install.sh
pytest tests  # Run verification tests
```

### On Windows:

1. Install [Python 3.10](https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Windows-x86_64.exe)

2. Install additional components:
   - [.NET 5.0 Runtime](https://dotnet.microsoft.com/en-us/download/dotnet/5.0),
   - [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/) or
   - [Microsoft Visual Studio](https://visualstudio.microsoft.com/ru/downloads/),
   - [eSpeak](https://github.com/espeak-ng/espeak-ng),
   - [FFmpeg](https://github.com/icedterminal/ffmpeg-installer)

3. Install Python packages:

```bash
pip install -r requirements.txt
sh libs/install.sh
```

### [Singularity Installation](https://docs.sylabs.io/guides/3.11/admin-guide/installation.html)

For containerized deployment:

```bash
sh env/singularity.sh  # Installs Singularity
sh install.sh
singularity shell --nv --writable --no-home -B .:/src --pwd /src torch_*.img
source /ext3/miniconda3/etc/profile.d/conda.sh && conda activate py310
```

## Data Annotation

Convert audio datasets to TextGrid format for TTS training.
Our annotation pipeline automates:
 - Audio segmentation into utterances
 - Text normalization and phonetic transcription
 - Forced alignment
 - Audio postprocessing (sample rate conversion, volume normalization)

### Supported Languages

RU, EN, IT, ES, FR-FR, DE, PT, PT-BR, KK (additional languages via [eSpeak-NG](https://github.com/espeak-ng/espeak-ng))

### Annotation Process

**1) Prepare Dataset Structure**

        dataset_root:
        - languages.yml
        - language_code_1
          - speakers.yml
          - dataset_1
            - speaker_1
              - file_1.wav
              - file_1.txt
              ...
              - file_n.wav
              - file_n.txt
              ...
              - speaker_n
                - file_1.wav
                - file_1.txt
                ...
                - file_n.wav
                - file_n.txt
          ...
          - dataset_n
        - language_code_n
          - speakers.yml
          - dataset_1
          ...

We recommend using normalized transcriptions that exclude numbers and abbreviations. For supported languages, [this package](https://github.com/just-ai/multilingual-text-parser) will automatically handle text normalization.

Transcription files are optional. If only audio files are provided, transcriptions will be generated automatically using the [Whisper Large v2](https://huggingface.co/openai/whisper-large-v2) ASR model.

For optimal processing, split large audio files into 20–30 minute segments.

The tool supports annotation of datasets with single or multiple speakers. To better understand the structure of source data directories and the formats of the [languages.yml](examples/simple_datasets/speech/SRC/languages.yml) and [speakers.yml](examples/simple_datasets/speech/SRC/EN/speakers.yml) configuration files, refer to the provided [example](examples/simple_datasets/speech/SRC).

**2) Run annotation processing**

The annotation process includes segmenting the audio file into single utterances, normalizing the text, generating a phonetic transcription, performing forced alignment of the transcription with the audio chunk, detecting silence, converting the audio sample rate, and equalizing the volume.

We provide pre-trained [multilingual forced alignment models](https://huggingface.co/IlyaKalinovskiy/multilingual-forced-alignment/tree/main/mfa_v1.0) at the phoneme level. These models were trained on 1,500 hours of audio (from over 8,000 speakers across 9 languages), including datasets such as LibriTTS, Hi-Fi TTS, VCTK, LJSpeech, and others.

Run this script to get segmentations:
```
# single GPU (the minimum requirement is 64GB RAM and 24GB VRAM)
python -m annotator.runner -d source_data_root -o segmentation_dataset_name -l=MULTILANG -ngpu=1 -nproc=16 -bs=16 --pretrained_models mfa_stage1_epoch=29-step=468750.pt mfa_stage2_epoch=29-step=312510.pt

# multi GPU (the minimum requirement is 256GB RAM and 24GB VRAM per GPU)
python -m annotator.runner -d source_data_root -o segmentation_dataset_name -l=MULTILANG -ngpu=4 -nproc=32 -ngw=8 --pretrained_models mfa_stage1_epoch=29-step=468750.pt mfa_stage2_epoch=29-step=312510.pt
```

To improve the alignment of your data, use the flag `--finetune_model`:
```
python -m annotator.runner -d source_data_root -o segmentation_dataset_name -l=MULTILANG -ngpu=1 -nproc=16 -bs=16 --finetune_model mfa_stage1_epoch=29-step=468750.pt
```

To process individual audio files, use [this interface](annotator/eval_interface.py).

The resulting segmentations can be opened in [Praat](https://www.fon.hum.uva.nl/praat/).
Additional examples are available [here](examples/simple_datasets/speech/SEGS).
![segmentation_example](docs/images/segmentation_example.jpg)

The alignment model is based on the [Glow-TTS](https://github.com/jaywalnut310/glow-tts) codebase.
Our implementation can be reviewed [here](tts/forced_alignment/model/glow_tts.py).

## Training TTS

### Training acoustic models

1. Build a dump with precompute features for TTS task.

Calculating certain features (e.g., biometric vectors or SSL features) can be computationally expensive.
To optimize batch processing, we precompute these features using a GPU for each data sample and store them on disk.
For details about which handlers are cached, refer to the [dump section](tts/acoustic_models/configs/tts/tts_data_24khz.yml#L160).

```
# single GPU
python -m tts.acoustic_models.scripts.dump
            -cd tts/acoustic_models/configs/tts/tts_data_24khz.yml
            -nproc=5 -ngpu=1
            [-vs <languade>]

# multi GPU
python -m tts.acoustic_models.scripts.dump
            -cd tts/acoustic_models/configs/tts/tts_data_24khz.yml
            -nproc=20 -ngpu=4
            [-vs <languade>]
```

2. Training a Conditional Flow Matching (CFM) model

After the dump is created run the model training.

```
python -m tts.acoustic_models.scripts.train
           -cd tts/acoustic_models/configs/tts/tts_data_24khz.yml
           -c tts/acoustic_models/configs/tts/cfm_model.yml
           [-vs <languade>]
```

### Training vocoders

You can use [BigVGANv2](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x) for convert the output mel-spectrogram of acoustic model into an audio signal.
However, we recommend fine-tuning this vocoder for your voices.

```
python -m tts.vocoders.scripts.train
           -cd tts/vocoders/configs/vocos/mel_bigvgan_data_24khz.yml
           -c tts/vocoders/configs/vocos/mel_bigvgan.yml
```

### Training end-to-end TTS

You can also perform joint training of the acoustic model and the vocoder with GAN-like sheme.
Please note that the current batch size is selected for single A100 80GB GPU.

```
python -m tts.vocoders.scripts.train
           -cd tts/vocoders/configs/vocos/voc_data_24khz.yml
           -c tts/vocoders/configs/vocos/styletts2_bigvgan.yml
```

* *prebuilt feature extraction dump for the voc_data_24khz.yml configuration file*

### Training expressive TTS

You can build a prosodic model to enhance the expressiveness of synthetic voices. For further details on this method, please refer to our [paper](https://www.isca-archive.org/interspeech_2024/korotkova24_interspeech.html#).

1. Build a dump of the required features

```
python -m tts.acoustic_models.scripts.dump
            -cd tts/acoustic_models/configs/prosody/prosody_data_24khz.yml
            -nproc=20 -ngpu=4
            [-vs <languade>]
```

2. Training prosody model

```
python -m tts.acoustic_models.scripts.train
            -cd tts/acoustic_models/configs/prosody/prosody_data_24khz.yml
            -c tts/acoustic_models/configs/prosody/prosody_model.yml
            [-vs <languade>]
```

3. Update datasets

```
python -m tts.acoustic_models.scripts.prosody_annotation
            -ckpt /path/to/prosody_model_checkpoint
            --textgrid_ext_new .TextGridStage3
            -md=cuda -nproc=5 -ngpu=1
            [-vs <languade>]
```

4. Training TTS models

   Similar to the steps discussed above...


5. Training prosody prediction model using text

```
python -m nlp.prosody_prediction.scripts.train
            -cd nlp/prosody_prediction/configs/data.yml
            -c nlp/prosody_prediction/configs/model.yml
            [-vs <languade>]
```

### Inference example

See [eval.py](tts/acoustic_models/scripts/eval.py#L107)

## BibTeX
```
@inproceedings{korotkova24_interspeech,
  title     = {Word-level Text Markup for Prosody Control in Speech Synthesis},
  author    = {Yuliya Korotkova and Ilya Kalinovskiy and Tatiana Vakhrusheva},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {2280--2284},
  doi       = {10.21437/Interspeech.2024-715},
  issn      = {2958-1796},
}
```
