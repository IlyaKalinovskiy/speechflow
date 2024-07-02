# Speechflow

### Installation



1. Clone a repository

```
git clone https://github.com/just-ai/speechflow
cd speechflow && git submodule update --init --recursive -f
```

On Ubuntu:

1. Installation [Singularity](https://docs.sylabs.io/guides/3.11/admin-guide/installation.html) (or run `env/singularity.sh`)
2. Run `install.sh`
3. Run singularity container `singularity shell --nv --writable --no-home -B /run/user/:/run/user/,.:/src --pwd /src torch_*.sif`
4. Activate conda environment `source /ext3/miniconda3/etc/profile.d/conda.sh && conda activate py38`

On Windows:
1. Install [Python 3.8](https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Windows-x86_64.exe)
2. Install additional packages `pip install -r requirements.txt`
3. Install submodules  `libs/install.sh`
4. Installations additional dependencies:
[.NET 5.0](https://dotnet.microsoft.com/en-us/download/dotnet/5.0),
[C++ Build Tools](https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/) or
[Visual Studio](https://visualstudio.microsoft.com/ru/downloads/),
[eSpeak](https://github.com/espeak-ng/espeak-ng),
[FFmpeg](https://github.com/icedterminal/ffmpeg-installer)

*For other systems see `env/Singularityfile`*
