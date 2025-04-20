FROM nvcr.io/nvidia/pytorch:21.03-py3

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/Moscow apt-get -y install tzdata

RUN apt-get install -y -qq sox zip unzip p7zip-full ffmpeg

RUN ln -s /opt/conda/bin/pip /usr/bin/pip
RUN ln -s /opt/conda/bin/conda /usr/bin/conda
RUN ln -s /opt/conda/bin/python /usr/bin/python
RUN ln -s /opt/conda/bin/cmake /usr/bin/cmake
RUN ln -s /opt/conda/bin/ninja /usr/bin/ninja
RUN ln -s /usr/local/cuda-11.4 /usr/local/cuda

RUN export PATH=$PATH:/usr/local/cuda/bin
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/lib
RUN export CPLUS_INCLUDE_PATH=/usr/local/cuda/include

# install dotnet
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
RUN dpkg -i packages-microsoft-prod.deb
RUN rm packages-microsoft-prod.deb

RUN apt-get install -y apt-transport-https && apt-get update
RUN apt-get install -y dotnet-sdk-5.0
RUN apt-get install -y aspnetcore-runtime-5.0
RUN apt-get install -y dotnet-runtime-5.0
RUN apt-get install -y nuget

# eSpeak
RUN apt-get install -y espeak-ng

# install conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
RUN rm Miniconda3-latest-Linux-x86_64.sh

# install SpeechFlow
RUN conda create -p /ext3/miniconda3/envs/py310 python=3.10
RUN source /ext3/miniconda3/etc/profile.d/conda.sh && conda activate py310 && pip install git+https://github.com/just-ai/speechflow.git
RUN source /ext3/miniconda3/etc/profile.d/conda.sh && conda activate py310 && pip install git+https://github.com/just-ai/multilingual-text-parser.git
RUN source /ext3/miniconda3/etc/profile.d/conda.sh && conda activate py310 && pip install fairseq==0.12.2 --no-deps
RUN source /ext3/miniconda3/etc/profile.d/conda.sh && conda activate py310 && pip install streamlit==1.44.1
RUN source /ext3/miniconda3/etc/profile.d/conda.sh && conda activate py310 && pip install st-annotated-text==4.0.2
