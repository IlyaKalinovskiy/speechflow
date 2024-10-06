#!/bin/bash

IMG_NAME="torch_$(date '+%d%m%Y').img"

echo "### Build singularity container ###"
sudo env "PATH=$PATH" singularity build --sandbox $IMG_NAME env/Singularityfile

echo "### Install python packages ###"
singularity run --writable --no-home -B .:/src --pwd /src $IMG_NAME pip install -r requirements.txt
singularity run --writable --no-home -B .:/src --pwd /src $IMG_NAME sh libs/install.sh
