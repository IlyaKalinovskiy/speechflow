#!/bin/bash

IMG_NAME="torch_$(date '+%d%m%Y').sif"

echo "### Build singularity container ###"
sudo env "PATH=$PATH" singularity build $IMG_NAME env/Singularityfile

echo "### Create overlay file ###"
singularity overlay create --size 24576 $IMG_NAME

echo "### Install python packages ###"
singularity run --writable --no-home -B .:/src --pwd /src $IMG_NAME pip install -r requirements.txt
singularity run --writable --no-home -B .:/src --pwd /src $IMG_NAME sh libs/install.sh
