#!/bin/bash

# Check number of arguments
if [ "$#" -lt 2 ]; then
	echo "Usage: bash train.sh <config-file> <image-directory-path>"
	exit 1
fi

export PYTHONPATH=.

python3 ./src/train.py --config $1 --data_path $2 --resume_training