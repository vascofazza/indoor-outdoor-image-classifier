#!/bin/bash

#predicts the most probable label
# Check number of arguments
if [ "$#" -lt 2 ]; then
	echo "Usage: bash classify.sh <config-file> <input-image-path>"
	exit 1
fi

export PYTHONPATH=.

#evaluation set

python3 ./src/predict.py --config $1 --input $2