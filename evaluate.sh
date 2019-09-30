#!/bin/bash

# Check number of arguments
if [ "$#" -lt 2 ]; then
	echo "Usage: bash test.sh <config-file> <image-directory-path>"
	exit 1
fi

export PYTHONPATH=.

#evaluation set
echo "Running evaluation test..."
python3 ./src/run_evaluation.py --config $1 --data_path $2

#unit test
echo "Running unit test..."

python3 ./src/unit_test/classification_test.py

echo "Done!"