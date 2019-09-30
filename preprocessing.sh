#!/bin/bash

#resizes the dataset images recursively in parallel batches

# Check number of arguments
if [ "$#" -lt 2 ]; then
	echo "Usage: bash preprocessing.sh <image-directory-path> <image-size(64x64)>"
	exit 1
fi

#CWD=$(pwd)
main_dir=$(realpath $1)
resize_param=$2
for dir in $main_dir/*
do
  echo "Processing $dir"
  find "$dir" -name '*.png' -execdir sh -c "mogrify -resize ${resize_param}\! *.png" {} \; &
  pids[$!]=$!
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

echo "Done!"