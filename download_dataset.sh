#!/bin/bash

# Check number of arguments
if [ "$#" -lt 2 ]; then
	echo "Usage: bash download_dataset.sh <category_file> <num_video> <num_frames> <image-directory-path>"
	exit 1
fi

CWD=$(pwd)
CAT_FILE=$(realpath $1)
IMG_DIR=$(realpath $4)
#download videos first
#downloads <num_video> per category, according to <category file>
cd ./src/yt8m_downloader
bash downloadmulticategoryvideos.sh $2 $CAT_FILE

#extract <num_frames> for each video previously downloaded in PNG format
mkdir ${CWD}/frames
bash generateframesfromvideos.sh $IMG_DIR ${CWD}/frames png