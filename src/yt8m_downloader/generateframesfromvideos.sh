# Check if FFMPEG is installed
FFMPEG=ffmpeg
command -v $FFMPEG >/dev/null 2>&1 || {
	echo >&2 "This script requires ffmpeg. Aborting."; exit 1;
}

# Check number of arguments
if [ "$#" -ne 4 ]; then
	echo "Usage: bash generateframesfromvideos.sh <path_to_directory_containing_videos> <path_to_directory_to_store_frames> <frames_format> <#frames_per_video>"
	exit 1
fi


# Parse videos and generate frames in a directory
for video in "$1"/*
do
	videoname=$(basename "${video}")
	videoname="${videoname%.*}"
	videoname=${videoname//[%]/x}
	seconds=$(ffmpeg -i "${video}" 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | sed 's@\..*@@g' | awk '{ split($1, A, ":"); split(A[3], B, "."); print 3600*A[1] + 60*A[2] + B[1] }')
	fps=$(bc <<< "scale=2; $4 / $seconds")
	#$(ffmpeg -i "${video}" 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p")")
	mkdir -p $2/"${videoname}"/frames
	$FFMPEG -i "${video}" -r ${fps} $2/"${videoname}"/frames/frame_%05d.$3
done
