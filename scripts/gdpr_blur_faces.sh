#!/bin/bash

# Path to the deface executable
deface_cmd="deface"

# Directory containing the folders with videos
parent_dir="videos"

# Loop through each folder
for folder in "$parent_dir"/*; do
    if [ -d "$folder" ]; then
        # Get the video file in the folder
        video_file=$(find "$folder" -maxdepth 1 -type f -name "*.mp4" -print -quit)

        if [ -n "$video_file" ]; then
            # Blur the video using deface
            output_file="$folder/blurred_video.mp4"
            $deface_cmd "$video_file" -o "$output_file"

            echo "Video in $folder blurred and saved as blurred_video.mp4"
        else
            echo "No video file found in $folder"
        fi
    fi
done