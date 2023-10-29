#!/bin/bash

# Directory containing images
IMAGE_DIR="/Users/sophiakalanovska/Documents/REVEAL_SAM/examples/ILSVRC"
 echo "Processing $IMAGE_DIR"
# Iterate over jpg and png images in the directory
for image_file in $IMAGE_DIR/*; do
    # Check if the file exists (this avoids issues if there are no matches)
    echo "Processing $image_file"
    if [[ -f "$image_file" ]]; then
        novel_part=${image_file#$IMAGE_DIR/}
        
        echo "Processing $novel_part"
        python3 examples/readme_code_snippet.py "$novel_part"
    fi
done
