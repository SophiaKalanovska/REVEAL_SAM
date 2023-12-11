#!/bin/bash

# Directory containing images
IMAGE_DIR="examples/ILSVRC"
 echo "Processing $IMAGE_DIR"

# Iterate over jpg and png images in the directory
for image_file in $IMAGE_DIR/*; do
    # Check if the file exists (this avoids issues if there are no matches)
    echo "Processing $image_file"
    if [[ -f "$image_file" ]]; then
        novel_part=${image_file#$IMAGE_DIR/}
        
        echo "Processing $novel_part"
        returns= poetry run python3 examples/readme_code_snippet.py "$novel_part"
        # "$novel_part" + novel_part_gausian_big
        name_before_dot=$(echo $full_name | cut -d '.' -f 1)
        # Extract the name after the dot
        name_after_dot=$(echo $full_name | cut -d '.' -f 2)

        blur="${name_before_dot}_blur.${name_after_dot}"
        echo "Processing $blur"

        poetry run python3 examples/noise_execute_method.py "$blur" $returns[0] $returns[1] $returns[2] $returns[3] $returns[4] "ILSVRC_blur" "_blur"

        big="${name_before_dot}_gaussian_big.${name_after_dot}"
        echo "Processing $big"
        poetry run python3 examples/noise_execute_method.py "$big" $returns[0] $returns[1] $returns[2] $returns[3] $returns[4] "ILSVRC_big" "_big"

        small="${name_before_dot}_gaussian_small.${name_after_dot}"
        echo "Processing $small"
        poetry run python3 examples/noise_execute_method.py "$small" $returns[0] $returns[1] $returns[2] $returns[3] $returns[4] "ILSVRC_small" "_small"
    fi
done
