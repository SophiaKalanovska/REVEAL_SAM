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

        # rm examples/temp_results.json
        
        echo "Processing $novel_part"

        returns=$(poetry run python3 examples/readme_code_snippet.py "$novel_part")

        # "$novel_part" + novel_part_gausian_big
        name_before_dot=$(echo $novel_part | cut -d '.' -f 1)
        # Extract the name after the dot
        name_after_dot=$(echo $novel_part | cut -d '.' -f 2)



        blur="${name_before_dot}_blur.${name_after_dot}"
        echo "Processing $blur"
        poetry run python3 examples/noise_execute_method.py "$blur" "ILSVRC_blur" "_blur"

        small="${name_before_dot}_gausian_small.${name_after_dot}"
        echo "Processing $small"
        poetry run python3 examples/noise_execute_method.py "$small" "ILSVRC_small" "_gausian_small"

        big="${name_before_dot}_gausian_big.${name_after_dot}"
        echo "Processing $big"
        poetry run python3 examples/noise_execute_method.py "$big" "ILSVRC_big" "_gausian_big"


        # big="${name_before_dot}_gausian_big.${name_after_dot}"
        # echo "Processing $big"
        # poetry run python3 examples/noise_execute_method.py "$big" "ILSVRC_big_normal" "_gausian_big"


        rm examples/temp_results.json


    fi
done
