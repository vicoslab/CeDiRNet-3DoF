#!/bin/bash

# Define the URLs for the models
LOCALOZATION_MODEL=(
    "https://box.vicos.si/skokec/rtfm/CeDiRNet-3DoF/localization_checkpoint.pth"
)

# Get the directory where the script is located
script_dir=$(dirname "$(realpath "$0")")

# Download each model
for url in "${LOCALOZATION_MODEL[@]}"; do
    wget -P "$script_dir" "$url"
done

