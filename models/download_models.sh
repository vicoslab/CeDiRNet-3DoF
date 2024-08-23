#!/bin/bash

# Define the URLs for the models
MODELS=(
    "https://box.vicos.si/skokec/rtfm/CeDiRNet-3DoF/ConvNext-L-RGB.pth"
    "https://box.vicos.si/skokec/rtfm/CeDiRNet-3DoF/ConvNext-L-RGB-D.pth"
    "https://box.vicos.si/skokec/rtfm/CeDiRNet-3DoF/ConvNext-B-RGB.pth"
    "https://box.vicos.si/skokec/rtfm/CeDiRNet-3DoF/ConvNext-B-RGB-D.pth"
)

# Get the directory where the script is located
script_dir=$(dirname "$(realpath "$0")")

# Download each model
for url in "${MODELS[@]}"; do
    wget -P "$script_dir" "$url"
done

