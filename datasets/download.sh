#!/bin/bash

CURDIR=$(dirname "$(realpath "$0")")

# download and extract ViCoS Towel Dataset
wget https://go.vicos.si/toweldataset -O - | unzip -d $CURDIR

# download and extract MuJoCo Dataset
wget https://go.vicos.si/towelmujocodataset -O - | unzip -d $CURDIR
