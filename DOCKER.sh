#!/bin/bash

DOCKER_BUILDKIT=0 docker build -t cuda_exe .
docker run --rm -it --gpus all   -e DISPLAY=$DISPLAY   -v /tmp/.X11-unix:/tmp/.X11-unix   -v ~/data/example_data_for_uiPSF:/app/example_data_for_uiPSF   cuda_exe
