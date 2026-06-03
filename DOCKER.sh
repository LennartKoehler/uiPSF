#!/bin/bash

DOCKER_BUILDKIT=0 docker build -t cuda_exe .
mkdir -p test_output
docker run --rm -it --gpus all \
  -v ~/data/example_data_for_uiPSF:/app/example_data_for_uiPSF \
  -v "$(pwd)"/test_output:/app/test_output \
  cuda_exe
