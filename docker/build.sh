#!/bin/bash

docker build --network host --no-cache -t isaacgym -f docker/Dockerfile .
