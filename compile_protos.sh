#!/bin/bash

python3 -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/pose_interaction.proto
