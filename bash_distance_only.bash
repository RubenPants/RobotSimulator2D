#!/bin/bash

# Default parameters
version=2;  # Version of the file
iter=100;  # Number of training-iterations each loop
gru=1;  # 0=False, 1=True

# Run the program
for i in {1..5}
do
  python3 run_distance_only.py --gru_enabled=$gru --iterations=$iter --version=$version --reproduce=0;
#  python3 run_distance_only.py --gru_enabled=$gru --iterations=$iter --version=$version --reproduce=1;
done
