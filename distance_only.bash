#!/bin/bash

# Default parameters
version=4;  # Version of the file
iter=100;  # Number of training-iterations each loop
gru=1;  # 0=False, 1=True
repr=1;  # 0=False, 1=True

# Run the program
for i in {1..10}
do
  python3 run_distance_only.py --gru_enabled=$gru --iterations=$iter --version=$version --reproduce=$repr;
done
