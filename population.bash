#!/bin/bash

# Default parameters
version=1;  # Version of the trained population
iter=100;  # Number of training iterations
blueprint=1;  # Create a blueprint each iter generations
eval=1;  # Evaluate the population each iter generations
trace=1;  # Trace the complete population each iter generations
trace_fit=1;  # Trace only the most fit genome each iter generations

for i in {1..10}
do
  # Sexual GRU
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=distance;
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=diversity;
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=novelty;
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=path;

  # Asexual GRU
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=distance;
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=diversity;
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=novelty;
  python3 run_population.py --version=$version --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=path;

  # Sexual non-GRU
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=distance;
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=diversity;
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=novelty;
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=path;

  # Asexual non-GRU
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=distance;
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=diversity;
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=novelty;
#  python3 run_population.py --version=$version --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=path;
done