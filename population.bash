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
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=distance --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=diversity --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=novelty --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=path --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;

  # Asexual GRU
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=distance --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=diversity --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=novelty --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;
  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=path --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;

  # Sexual non-GRU
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=distance --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=diversity --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=novelty --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=path --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;

  # Asexual non-GRU
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=distance --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=diversity --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=novelty --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;
#  python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=path --blueprint=$blueprint --evaluate=$eval --trace=$trace --trace_fit=$trace_fit;
done