#!/bin/bash

# Default parameters
version=2;
iter=100;
blueprint=1;
trace=1;

# Sexual GRU
python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=distance --blueprint=$blueprint --trace=$trace;
python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=path --blueprint=$blueprint --trace=$trace;
python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=novelty --blueprint=$blueprint --trace=$trace;

# Asexual GRU
python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=distance --blueprint=$blueprint --trace=$trace;
python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=path --blueprint=$blueprint --trace=$trace;
python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=novelty --blueprint=$blueprint --trace=$trace;

# Sexual non-GRU
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=distance --blueprint=$blueprint --trace=$trace;
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=path --blueprint=$blueprint --trace=$trace;
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=novelty --blueprint=$blueprint --trace=$trace;

# Asexual non-GRU
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=distance --blueprint=$blueprint --trace=$trace;
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=path --blueprint=$blueprint --trace=$trace;
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=novelty --blueprint=$blueprint --trace=$trace;
