#!/bin/bash

# Default parameters
version=2;
iter=0;
blueprint=1;
eval=1;
trace=1;

# Sexual GRU
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=distance --blueprint=$blueprint --evaluate=$eval --trace=$trace;
python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=diversity --blueprint=$blueprint --evaluate=$eval --trace=$trace;
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=novelty --blueprint=$blueprint --evaluate=$eval --trace=$trace;
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=1 --fitness=path --blueprint=$blueprint --evaluate=$eval --trace=$trace;

# Asexual GRU
python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=distance --blueprint=$blueprint --evaluate=$eval --trace=$trace;
python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=diversity --blueprint=$blueprint --evaluate=$eval --trace=$trace;
python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=novelty --blueprint=$blueprint --evaluate=$eval --trace=$trace;
python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=1 --fitness=path --blueprint=$blueprint --evaluate=$eval --trace=$trace;

# Sexual non-GRU
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=distance --blueprint=$blueprint --evaluate=$eval --trace=$trace;
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=diversity --blueprint=$blueprint --evaluate=$eval --trace=$trace;
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=novelty --blueprint=$blueprint --evaluate=$eval --trace=$trace;
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=1 --gru_enabled=0 --fitness=path --blueprint=$blueprint --evaluate=$eval --trace=$trace;

# Asexual non-GRU
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=distance --blueprint=$blueprint --evaluate=$eval --trace=$trace;
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=diversity --blueprint=$blueprint --evaluate=$eval --trace=$trace;
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=novelty --blueprint=$blueprint --evaluate=$eval --trace=$trace;
#python3 run_population.py --version=$version --train=1 --iterations=$iter --reproduce=0 --gru_enabled=0 --fitness=path --blueprint=$blueprint --evaluate=$eval --trace=$trace;
