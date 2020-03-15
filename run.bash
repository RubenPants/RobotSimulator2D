#!/bin/bash

# Default parameters
iter=100
blueprint=1
trace=1

# Sexual GRU
#python3 run_population.py --train=1 --iterations=$iter --reproduce=1 --enable_gru=1 --fitness=distance --blueprint=$blueprint --trace=$trace;
#python3 run_population.py --train=1 --iterations=$iter --reproduce=1 --enable_gru=1 --fitness=path --blueprint=$blueprint --trace=$trace;
#python3 run_population.py --train=1 --iterations=$iter --reproduce=1 --enable_gru=1 --fitness=novelty --blueprint=$blueprint --trace=$trace;

# Asexual GRU
#python3 run_population.py --train=1 --iterations=$iter --reproduce=0 --enable_gru=1 --fitness=distance --blueprint=$blueprint --trace=$trace;
#python3 run_population.py --train=1 --iterations=$iter --reproduce=0 --enable_gru=1 --fitness=path --blueprint=$blueprint --trace=$trace;
#python3 run_population.py --train=1 --iterations=$iter --reproduce=0 --enable_gru=1 --fitness=novelty --blueprint=$blueprint --trace=$trace;

# Sexual non-GRU
python3 run_population.py --train=1 --iterations=$iter --reproduce=1 --enable_gru=0 --fitness=distance --blueprint=$blueprint --trace=$trace;
python3 run_population.py --train=1 --iterations=$iter --reproduce=1 --enable_gru=0 --fitness=path --blueprint=$blueprint --trace=$trace;
python3 run_population.py --train=1 --iterations=$iter --reproduce=1 --enable_gru=0 --fitness=novelty --blueprint=$blueprint --trace=$trace;

# Asexual non-GRU
python3 run_population.py --train=1 --iterations=$iter --reproduce=0 --enable_gru=0 --fitness=distance --blueprint=$blueprint --trace=$trace;
python3 run_population.py --train=1 --iterations=$iter --reproduce=0 --enable_gru=0 --fitness=path --blueprint=$blueprint --trace=$trace;
python3 run_population.py --train=1 --iterations=$iter --reproduce=0 --enable_gru=0 --fitness=novelty --blueprint=$blueprint --trace=$trace;
