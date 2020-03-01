"""
inspect.py

Inspect a population of choice.
"""
import argparse
from collections import Counter

from population.population import Population


def count_genome_sizes(population):
    c = Counter()
    for g in population.population.values():
        c[g.size()] += 1
    return c


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pop', type=str, default='test')
    parser.add_argument('--count_size', type=bool, default=True)
    args = parser.parse_args()
    
    pop = Population(
            name=args.pop,
            # version=1,
    )
    # mut_idx = 2
    # print(list(pop.population.values())[mut_idx].size())
    # for _ in range(10):
    #     list(pop.population.values())[mut_idx].mutate(pop.config.genome_config)
    # print(list(pop.population.values())[mut_idx].size())
    
    if args.count_size:
        counter = count_genome_sizes(pop)
        for k, v in counter.items():
            print(f"Number of {k} shapes: {v}")
