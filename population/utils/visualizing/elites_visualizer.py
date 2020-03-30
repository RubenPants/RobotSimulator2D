"""
elite_visualizer.py

Display a graph of the elites for each generation.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from population.population import Population
from population.utils.visualizing.averaging_functions import EMA, Forward, SMA
from utils.myutils import get_subfolder


def main(pop: Population, window: int = 5, show: bool = True):
    """
    Visualize the elites of the given population. Each generation, the average fitness of the three stored elites is
    taken.
    
    :param pop: Population object
    :param window: Window-size used in the function
    :param show: Show the result
    """
    get_subfolder(f'population/storage/{pop.folder_name}/{pop}/', 'images')
    get_subfolder(f'population/storage/{pop.folder_name}/{pop}/images/', 'elites')
    for func in [Forward, SMA, EMA]:
        # Fetch name based on used function
        name = f'{"EMA_" if func == EMA else "SMA_" if func == SMA else ""}gen_{pop.generation}'
        
        # Load in the relevant data
        history = sorted(pop.best_genome_hist.items(), key=lambda x: x[0])
        generations, elite_list = zip(*history)
        
        # Average the elite-fitness
        elite_fitness = [sum([e[1].fitness for e in e_list]) / len(e_list) for e_list in elite_list]
        assert len(elite_fitness) == len(generations)
        
        # Create the figure
        ax = plt.figure().gca()
        plt.plot(generations, func(elite_fitness, window))
        if func == SMA:
            plt.title(f"Elite fitness in population: {pop}\nSimple Moving Average (window={window})")
        elif func == EMA:
            plt.title(f"Elite fitness in population: {pop}\nExponential Moving Average (window={window})")
        else:
            plt.title(f"Elite fitness in population: {pop}")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
        plt.yticks([i / 10 for i in range(11)])  # Fitness expressed in range of 0..1 (hops of 0.1)
        plt.grid(axis='y')
        plt.tight_layout()
        
        # Save the result
        plt.savefig(f'population/storage/{pop.folder_name}/{pop}/images/elites/{name}')
        if show:
            plt.show()
        plt.close()


if __name__ == '__main__':
    os.chdir("../../../")
    
    population = Population(
            name='delta_distance_2',
            # version=1,
            folder_name='test',
            # folder_name='NEAT-GRU',
    )
    
    main(
            pop=population,
    )
