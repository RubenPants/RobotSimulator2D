"""
specie_visualizer.py

Visualization of a specie's elites.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from population.population import Population
from population.utils.visualizing.averaging_functions import EMA, Forward, SMA
from utils.myutils import get_subfolder


def main(pop: Population, func, window: int = 5, show: bool = True):
    """
    Visualize the elites of the given population. Each generation, the average fitness of the three stored elites is
    taken.
    
    :param pop: Population object
    :param func: Function used to flatten the curve
    :param window: Window-size used in the function
    :param show: Show the result
    """
    # Fetch name based on used function
    name = f'elites{"_EMA" if func == EMA else "_SMA" if func == SMA else ""}_gen_{pop.generation}'
    
    ax = plt.figure().gca()
    max_gen = 0
    for specie_id, specie in pop.species_hist.items():
        # Fetch specie-data
        history = sorted(specie.items(), key=lambda x: x[0])
        generations, elite_list = zip(*history)
        
        # Update max_gen
        if generations[-1] > max_gen: max_gen = generations[-1]
        
        # Average the elite-fitness
        elite_fitness = [sum([e.fitness for e in e_list]) / len(e_list) for e_list in elite_list]
        assert len(elite_fitness) == len(generations)
        
        # Plot the specie
        plt.plot(generations, func(elite_fitness, window), label=f'specie {specie_id}')
    
    # Additional plot attributes
    if func == SMA:
        plt.title(f"Specie fitness in population: {pop}\nSimple Moving Average (window={window})")
    elif func == EMA:
        plt.title(f"Specie fitness in population: {pop}\nExponential Moving Average (window={window})")
    else:
        plt.title(f"Specie  fitness in population: {pop}")
    plt.xlabel("generation")
    plt.ylabel("fitness of specie's elites")
    plt.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
    plt.yticks([i / 10 for i in range(11)])  # Fitness expressed in range of 0..1 (hops of 0.1)
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Save the result
    get_subfolder(f'population/storage/{pop.folder_name}/{pop}/', 'images')
    get_subfolder(f'population/storage/{pop.folder_name}/{pop}/images/', 'species')
    plt.savefig(f'population/storage/{pop.folder_name}/{pop}/images/species/{name}')
    if show:
        plt.show()
        plt.close()


if __name__ == '__main__':
    os.chdir("../../../")
    
    population = Population(
            name='distance_repr_1',
            # version=1,
            folder_name='test',
    )
    
    for f in [Forward, SMA, EMA]:
        main(
                pop=population,
                func=f,
        )
