"""
main.py

Run a single population on one or more of the provided functionalities.
"""
import argparse
import traceback

from population.population import Population


def train(population: Population, unused_cpu, iterations, debug: bool = False):
    """Train the population on the requested number of iterations."""
    from environment.env_training import TrainingEnv
    population.log("\n===> TRAINING <===\n")
    trainer = TrainingEnv(
            unused_cpu=unused_cpu,  # Use two cores less to keep laptop usable
            game_config=population.game_config,
    )
    trainer.evaluate_and_evolve(
            pop=population,
            n=iterations,
            parallel=not debug,
    )


def train_same_game(games: list, population: Population, unused_cpu, iterations, debug: bool = False):
    """Train the population on the same set of games for the requested number of iterations."""
    from environment.env_training import TrainingEnv
    population.log("\n===> TRAINING <===\n")
    trainer = TrainingEnv(
            unused_cpu=unused_cpu,  # Use two cores less to keep laptop usable
            game_config=population.game_config,
    )
    trainer.evaluate_same_game_and_evolve(
            pop=population,
            games=games,
            n=iterations,
            parallel=not debug,
            random_init=True,
    )


def blueprint(population: Population, games: list):
    """Create a blueprint evaluation for the given population on the first 5 games."""
    from environment.env_visualizing import VisualizingEnv
    population.log("\n===> CREATING BLUEPRINTS <===\n")
    visualizer = VisualizingEnv(game_config=population.game_config)
    population.log(f"Creating blueprints for games: {games}")
    visualizer.set_games(games)
    visualizer.blueprint_genomes(pop=population)


def trace(population: Population, games: list):
    """Create a trace evaluation for the given population on the first 5 games."""
    from environment.env_visualizing import VisualizingEnv
    population.log("\n===> CREATING TRACES <===\n")
    visualizer = VisualizingEnv(game_config=population.game_config)
    population.log(f"Creating traces for games: {games}")
    visualizer.set_games(games)
    visualizer.trace_genomes(pop=population)


def evaluate(population: Population):
    """Evaluate the given population on the evaluation game-set."""
    population.log("\n===> EVALUATING <===\n")
    from environment.env_evaluation import EvaluationEnv
    evaluator = EvaluationEnv(game_config=population.game_config)
    genomes = sorted([g for g in population.population.values()],
                     key=lambda x: x.fitness if x.fitness else 0,
                     reverse=True)
    evaluator.evaluate_genome_list(
            genome_list=genomes[:10],  # Evaluate the five best performing genomes
            pop=population,
    )


def visualize_genome(population: Population, genome, debug: bool = True):
    """Visualize the requested genome."""
    print("\n===> VISUALIZING GENOME <===\n")
    print(f"Genome {genome.key} with size: {genome.size()}")
    population.visualize_genome(
            debug=debug,
            genome=genome,
    )


def live(game_id: int, population: Population, genome, debug: bool = False, random_init: bool = False):
    """Create a live visualization for the performance of the given genome."""
    from environment.env_visualizing_live import LiveVisualizer
    
    print("\n===> STARTING LIVE DEMO <===\n")
    print(f"Genome {genome.key} with size: {genome.size()}")
    visualizer = LiveVisualizer(
            pop=population,
            debug=debug,
            # speedup=1,
    )
    
    visualizer.visualize(
            genome=genome,
            game_id=game_id,
            random_init=random_init,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    
    # Main methods
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--train_same', type=bool, default=True)
    parser.add_argument('--blueprint', type=bool, default=False)
    parser.add_argument('--trace', type=bool, default=True)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--genome', type=bool, default=False)
    parser.add_argument('--live', type=bool, default=False)
    
    # Extra arguments
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--unused_cpu', type=int, default=2)
    parser.add_argument('--debug', type=bool, default=True)
    args = parser.parse_args()
    
    # Setup the population
    pop = Population(
            name='test',
            # version=1,
            folder_name='test',
    )
    if not pop.best_genome: pop.best_genome = list(pop.population.values())[0]
    # pop.best_genome = list(pop.population.values())[111]  # TODO
    # pop.population = {k: v for k, v in pop.population.items() if k in [111]}  # TODO
    
    # Set the blueprint and traces games
    chosen_games = [0]
    # chosen_games = [g for g in range(1, 6)]
    
    # Chosen genome used for genome-evaluation
    # g = list(pop.population.values())[2]
    chosen_genome = pop.best_genome
    
    try:
        if args.train:
            train(population=pop,
                  unused_cpu=args.unused_cpu,
                  iterations=args.iterations,
                  debug=args.debug,
                  )
        
        if args.train_same:
            train_same_game(games=chosen_games,
                            population=pop,
                            unused_cpu=args.unused_cpu,
                            iterations=args.iterations,
                            debug=args.debug,
                            )
        
        if args.blueprint:
            blueprint(population=pop, games=chosen_games)
        
        if args.trace:
            trace(population=pop, games=chosen_games)
        
        if args.evaluate:
            evaluate(population=pop)
        
        if args.genome:
            visualize_genome(population=pop,
                             genome=chosen_genome,
                             debug=True,  # TODO: args.debug
                             )
        
        if args.live:
            live(game_id=0,
                 population=pop,
                 genome=chosen_genome,
                 debug=args.debug,
                 random_init=True,  # TODO
                 )
    except Exception as e:
        pop.log(traceback.format_exc(), print_result=False)
        raise e
