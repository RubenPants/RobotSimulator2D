# Robot Simulator 2D
 Custom low-level robot simulator.


## TODO

* Keep best three genomes of each generation (store in population)
* Self-adaptive NEAT?
* Let make_net only use the active inputs (unchanged during run!)
* Species size-distribution seems off?
* Test all of the fitness-functions!
* Novelty not in crows-flight? (do intersection with walls! i.e. find reachable neighbours, then k-NN on those)
* What is the purpose of the gene-attributes? Figure out and write down! --> I should pay more focus on what to write down such that no redundant work is needed...
* Activation functions of hidden vs output nodes not taken into account! (hard-coded on tanh!)
* Update fitness functions to be more conform to that of James
* NEAT and variants implementation:
    * Implement Hyper-NEAT
    * Implement adaptive Hyper-NEAT



## Game Environment

The game environment is heavily inspired by OpenAI's *gym* environments. Each OpenAI gym has the following methods:

* `make` Create a new environment
* `reset` Prepare the environment for a new session
* `render` Visualize the current state
* `step` Progress the game by one and return current state
* `close` Stop the game and return final statistics

These are translated to my implementation as follows (respectively):

* `__init__` The game initialization functions as the `make` method.
* `reset` Same functionality.
* `render` Sequentially update the game whilst visualizing it. Since Pyglet doesn't handle Threads well, the agent must be given to the game.
* `step` Same functionality.
* `close` Unused.

### Step

The `step` call returns four values:

* `observation` Environment-specific object representing the observation of the environment (e.g. sensor readings).
* `reward` Amount of reward achieved by the previous action.
* `done` Boolean that says whether or not it's time to reset the environment again. If True, this indicates that the episode has terminated.
* `info` Diagnostic information useful for debugging, but may not be used by the agent.

![environment_loop](img/openai_environment_loop.png)

### Reset

The whole process gets started by calling `reset`,which returns an initial `observation`.

## CyEntities

Do the build inside of each cy-folder. This build file is responsible for moving all the cy-files to the correct folder!
