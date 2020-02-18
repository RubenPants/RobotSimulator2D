# Pyglet - Foot-Bot Simulation
 Custom low-level robot simulator.



## Game Environment

The game environment is heavily inspired by OpenAI's *gym* environments. Each OpenAI gym has the following methods:

* `make` Create a new environment
* `reset` Prepare the environment for a new session
* `render` Visualize the current state
* `step` Progress the game by one and return current state
* `close` Stop the game and return final statistics

These are translated to my implementation as follows (respectively):

* `__init__` The game initialization functions as the `make` method.
* `reset` Same functionality.````````
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

## TODO - Urgent

* One config file to rule them all

## TODO

* Express path of maze in terms of meters (do not normalize yet during creation! Perhaps introduce a max_path parameter as well?)
* Update fitness functions to be more conform to that of James
* Add obstacles to the maze (perhaps in an empty room adding a cube?)
* Timing mechanics in `main.py`, possible to visualize this as well?
* Automatic trigger to reset game and algorithm (e.g. also trigger training of algorithm) via control in main
* NEAT and variants implementation:
    * Start with framework to put algorithms in that interacts with the environment (give `game` as an argument, such that calls happen efficient)
    * Implement NEAT with novelty search and debug
    * Implement Hyper-NEAT
    * Implement adaptive Hyper-NEAT
