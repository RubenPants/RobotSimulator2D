# RobotSimulator2D [deprecated]
Custom low-level robot simulator, which evaluates evolved Recurrent Neural Networks (RNN) in a room-like maze 
environment. This repository was the initial repo of my Master's thesis. However, due to a change in focus, I created a 
new repository called *EvolvableRNN* (https://github.com/RubenPants/EvolvableRNN), which builds further on this repo. In
other words, this repository is unfinished. If you are interested in how the networks are RNNs are evolved via NEAT, 
then please refer to the EvolvableRNN repository. On the other hand, the environment creation, which is discarded in 
EvolvableRNN, is still interesting, hence I left this project on my GitHub.


## Project Overview
The `main.py` file (root directory) contains all the supported functionality to evolve and evaluate populations of RNNs.
Since this project mostly resembles the *EvolvableRNN*, please refer to https://github.com/RubenPants/EvolvableRNN for
a detailed explanation on each of the project's functionality.


## Differences with EvolvableRNN
This section addresses the main differences between this (intermediate) project and the finished EvolvableRNN project.

### Networks
In this project, it is only possible to evolve RNNs which support the GRU cell, whereas EvolvableRNN supports several 
other recurrent units, as well as an improved implementation of these networks. The figure below shows a network that
has no hidden nodes.
<p align="center">
  <img src="https://github.com/RubenPants/RobotSimulator2D/blob/master/population/storage/NEAT-GRU/path_2/images/architectures/genome_9593.png"/>
</p>

### Environment
The EvolvableRNN environment uses a sparse environment that lacks obstacles. In other words, its environments only 
consist of a single agent (robot) and a single target to which the robot should navigate to. This project, however, has
as environments mazes (in the form of an indoor single level ground plan), which the robot should navigate in order to
reach its target. This implies that the environment of this project is far more sophisticated than that of EvolvableRNN.

The figure below shows an example of such a maze environment.
<p align="center">
  <img src="https://github.com/RubenPants/RobotSimulator2D/blob/master/environment/visualizations/blueprint_game00002.png"/>
</p>

### Sensory Inputs
Where EvolvableRNN only considers the distance between the robot and its target, the sensory implementation in this
repository has a wider variety of sensors to choose from. Not only the distance towards the target is supported, but
also proximity sensors to sense obstacles, as well as a bearing sensor that indicates the relative direction towards the
target. The combination of such sensors make that it is possible for a population to navigate the environment 
successfully, as the figure below shows.
<p align="center">
  <img src="https://github.com/RubenPants/RobotSimulator2D/blob/master/population/storage/NEAT-GRU/path_2/images/game00002/trace_gen00100.png"/>
</p>

### Heatmap Visualisations
To help improve the fitness calculation of the networks, a heatmap is created for each environment representing the A*
distance (which includes the fact that the robot cannot travel through a wall) to the target. An example is shown in the
image below.
<p align="center">
  <img src="https://github.com/RubenPants/RobotSimulator2D/blob/master/environment/visualizations/heatmap_game00002.png"/>
</p>