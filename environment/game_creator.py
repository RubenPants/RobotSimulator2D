"""
game_creator.py

Create a maze. A maze consists out of three different types of tiles:
 * -1 = Wall
 * 0 = Empty tile
 * >0 = Visited tile (used for path-finding)
Walls can only be placed on tiles with odd positions. If a wall goes from (1,1) to (3,1), then the wall spans over all
the tiles on [(1,1), (2,1), (3,1)].

The creation of a stage will happen in three different stages:
 1. Large rooms will be added
 2. Walls separating open spots will be added such that one room covers at most 1/4th of the tiles
 3. Doors will be made in the walls such that the starting agent can reach every part of the field

After successfully creating the maze-matrix, it is converted to a Pymunk game and then saved in the 'games_db' folder.
"""
import argparse
import os
import random
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from config import GameConfig
from environment.entities.game import Game
from environment.entities.robots import FootBot
from utils.line2d import Line2d
from utils.vec2d import Vec2d

# Constants
MIN_ROOM_WIDTH = 5
ROOM_ATTEMPTS = 5
FILLED_ROOM_RATIO = 0.2


class Maze:
    def __init__(self, cfg: GameConfig, visualize: bool = False):
        """
        Auto-generate a maze, which is a model for the final games.
        
        :param cfg: Config file
        :param visualize: Visualize intermediate steps of creation
        """
        # Load in the config file
        self.cfg = cfg
        
        # Set the maze's main parameters
        self.x_width = 2 * self.cfg.x_axis + 1
        self.y_width = 2 * self.cfg.y_axis + 1
        self.tiles_amount = self.x_width * self.y_width
        self.maze = np.zeros((self.y_width, self.x_width))
        
        # Add initial walls
        for x in range(self.x_width):
            self.maze[0, x] = -1
            self.maze[self.y_width - 1, x] = -1
        for y in range(self.y_width):
            self.maze[y, 0] = -1
            self.maze[y, self.x_width - 1] = -1
        self.wall_tiles = self.get_wall_tiles()
        
        # Generate rooms
        self.generate_rooms()
        if visualize: self.visualize()
        
        # Add division-walls
        self.generate_division_walls()
        if visualize: self.visualize()
        
        # Add doors in walls such that every room is connected
        self.connect_rooms()
        if visualize: self.visualize()
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def connect_rooms(self):
        """
        Connect the rooms by making 2 meter doors in them. This is done by filling the start-location of the agent
        (which is the bottom right corner), and then checked if there are any values in the matrix left that have a zero
        value. If those exist, then drill a hole in a wall that connects both a one and a zero value (this hole consists
        out of four consecutive wall-segments).
        """
        pos = self.get_empty_tile_room()
        while pos:
            self.reset_empty_tiles()
            self.fill_room(pos)
            self.create_door()
            pos = self.get_empty_tile_room()
    
    def generate_rooms(self):
        """
        Try to add several rooms for a few times. This method is very stochastic and has a high tendency to fail when
        several rooms are already added. This is however no problem, thus ignored.
        """
        for _ in range(ROOM_ATTEMPTS):
            try:
                self.add_room()
            except IndexError:
                pass  # ignore
    
    def generate_division_walls(self):
        """
        Go over each room and check how many tiles are in it. If there are more than 1/4th of the total tiles inside of
        this room, the room is divided.
        """
        # Go over all potential wall-tiles that are empty
        add_hor = True if random.random() > 0.5 else False
        for x in range(2, self.x_width - 1, 2):
            for y in range(2, self.y_width - 1, 2):
                # Identify if empty
                if self.maze[y, x] >= 0:
                    # Check ratio of empty tiles
                    filled_tiles = self.fill_room((x, y), reset=True)
                    if len(filled_tiles) / self.tiles_amount > FILLED_ROOM_RATIO:
                        # Add a wall on a potential wall-tile in the empty room
                        self.add_wall([(x, y) for (x, y) in filled_tiles if (x % 2 == 0 and y % 2 == 0)], hor=add_hor)
                        add_hor = not add_hor
    
    def get_wall_coordinates(self):
        """
        :return: Wall coordinates of final maze (excluding boundaries) in original axis-format.
        """
        wall_list = []
        # Horizontal segments
        for x in range(1, self.x_width - 1, 2):
            for y in range(2, self.y_width, 2):
                if self.maze[y, x] == -1:
                    wall_list.append(Line2d(Vec2d((x - 1) // 2, y // 2), Vec2d((x + 1) // 2, y // 2)))
        # Vertical segments
        for x in range(2, self.x_width, 2):
            for y in range(1, self.y_width - 1, 2):
                if self.maze[y, x] == -1:
                    wall_list.append(Line2d(Vec2d(x // 2, (y - 1) // 2), Vec2d(x // 2, (y + 1) // 2)))
        
        combine_walls(wall_list)
        return wall_list
    
    def get_path_coordinates(self, visualize: bool = False):
        """ Define all free-positions together with their distance to the target """
        
        # Expand the maze such that every 10cm gets a tile
        y_tiles = self.cfg.y_axis * 11 + 1
        x_tiles = self.cfg.x_axis * 11 + 1
        maze_expanded = np.zeros((y_tiles, x_tiles))
        
        # Copy tiles from current maze to the extended maze
        for row in range(y_tiles):
            for col in range(x_tiles):
                #  Cross-section
                if row % 11 == 0 and col % 11 == 0:
                    maze_expanded[row, col] = -1
                
                # Horizontal walls
                elif row % 11 == 0 and self.maze[(row // 11) * 2, (col // 11) * 2 + 1] < 0:
                    maze_expanded[row, col] = -1
                
                # Vertical walls
                elif col % 11 == 0 and self.maze[(row // 11) * 2 + 1, (col // 11) * 2] < 0:
                    maze_expanded[row, col] = -1
        
        def update(pos_1, pos_2, dist=0.1):
            """
            Update a single position based on its neighbour position.
            
            :param pos_1: Current position
            :param pos_2: Neighbour position that needs to update
            :param dist: Distance (real-life) between the two positions
            """
            if self.in_maze([pos_2[0], pos_2[1]], maze=maze_expanded) and \
                    0 <= maze_expanded[pos_2[0], pos_2[1]] < maze_expanded[pos_1[0], pos_1[1]] - dist:
                maze_expanded[pos_2[0], pos_2[1]] = maze_expanded[pos_1[0], pos_1[1]] - dist
                return True
            return False
        
        def update_neighbours(p):
            updated = set()
            for i in [-1, 1]:
                if update(p, [p[0] + i, p[1]]): updated.add((p[0] + i, p[1]))  # Horizontal
                if update(p, [p[0], p[1] + i]): updated.add((p[0], p[1] + i))  # Vertical
                if update(p, [p[0] + i, p[1] + i], dist=sqrt(0.02)): updated.add((p[0] + i, p[1] + i))  # Diagonal
                if update(p, [p[0] + i, p[1] - i], dist=sqrt(0.02)): updated.add((p[0] + i, p[1] - i))  # V-shaped
            return updated
        
        # Constant
        VALUE_START = 100
        
        # Find distance to target
        if visualize: self.visualize_extend(maze=maze_expanded)
        maze_expanded[y_tiles - 6, 5] = VALUE_START
        updated_pos = {(y_tiles - 6, 5)}
        
        # Keep updating all the set_positions' neighbours while change occurs
        while updated_pos:
            new_pos = set()
            for pos in updated_pos:
                new_updated_pos = update_neighbours(pos)
                if new_updated_pos:
                    new_pos.update(new_updated_pos)
            updated_pos = new_pos
        if visualize: self.visualize_extend(maze=maze_expanded)
        
        # Invert values such that distance to target @target equals 0, and distance at start equals |X-VALUE_START| / 2
        # Note the /2 since 1m in-game is 2 squared here
        for row in range(y_tiles):
            for col in range(x_tiles):
                if maze_expanded[row, col] > 0:
                    maze_expanded[row, col] = abs(maze_expanded[row, col] - VALUE_START)
        
        # Put values in list
        values = []
        for row in range(y_tiles - 1):
            for col in range(x_tiles - 1):
                if row % 11 == 0 or col % 11 == 0: continue
                # Note the maze_expanded[col, row] which is a bug rippled through whole project (everywhere switched!)
                values.append((((row - row // 11) / 10, (col - col // 11) / 10), maze_expanded[col, row]))
        if visualize:
            self.visualize_extend(maze=maze_expanded)
            print("Coordinate (fitness) values:")
            for v in values: print(v)
        return values
    
    # -------------------------------------------> MAIN SECONDARY METHODS <------------------------------------------- #
    
    def add_room(self):
        """
        Create a room. A room starts always adjacent to a wall and is of maximum size the size of this wall.
        """
        # Random starting position
        p = self.get_random_wall_position()
        
        # Determine the lengthiest side  (Note: Changed down and left, so self.maze coordinates are correct)
        start_p = Vec2d(p[0], p[1])  # Start is situated at the bottom left
        while self.in_maze(start_p + Vec2d(0, -2)) and self.maze[start_p[1] - 2, start_p[0]] < 0:
            start_p += Vec2d(0, -2)
        if start_p == Vec2d(p[0], p[1]):
            while self.in_maze(start_p + Vec2d(-2, 0)) and self.maze[start_p[1], start_p[0] - 2] < 0:
                start_p += Vec2d(-2, 0)
        end_p = p  # End is situated at the top right
        while self.in_maze(end_p + Vec2d(0, 2)) and self.maze[end_p[1] + 2, end_p[0]] < 0:
            end_p += Vec2d(0, 2)
        if end_p == Vec2d(p[0], p[1]):
            while self.in_maze(end_p + Vec2d(2, 0)) and self.maze[end_p[1], end_p[0] + 2] < 0:
                end_p += Vec2d(2, 0)
        
        # Determine the length between the two positions
        diff = end_p - start_p
        max_length = int(diff.get_length())
        
        # Early-stop if base wall is not wide enough
        if max_length < MIN_ROOM_WIDTH:
            raise IndexError
        
        # Define a new starting position
        direction = Vec2d(int(diff.normalized()[0]), int(diff.normalized()[1]))
        offset_start = random.choice([x * 2 for x in range(max_length // 2 - 2) if x != 1])
        start_new = start_p + direction * offset_start
        
        # Define a new end position
        max_length_end = int(max_length - (start_p - start_new).get_length())
        offset_end = random.choice([x * 2 for x in range(max_length_end // 2 - 2) if x != 1])
        end_new = end_p - direction * offset_end
        
        # Check if wall is wide enough
        room_width = int((end_new - start_new).get_length())
        if room_width < MIN_ROOM_WIDTH:
            raise IndexError
        
        # Grow to both sides
        direction_ort1 = Vec2d(direction[1], direction[0])
        depth1 = 0
        no_wall = True
        while no_wall:
            for x in range(room_width):
                pos = start_new + direction * x + direction_ort1 * (depth1 + 1)
                if not self.in_maze(pos) or self.maze[pos[1], pos[0]] < 0:
                    no_wall = False
                    break
            if no_wall:
                depth1 += 1
        direction_ort2 = -direction_ort1
        depth2 = 0
        no_wall = True
        while no_wall:
            for x in range(room_width):
                pos = start_new + direction * x + direction_ort2 * (depth2 + 1)
                if not self.in_maze(pos) or self.maze[pos[1], pos[0]] < 0:
                    no_wall = False
                    break
            if no_wall:
                depth2 += 1
        
        # Set the depth of the room
        max_depth = depth1 + 1 if depth1 > depth2 else depth2 + 1
        direction_ort = direction_ort1 if depth1 > depth2 else direction_ort2
        room_depth = random.choice([x * 2 for x in range(2, max_depth // 2 + 1) if x != (max_depth // 2 - 1)])
        
        # Create walls
        for x in range(room_depth + 1):
            self.maze[start_new[1] + direction_ort[1] * x, start_new[0] + direction_ort[0] * x] = -1
            self.maze[end_new[1] + direction_ort[1] * x, end_new[0] + direction_ort[0] * x] = -1
        for x in range(0, room_width + 1):
            self.maze[start_new[1] + direction[1] * x + direction_ort[1] * room_depth,
                      start_new[0] + direction[0] * x + direction_ort[0] * room_depth] = -1
            self.maze[start_new[1] + direction[1] * x + direction_ort[1] * room_depth,
                      start_new[0] + direction[0] * x + direction_ort[0] * room_depth] = -1
    
    def add_wall(self, lst: list, hor: bool = True):
        """
        Add a random straight wall starting on one of the given positions, such that it connects two walls.
        
        :param lst: List of positions which may be the starting position of the wall
        :param hor: True: add a horizontal wall | False: add a vertical wall
        """
        # Find the best position
        best, pos = 0, []
        for p in lst:
            v = self.get_nr_empty_neighbours(p)
            if v > best:
                pos = [p]
                best = v
            elif v == best:
                pos.append(p)
        
        # Choose random position from best positions
        pos = random.choice(pos)
        
        # Add the wall
        self.maze[pos[1], pos[0]] = -1
        if hor:  # Horizontal
            x = 1
            while self.maze[pos[1], pos[0] + x] >= 0:
                self.maze[pos[1], pos[0] + x] = -1
                x += 1
            x = 1
            while self.maze[pos[1], pos[0] - x] >= 0:
                self.maze[pos[1], pos[0] - x] = -1
                x += 1
        else:  # Vertical
            x = 1
            while self.maze[pos[1] + x, pos[0]] >= 0:
                self.maze[pos[1] + x, pos[0]] = -1
                x += 1
            x = 1
            while self.maze[pos[1] - x, pos[0]] >= 0:
                self.maze[pos[1] - x, pos[0]] = -1
                x += 1
    
    def create_door(self):
        """
        Create door-openings of at least three wide.
    
        :return: One of the door-positions
        """
        
        # Get all the potential doors
        potential_doors = self.get_potential_doors()
        
        # Filter out doors from the potential doors
        door = random.choice(potential_doors)
        self.maze[door[1], door[0]] = 0
        self.fill_room(door)
        
        # Add extra doors based on the total potential_doors size
        if random.random() > (10.0 / len(potential_doors)):
            door = random.choice(potential_doors)
            self.maze[door[1], door[0]] = 0
            self.fill_room(door)
        if random.random() > (20.0 / len(potential_doors)):
            door = random.choice(potential_doors)
            self.maze[door[1], door[0]] = 0
            self.fill_room(door)
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def fill_room(self, pos, reset: bool = False):
        """
        Fill a room (integer > 0) and count the number of tiles in it.
        
        :param pos: Starting position on which the room will be filled
        :param reset: Set the filled tiles back to zero after the room has been filled
        :return: Number of tiles in room
        """
        
        def get_empty_neighbours(init_pos):
            neighbours = []  # Only if value > 0
            if init_pos[0] + 1 < self.x_width and self.maze[init_pos[1], init_pos[0] + 1] == 0:
                neighbours.append((init_pos[0] + 1, init_pos[1]))
            if init_pos[0] - 1 >= 0 and self.maze[init_pos[1], init_pos[0] - 1] == 0:
                neighbours.append((init_pos[0] - 1, init_pos[1]))
            if init_pos[1] + 1 < self.y_width and self.maze[init_pos[1] + 1, init_pos[0]] == 0:
                neighbours.append((init_pos[0], init_pos[1] + 1))
            if init_pos[1] - 1 >= 0 and self.maze[init_pos[1] - 1, init_pos[0]] == 0:
                neighbours.append((init_pos[0], init_pos[1] - 1))
            return neighbours
        
        # Initialize
        self.maze[pos[1], pos[0]] = 1
        filled_positions = [pos]
        
        # Fill the room until no tile changes
        new_value = True
        while new_value:
            new_value = False
            for p in filled_positions:
                empty_n = get_empty_neighbours(p)
                for n in empty_n:
                    self.maze[n[1], n[0]] = 1
                    filled_positions.append(n)
                    new_value = True
        
        # Put filled tiles back to zero
        if reset:
            self.reset_empty_tiles()
        
        return filled_positions
    
    def get_empty_tile_room(self):
        """
        :return: Random empty tile
        """
        empty_tiles = []  # Only the odd ones are important (real final positions)
        for x in range(1, self.x_width - 1, 2):
            for y in range(1, self.y_width - 1, 2):
                if self.maze[y, x] == 0:
                    empty_tiles.append((x, y))
        return random.choice(empty_tiles) if empty_tiles else None
    
    def get_potential_doors(self):
        """
        A potential door is a wall-tile with at one side a filled tile, and on the other side an empty tile.
        
        :return: Position tiles that are currently walls
        """
        occupied_tiles = []
        for x in range(1, self.x_width - 1):
            for y in range(1, self.y_width - 1):
                if self.maze[y, x] == -1:
                    # Vertical
                    if (self.maze[y - 1, x] + self.maze[y + 1, x]) == 1:
                        occupied_tiles.append((x, y))
                    # Horizontal
                    if (self.maze[y, x - 1] + self.maze[y, x + 1]) == 1:
                        occupied_tiles.append((x, y))
        
        # Filter out all the wall-tiles
        return [Vec2d(x, y) for (x, y) in occupied_tiles if (x % 2 == 1 or y % 2 == 1)]
    
    def get_wall_tiles(self):
        wall_positions = []
        for x in range(0, self.x_width + 1, 2):
            for y in range(0, self.y_width + 1, 2):
                if self.maze[y, x] < 0:
                    wall_positions.append((x, y))
        return wall_positions
    
    def in_maze(self, pos, maze=None):
        if maze is not None: return 0 <= pos[0] < maze.shape[0] and 0 <= pos[1] < maze.shape[1]
        return 0 <= pos[0] < self.x_width and 0 <= pos[1] < self.y_width
    
    def reset_empty_tiles(self):
        """
        Put all the empty tiles back to zero.
        """
        for x in range(self.x_width):
            for y in range(self.y_width):
                if self.maze[y, x] > 0:
                    self.maze[y, x] = 0
    
    def visualize(self, clip: bool = True):
        """Visualize the main maze representation."""
        c = self.maze.copy()
        if clip:
            c = np.clip(c, a_min=-1, a_max=0)
        for x in range(self.x_width):
            for y in range(self.y_width):
                if (x % 2 == 1) and (y % 2 == 1) and clip:
                    c[y, x] = 0.1
                elif (x % 2 == 0) and (y % 2 == 0):
                    if c[y, x - 1] >= 0 and c[y - 1, x] >= 0 and c[y, x + 1] >= 0 and c[y + 1, x] >= 0:
                        c[y, x] = 0
        plt.figure(figsize=(8, 8))
        plt.imshow(c, origin='lower')
        plt.colorbar()
        plt.show()
        plt.close()
    
    def visualize_extend(self, maze, clip: bool = True):
        """Visualize maze where 1 meter is represented by 10 tiles. No matrix-processing must be done."""
        c = maze.copy()
        plt.figure(figsize=(8, 8))
        plt.imshow(c, origin='lower')
        plt.xticks([i * 11 for i in range(self.cfg.y_axis + 1)])
        plt.yticks([i * 11 for i in range(self.cfg.x_axis + 1)])
        plt.colorbar()
        plt.show()
        plt.close()
    
    def get_random_wall_position(self):
        self.wall_tiles = self.get_wall_tiles()
        return random.choice(self.wall_tiles)
    
    def get_nr_empty_neighbours(self, pos):
        """
        :return: Number of empty tiles in a surrounding 5x5 square
        """
        return len([(x, y) for x in range(pos[0] - 2, pos[0] + 3) for y in range(pos[1] - 2, pos[1] + 3) if
                    (self.in_maze((x, y)) and self.maze[y, x] >= 0)])


def combine_walls(wall_list: list):
    """
    Combine every two wall-segments (Line2d) together that can be represented by only one line segment. This will
    increase the performance later on, since the intersection methods must loop over less wall-segments.

    :param wall_list: List<Line2d>
    :return: List<Line2d>
    """
    i = 0
    while i < len(wall_list) - 1:
        concat = False
        # Check if wall at i can be concatenated with wall after i
        for w in wall_list[i + 1:]:
            # Check if in same horizontal line
            if wall_list[i].x.y == wall_list[i].y.y == w.x.y == w.y.y:
                # Check if at least one point collides
                if wall_list[i].x.x == w.x.x:
                    wall_list[i] = Line2d(Vec2d(wall_list[i].y.x, w.x.y), Vec2d(w.y.x, w.x.y))
                    concat = True
                elif wall_list[i].y.x == w.x.x:
                    wall_list[i] = Line2d(Vec2d(wall_list[i].x.x, w.x.y), Vec2d(w.y.x, w.x.y))
                    concat = True
                elif wall_list[i].y.x == w.y.x:
                    wall_list[i] = Line2d(Vec2d(wall_list[i].x.x, w.x.y), Vec2d(w.x.x, w.x.y))
                    concat = True
                elif wall_list[i].x.x == w.y.x:
                    wall_list[i] = Line2d(Vec2d(wall_list[i].y.x, w.x.y), Vec2d(w.x.x, w.x.y))
                    concat = True
            
            # Check if in same vertical line
            elif wall_list[i].x.x == wall_list[i].y.x == w.x.x == w.y.x:
                # Check if at least one point collides
                if wall_list[i].x.y == w.x.y:
                    wall_list[i] = Line2d(Vec2d(w.x.x, wall_list[i].y.y), Vec2d(w.x.x, w.y.y))
                    concat = True
                elif wall_list[i].y.y == w.x.y:
                    wall_list[i] = Line2d(Vec2d(w.x.x, wall_list[i].x.y), Vec2d(w.x.x, w.y.y))
                    concat = True
                elif wall_list[i].y.y == w.y.y:
                    wall_list[i] = Line2d(Vec2d(w.x.x, wall_list[i].x.y), Vec2d(w.x.x, w.x.y))
                    concat = True
                elif wall_list[i].x.y == w.y.y:
                    wall_list[i] = Line2d(Vec2d(w.x.x, wall_list[i].y.y), Vec2d(w.x.x, w.x.y))
                    concat = True
            
            # If w concatenated with i'th wall in wall_list, then remove w from list and break for-loop
            if concat:
                wall_list.remove(w)
                break
        
        # Current wall cannot be extended, go to next wall
        if not concat:
            i += 1


def create_custom_game(cfg: GameConfig, overwrite=False):
    """ Dummy to create a custom-defined game. """
    # Initial parameters
    game_id = 0
    
    # Create empty Game instance
    game = Game(config=cfg,
                game_id=game_id,
                overwrite=overwrite)
    
    # Set game path
    path = dict()
    for x in range(cfg.x_axis):
        for y in range(cfg.y_axis):
            path[(x + 0.5, y + 0.5)] = Line2d(Vec2d(0.5, cfg.y_axis - 0.5), Vec2d(x + 0.5, y + 0.5)).get_length()
    game.path = path
    
    # Put the target on a fixed position
    game.target = Vec2d(0.5, cfg.y_axis - 0.5)
    
    # Create random player
    game.player = FootBot(game=game,
                          init_pos=Vec2d(cfg.x_axis - 0.5, 0.5),
                          init_orient=np.pi / 2)
    
    # Check if implemented correctly
    game.close()
    game.reset()
    game.get_blueprint()
    game.get_observation()
    game.step(0, 0)
    
    # Save the final game
    game.save()


def create_game(cfg: GameConfig, game_id=0, path_list=None, wall_list=None, overwrite=False):
    """
    Create a game based on a list of walls.
    
    :param cfg: The game config
    :param game_id: ID of the game (Integer)
    :param path_list: List of paths together with value [0..1] indicating how close the tile is to the target
    :param wall_list: List of Line2d objects containing the begin and end coordinate of a wall, excluding boundary walls
    :param overwrite: Overwrite pre-existing games
    """
    # Create empty Game instance
    game = Game(config=cfg,
                game_id=game_id,
                overwrite=overwrite,
                silent=True)
    
    # Add additional walls to the game
    game.walls.update(set(wall_list))
    
    # App path to the game
    game.path = {p[0]: p[1] for p in path_list}
    
    # Put the target on a fixed position
    game.target = Vec2d(0.5, cfg.y_axis - 0.5)
    
    # Create random player
    game.player = FootBot(game=game,
                          init_pos=Vec2d(cfg.x_axis - 0.5, 0.5),
                          init_orient=np.pi / 2)
    
    # Save the final game
    game.save()


if __name__ == '__main__':
    """
    Create game, option to choose from custom or auto-generated.
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--custom', type=bool, default=False)
    parser.add_argument('--overwrite', type=bool, default=True)
    parser.add_argument('--nr_games', type=int, default=None)
    parser.add_argument('--visualize', type=bool, default=False)
    args = parser.parse_args()
    
    # Point back to root
    os.chdir('..')
    
    # Load in the config file
    config = GameConfig()
    
    # Setup the params
    nr_games = args.nr_games
    if not nr_games: nr_games = config.max_eval_game_id
    
    if args.custom:
        create_custom_game(cfg=config, overwrite=args.overwrite)
    else:
        for g_id in [-1]:#tqdm(range(1, nr_games + 1), desc="Generating Mazes"):
            maze = None
            while not maze:
                try:
                    maze = Maze(cfg=config, visualize=args.visualize)
                except IndexError:
                    maze = None  # Reset and try again
            create_game(cfg=config,
                        game_id=g_id,
                        path_list=maze.get_path_coordinates(visualize=args.visualize),
                        wall_list=maze.get_wall_coordinates(),
                        overwrite=args.overwrite)
            
            # Quality check the created game
            try:
                game = Game(
                        game_id=g_id,
                        save_path="environment/games_db/",
                        overwrite=False,
                        silent=True,
                )
                game.close()
                game.reset()
                game.get_blueprint()
                game.get_observation()
                game.step(0, 0)
            except Exception:
                print(f"Faulty created game: {g_id}, please manually redo this one")
                os.remove(f"environment/games_db/game_{g_id:05d}")
