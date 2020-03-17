"""
game_creator.py

Create a maze. A maze consists out of three different types of tiles:
 * -1 = Wall
 * 0 = Empty tile
 * >0 = Visited tile (used for path-finding)
Walls can only be placed on tiles with odd positions. If a wall goes from (1,1) to (3,1), then the wall spans over all
 the tiles on [(1,1), (2,1), (3,1)].

The creation of a maze will happen in three different stages:
 1. Between 2 and 4 corridors will be added, these have a width of 1 meter and go alternately horizontal and vertical
     They continue running until they meet a wall at both sides.
 2. Walls separating open spots will be added such that one room covers at most 1/4th of the tiles
 3. Doors will be made in the walls such that the starting agent can reach every part of the field

After successfully creating the maze-matrix, it is converted to a Pymunk game and then saved in the 'games_db' folder.

:note: The following protocol holds, lists save coordinates in [x,y], but to query the maze one needs to enter [y,x],
 this is due to the x=row, y=col mismatch.
"""
import argparse
import os
from math import sqrt
from random import choice, randint, random, shuffle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import GameConfig
from environment.entities.game import Game
from environment.entities.robots import MarXBot
from utils.line2d import Line2d
from utils.vec2d import Vec2d

# Constants
FILLED_ROOM_RATIO = 0.3
DOOR_ROOM_RATIO = 0.2


class MazeMalfunctionException(Exception):
    """Custom exception indicating a malfunction during maze creation."""
    pass


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
        corridor_tiles = self.generate_corridors()
        if visualize: self.visualize()
        
        # Add division-walls
        self.generate_division_walls(corridor=corridor_tiles)
        if visualize: self.visualize()
        
        # Add doors in walls such that every room is connected
        self.connect_rooms(corridor=corridor_tiles)
        if visualize: self.visualize()
        
        # Set the position for the target
        r = random()
        if r < 1 / 5:
            self.target = Vec2d(cfg.x_axis - 0.5, cfg.y_axis - 0.5)  # Top right
        elif r < 2 / 5:
            self.target = Vec2d(0.5, 0.5)  # Bottom left
        else:
            self.target = Vec2d(0.5, cfg.y_axis - 0.5)  # Top left
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def generate_corridors(self):
        """
        Generate three walls that go alternately horizontal and vertical. A wall has a width of 1 meter and go from one
        wall until another. The tiles that represent the corridor are returned.
        """
        # Create corridors
        is_hor = random() < 0.5
        
        # Add main corridor
        tiles = set()
        added = False
        while not added:
            added = self.add_corridor(horizontal=is_hor)
        tiles.update(added)
        
        # Add two side corridors
        added = 0
        while added < 2:
            added_tiles = self.add_corridor(horizontal=not is_hor)
            if added_tiles:
                added += 1
                tiles.update(added_tiles)
        
        # Make intersections of corridors door-free
        self.remove_corridor_doors()
        
        # Return each of the tiles that represent the corridor
        return tiles
    
    def generate_division_walls(self, corridor):
        """
        Go over each room and check how many tiles are in it. If there are more than 1/4th of the total tiles inside of
        this room, the room is divided.
        """
        # Get all non-corridor tiles
        all_tiles = {(x, y) for x in range(1, self.x_width - 1, 2) for y in range(1, self.y_width - 1, 2)}
        for c in corridor: all_tiles.remove(c)
        total_size = len(all_tiles)
        
        # Check for each of the rooms if they are too large, if so then separate them
        while len(all_tiles) > 0:
            pos = all_tiles.pop()  # pos (y-axis, x-axis)
            filled_tiles = self.fill_room(pos, reset=True)
            if (len(filled_tiles) / total_size) > FILLED_ROOM_RATIO:
                x_min = min([p[0] for p in filled_tiles])
                x_max = max([p[0] for p in filled_tiles])
                y_min = min([p[1] for p in filled_tiles])
                y_max = max([p[1] for p in filled_tiles])
                
                # Define the center, which is a tuple of two even numbers
                width = x_max - x_min
                height = y_max - y_min
                x_c = int(width / 2 + x_min)
                if x_c % 2 == 1: x_c += choice([-1, 1])
                if width >= 10: x_c += choice([-2, 0, 0, 2])  # Add noise
                y_c = int(height / 2 + y_min)
                if y_c % 2 == 1: y_c += choice([-1, 1])
                if height >= 10: y_c += choice([-2, 0, 0, 2])  # Add noise
                center = (x_c, y_c)
                
                # Add wall based on direction of room
                if width == height:  # Squared
                    self.add_wall(pos=center, hor=random() >= 0.5)
                elif width > height:  # Wide
                    self.add_wall(pos=center, hor=False)
                else:  # Tall
                    self.add_wall(pos=center, hor=True)
            else:
                for ft in filled_tiles:
                    if ft in all_tiles: all_tiles.remove(ft)
    
    def connect_rooms(self, corridor):
        """
        Connect the rooms by making 2 meter doors in them. This is done by filling the start-location of the agent
        (which is the bottom right corner), and then checked if there are any values in the matrix left that have a zero
        value. If those exist, then drill a hole in a wall that connects both a one and a zero value (this hole consists
        out of four consecutive wall-segments).
        """
        # Get all non-corridor tiles
        all_tiles = {(x, y) for x in range(1, self.x_width - 1, 2) for y in range(1, self.y_width - 1, 2)}
        total_size = len(all_tiles)
        for c in corridor: all_tiles.remove(c)
        corridor_tile = next(iter(corridor))
        
        # Check if room is connected to the corridor, if not then add door
        while len(all_tiles) > 0:
            pos = all_tiles.pop()
            filled_tiles = self.fill_room(pos, reset=True)
            room_tiles = filled_tiles.copy()  # Room tiles does not change (needed to create door)
            filled_size = len(filled_tiles)
            
            # Room already connected to the corridor
            if corridor_tile in filled_tiles:
                all_tiles = all_tiles - filled_tiles
            
            # Room not connected to the corridor, add door
            else:
                # Get corner coordinates
                x_min = min([p[0] for p in filled_tiles])
                x_max = max([p[0] for p in filled_tiles])
                y_min = min([p[1] for p in filled_tiles])
                y_max = max([p[1] for p in filled_tiles])
                
                # Remove corners from filled_tiles
                try:
                    filled_tiles.remove((x_min, y_min))
                    filled_tiles.remove((x_min, y_max))
                    filled_tiles.remove((x_max, y_min))
                    filled_tiles.remove((x_max, y_max))
                except KeyError:
                    # TODO: This is (perhaps?) due to door-creation of two neighbouring rooms of different size
                    raise MazeMalfunctionException("Error in wall-creation, created non-square rooms")
                
                # Remove tiles that are not suited for a door
                to_remove = set()
                for t in filled_tiles:
                    # Remove inner tiles from filled_tiles
                    if (t[0] not in [x_min, x_max]) and (t[1] not in [y_min, y_max]):
                        to_remove.add(t)
                    
                    # Remove tiles that are next to an edge-wall
                    elif t[0] in [1, self.x_width - 2]:
                        to_remove.add(t)
                    elif t[1] in [1, self.y_width - 2]:
                        to_remove.add(t)
                filled_tiles = filled_tiles - to_remove
                
                # Create a door between room and corridor, large rooms have two doors
                remaining_tiles = list(filled_tiles)
                shuffle(remaining_tiles)
                for _ in range(1 if (filled_size / total_size < DOOR_ROOM_RATIO) else 2):
                    if len(remaining_tiles) == 0: break
                    chosen_pos = remaining_tiles[0]
                    self.create_door(pos=chosen_pos, room_tiles=room_tiles)
                    
                    # Remove all the remaining_tiles of the chosen_pos' row/col
                    to_remove = []
                    for p in remaining_tiles:
                        if (p[0] == chosen_pos[0]) or (p[1] == chosen_pos[1]): to_remove.append(p)
                    for p in to_remove:
                        remaining_tiles.remove(p)
                
                # Room is now connected, remove from all_tiles
                all_tiles = all_tiles - room_tiles
    
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
    
    def get_path_coordinates(self, target_pos, visualize: bool = False):
        """Define all free-positions together with their distance to the target."""
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
            if self.in_maze([pos_2[1], pos_2[0]], maze=maze_expanded) and \
                    0 <= maze_expanded[pos_2[1], pos_2[0]] < maze_expanded[pos_1[1], pos_1[0]] - dist:
                maze_expanded[pos_2[1], pos_2[0]] = maze_expanded[pos_1[1], pos_1[0]] - dist
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
        if target_pos == Vec2d(maze.cfg.x_axis - 0.5, maze.cfg.y_axis - 0.5):
            target = (x_tiles - 6, y_tiles - 6)
        elif target_pos == Vec2d(0.5, 0.5):
            target = (5, 5)
        elif target_pos == Vec2d(0.5, maze.cfg.y_axis - 0.5):
            target = (5, y_tiles - 6)
        else:
            raise Exception("Invalid target_pos input")
        
        maze_expanded[target[1], target[0]] = VALUE_START
        updated_pos = {target}
        
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
        for row in range(1, y_tiles - 1):
            for col in range(1, x_tiles - 1):
                if maze_expanded[row, col] > 0:
                    maze_expanded[row, col] = abs(maze_expanded[row, col] - VALUE_START)
        
        # Put values in list
        values = []
        for row in range(1, y_tiles - 1):
            for col in range(1, x_tiles - 1):
                if row % 11 == 0 or col % 11 == 0: continue
                # Note the maze_expanded[col, row] which is a bug rippled through whole project (everywhere switched!)
                values.append((((row - row // 11) / 10, (col - col // 11) / 10), maze_expanded[col, row]))
        if visualize:
            self.visualize_extend(maze=maze_expanded)
            print("Coordinate (fitness) values:")
            for v in values: print(v)
        return values
    
    # -------------------------------------------> MAIN SECONDARY METHODS <------------------------------------------- #
    
    def add_corridor(self, horizontal):
        """
        Add a random corridor to the room. A corridor is placed on a random location and is 1 meter wide, spanning from
        one side straight to the other.
        """
        # Position to start on, this position will be in the corridor's left or bottom wall
        pos_x = randint(0, self.cfg.x_axis - 1) * 2
        pos_y = randint(0, self.cfg.y_axis - 1) * 2
        self.maze[pos_y, pos_x] = -1
        
        # Add the wall's remainders
        tiles = set()
        if horizontal:
            # Check on double and triple corridors
            if (pos_y != self.y_width - 3) and ((self.maze[pos_y + 2, pos_x] < 0) or
                                                ((pos_y - 2 >= 0) and (self.maze[pos_y - 2, pos_x] < 0)) or
                                                ((pos_y + 4 <= self.y_width) and (self.maze[pos_y + 4, pos_x] < 0))):
                return set()
            self.maze[pos_y + 2, pos_x] = -1
            left = 1
            while (self.maze[pos_y, pos_x - left] >= 0) or (self.maze[pos_y + 2, pos_x - left] >= 0):
                # Add walls
                self.maze[pos_y, pos_x - left] = -1
                self.maze[pos_y + 2, pos_x - left] = -1
                
                # Add corridor tile
                if (pos_x - left) % 2 == 1: tiles.add((pos_x - left, pos_y + 1))
                left += 1
            right = 1
            while (self.maze[pos_y, pos_x + right] >= 0) or (self.maze[pos_y + 2, pos_x + right] >= 0):
                # Add walls
                self.maze[pos_y, pos_x + right] = -1
                self.maze[pos_y + 2, pos_x + right] = -1
                
                # Add corridor tile
                if (pos_x + right) % 2 == 1: tiles.add((pos_x + right, pos_y + 1))
                right += 1
        else:
            # Check on double and triple corridors
            if (pos_x != self.x_width - 3) and ((self.maze[pos_y, pos_x + 2] < 0) or
                                                ((pos_x - 2 >= 0) and (self.maze[pos_y, pos_x - 2] < 0)) or
                                                ((pos_x + 4 <= self.x_width) and (self.maze[pos_y, pos_x + 4] < 0))):
                return set()
            self.maze[pos_y, pos_x + 2] = -1
            up = 1
            while (self.maze[pos_y + up, pos_x] >= 0) or (self.maze[pos_y + up, pos_x + 2] >= 0):
                # Add walls
                self.maze[pos_y + up, pos_x] = -1
                self.maze[pos_y + up, pos_x + 2] = -1
                
                # Add corridor tile
                if (pos_y + up) % 2 == 1: tiles.add((pos_x + 1, pos_y + up))
                up += 1
            down = 1
            while (self.maze[pos_y - down, pos_x] >= 0) or (self.maze[pos_y - down, pos_x + 2] >= 0):
                # Add walls
                self.maze[pos_y - down, pos_x] = -1
                self.maze[pos_y - down, pos_x + 2] = -1
                
                # Add corridor tile
                if (pos_y - down) % 2 == 1: tiles.add((pos_x + 1, pos_y - down))
                down += 1
        return tiles
    
    def remove_corridor_doors(self):
        """All the corridors are one."""
        
        def is_corridor(x, y):
            """Check if current wall-position is a corridor"""
            # Horizontal wall upper corridor
            if (self.maze[y, x - 1] < 0) and (self.maze[y + 1, x - 1] < 0) and (self.maze[y, x + 1] < 0) and \
                    self.maze[y + 1, x + 1] < 0:
                return True
            
            # Horizontal wall lower corridor
            elif (self.maze[y, x - 1] < 0) and (self.maze[y - 1, x - 1] < 0) and (self.maze[y, x + 1] < 0) and \
                    self.maze[y - 1, x + 1] < 0:
                return True
            
            # Vertical wall left corridor
            elif (self.maze[y - 1, x] < 0) and (self.maze[y - 1, x - 1] < 0) and (self.maze[y + 1, x] < 0) and \
                    self.maze[y + 1, x - 1] < 0:
                return True
            
            # Vertical wall right corridor
            elif (self.maze[y - 1, x] < 0) and (self.maze[y - 1, x + 1] < 0) and (self.maze[y + 1, x] < 0) and \
                    self.maze[y + 1, x + 1] < 0:
                return True
            
            # No door
            else:
                return False
        
        for pos_x in range(1, self.x_width - 1):
            for pos_y in range(1, self.y_width - 1):
                # Remove wall if it is a corridor
                if (self.maze[pos_y, pos_x] < 0) and is_corridor(x=pos_x, y=pos_y): self.maze[pos_y, pos_x] = 0
    
    def add_wall(self, pos: (tuple, list), hor: bool = True):
        """Add a wall on the given position with the requested direction."""
        assert (len(pos) == 2)
        assert ((pos[0] % 2 == 0) or (pos[1] % 2 == 0))
        
        # Add the wall
        self.maze[pos[1], pos[0]] = -1
        if hor:  # Horizontal
            right = 1
            while self.maze[pos[1], pos[0] + right] >= 0:
                self.maze[pos[1], pos[0] + right] = -1
                right += 1
            left = 1
            while self.maze[pos[1], pos[0] - left] >= 0:
                self.maze[pos[1], pos[0] - left] = -1
                left += 1
        else:  # Vertical
            up = 1
            while self.maze[pos[1] + up, pos[0]] >= 0:
                self.maze[pos[1] + up, pos[0]] = -1
                up += 1
            down = 1
            while self.maze[pos[1] - down, pos[0]] >= 0:
                self.maze[pos[1] - down, pos[0]] = -1
                down += 1
    
    def create_door(self, pos, room_tiles):
        """
        Creates a door between the given position and a neighbouring room. It should always be possible to add a door
        to the given position, hence no checks are done.
        """
        # neighbours contains items (door-pos, corridor-pos)
        neighbours = [((pos[0] + 1, pos[1]), (pos[0] + 2, pos[1])),
                      ((pos[0] - 1, pos[1]), (pos[0] - 2, pos[1])),
                      ((pos[0], pos[1] + 1), (pos[0], pos[1] + 2)),
                      ((pos[0], pos[1] - 1), (pos[0], pos[1] - 2))]
        shuffle(neighbours)
        
        # Create door for first matching position
        for door_pos, neighbouring_pos in neighbours:
            if neighbouring_pos not in room_tiles:
                # Creates door and stops the loop
                self.maze[door_pos[1], door_pos[0]] = 0
                return
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def fill_room(self, pos: (tuple, list), reset: bool = False):
        """Fill a room (put values on 1) to check how many tiles are in it. Only uneven positions are used."""
        assert (pos[0] % 2 == 1) and (pos[1] % 2 == 1)
        
        def get_unfilled_neighbours(p):
            """Get the unfilled neighbours for the given position."""
            neighbours = []
            # Left
            if (self.maze[p[1], p[0] - 1] >= 0) and (self.maze[p[1], p[0] - 2] < 1): neighbours.append((p[0] - 2, p[1]))
            # Right
            if (self.maze[p[1], p[0] + 1] >= 0) and (self.maze[p[1], p[0] + 2] < 1): neighbours.append((p[0] + 2, p[1]))
            # Above
            if (self.maze[p[1] + 1, p[0]] >= 0) and (self.maze[p[1] + 2, p[0]] < 1): neighbours.append((p[0], p[1] + 2))
            # Below
            if (self.maze[p[1] - 1, p[0]] >= 0) and (self.maze[p[1] - 2, p[0]] < 1): neighbours.append((p[0], p[1] - 2))
            return neighbours
        
        # Fill the room
        new_pos = [pos]
        filled = {pos}
        while new_pos:
            current_pos = new_pos.copy()
            new_pos = []
            for pos in current_pos:
                self.maze[pos[1], pos[0]] = 1
                new_pos += get_unfilled_neighbours(pos)
            filled.update(set(new_pos))
        
        # Set intermediate values back to zero if requested
        if reset:
            for x in range(1, self.x_width - 1, 2):
                for y in range(1, self.y_width - 1, 2):
                    self.maze[y, x] = 0
        
        return filled
    
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
    
    def visualize_extend(self, maze):
        """Visualize maze where 1 meter is represented by 10 tiles. No matrix-processing must be done."""
        c = maze.copy()
        plt.figure(figsize=(8, 8))
        plt.imshow(c, origin='lower')
        plt.xticks([i * 11 for i in range(self.cfg.y_axis + 1)])
        plt.yticks([i * 11 for i in range(self.cfg.x_axis + 1)])
        plt.colorbar()
        plt.show()
        plt.close()


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
        if not concat: i += 1


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
    game.player = MarXBot(game=game,
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


def create_game(cfg: GameConfig,
                game_id: int,
                maze: Maze,
                overwrite: bool = False,
                visualize: bool = False,
                ):
    """
    Create a game based on a list of walls.
    
    :param cfg: The game config
    :param game_id: ID of the game (Integer)
    :param maze: The maze on which the game is based
    :param overwrite: Overwrite pre-existing games
    :param visualize: Visualize the calculations
    """
    # Create empty Game instance
    game = Game(config=cfg,
                game_id=game_id,
                overwrite=overwrite,
                silent=True)
    
    # Add additional walls to the game
    game.walls.update(set(maze.get_wall_coordinates()))
    
    # App path to the game
    path_list = maze.get_path_coordinates(target_pos=maze.target, visualize=visualize)
    game.path = {p[0]: p[1] for p in path_list}
    
    # Set the target on the predefined position
    game.target = maze.target
    
    # Create random player
    game.player = MarXBot(game=game,
                          init_pos=Vec2d(cfg.x_axis - 0.5, 0.5),  # Fixed initial position
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
        # for g_id in tqdm(range(1, nr_games + 1), desc="Generating Mazes"):
        for g_id in tqdm([-1]):
            maze = None
            while not maze:
                try:
                    maze = Maze(cfg=config, visualize=args.visualize)
                except MazeMalfunctionException:
                    maze = None
            create_game(cfg=config,
                        game_id=g_id,
                        maze=maze,
                        overwrite=args.overwrite,
                        visualize=args.visualize,
                        )
            
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
                game.get_observation()
                game.step(0, 0)
            except Exception:
                print(f"Faulty created game: {g_id}, please manually redo this one")
                os.remove(f"environment/games_db/game_{g_id:05d}")
