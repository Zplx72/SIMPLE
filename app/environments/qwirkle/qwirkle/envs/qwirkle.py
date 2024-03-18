# Adapted from https://mblogscode.com/2016/06/03/python-naughts-crossestic-tac-toe-coding-unbeatable-ai/

# Metting 2: observation space, you encoded your own game board and your tiles, and they have (qwrikle game) to link it.
# 
from random import Random
import gym
import numpy as np
import random
from termcolor import colored


from stable_baselines import logger

# colours = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
# shapes = ['circle', 'square', 'diamond', 'clover', 'star', 'cross']
# Step 1: conversion of how the tiles are represented.

class COLORS:
    RED = 'red'
    YELLOW = 'yellow'
    GREEN = 'green'
    CYAN = 'cyan'
    MAGENTA = 'magenta'
    BLUE = 'blue'


class SHAPES:
    # TRIANGLE = 'triangle'
    # DIAMOND = 'diamond'
    # SQUARE = 'square'
    # CIRCLE = 'circle'
    # STAR = 'star'
    # SPARKLE = 'sparkle'
    TRIANGLE = '▲'
    DIAMOND = '◆'
    SQUARE = '■'
    CIRCLE = '●'
    STAR = '★'
    SPARKLE = '❈'


class Piece:
    def __init__(self, color=None, shape=None):
        self.color = color
        self.shape = shape

    def __str__(self):
        return '%s %s' % (self.color, self.shape)

    def __repr__(self):
        return self.__str__()


# if you wanted to add a method that allows a player to swap tiles from their hand with tiles from the bag, it might make sense to add this method to a Player class.
# However, if you later decide to add more complexity to your game (like more player-specific attributes or methods), you might want to consider using a `Player` class. For now, your current implementation is functional and appropriate for your game's requirements.

class Player():
    def __init__(self, id):
        self.id = id
        self.score = 0
        self.tiles = []        

class Token():
    def __init__(self, symbol, number):
        self.number = number
        self.symbol = symbol
        

    
##### This is the first step. The most important is to think about how to encode the state space and action space. And then it's the reward function. DONE
##### Check how the tictactooe is conceptually encoded and try to conceptually encode qwrikle. and then try to implement it.  Chekc how they all games do ti and see how you can 
##### no matter how you  are encoding the board it needst be passed as a list at the end. DONE
##### Islegal move is a good place to start, but how you can access the space.
class QwirkleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        # The first place to change
        # Only two players
        super(QwirkleEnv, self).__init__()
        self.name = 'qwirkle'
        self.manual = manual
        
        # color_to_dimesion 
        # self.colours = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        # self.shapes = ['circle', 'square', 'diamond', 'clover', 'star', 'cross']
        self.n_tiles = 6
        self.n_players = 2
        self.current_player_num = 1

        # # Initialize the bag of tiles going through each color
        # # Step 2 of conversion, _bag_of_tiles, has been implemented and changed
        # self._bag_of_tiles = []
        # self._generate_new_bag_of_tiles()

        # self._tiles = []
        # self.pick_tiles(self._bag_of_tiles)
        # # print(len(self.bag_of_tiles))

        # # Initialize the players' hands
        # # self.player_hands = [self.draw_tiles(self.n_tiles) for i in range(self.n_players)]

        # No need for all the grid stuff perhaps something similar.
        self.grid_length = 91
        self.num_squares = self.grid_length * self.grid_length
        self.grid_shape = (self.grid_length, self.grid_length)
    
        # Initialize the board
        # 12 has been gotten rid of as, the only thing on the board would be an integer mapped from the tile to the class. 

        # # This board is purly numeric.
        # self.board = np.zeros((self.grid_length, self.grid_length), dtype=np.float32)

        # # This board is aligned with what the other source code have. 
        # self._board = [[None] * self.grid_length for i in range(self.grid_length)]

        # # have this flag just to change it to false later on after the first tile is put down
        # self.flag_is_board_empty = True

        # # It is like a history of plays, in the game this can happen more than once in one round. 
        # self._plays = []

        #Most importatn one is the action space and observation space
        # The action space is the number of squares on the board, as you can place a token on any square
        # self.action_space = gym.spaces.Discrete(self.num_squares)

        self.action_space = gym.spaces.Discrete(self.n_tiles*self.grid_length*self.grid_length)
        # The set of all possible actions an agent could take. In this case is the index of the tiles of the player's hand.
        # gym.spaces.Discrete(self.n_tiles*self.grid_length*self.grid_length),  # The player's hand has 6 tiles
        # #gym.spaces.MultiBinary(self.n_tiles)
        # gym.spaces.MultiBinary((self.grid_length, self.grid_length))  # The board is grid_length x grid_length
        # Introduce the tile you are placing
        self.look_up_dict = {}

        # For a game like Tic Tac Toe, the observation space might be a 3x3 grid
        # where each cell can be in one of three states (empty, occupied by player 1, or occupied by player 2)
        # self.observation_space = gym.spaces.Box(-1, 1, self.grid_shape+(2,))
        # defines the structure of the observations that the environment provides to the agent.
        # NEW: you need to observe the hand as well
        # NEW: put what the algorithm does as a flowchart. 
        # self.observation_space = gym.spaces.Tuple((
        #     # step 4: observation space is modified, to encode each tile to be an float between -1 and 1.
        #     # Question: is there a significance in if the box value is negative or postive????
        #     gym.spaces.Box(low=-1, high=1, shape=(self.n_tiles,), dtype=np.float32),  # Player's hand
        #     gym.spaces.Box(low=-1, high=1, shape=(self.grid_length, self.grid_length), dtype=np.float32)  # Board
        # ))     
        # Instead of an observation space that is a tuple. I added a Box that concatinates the board and the tiles on one dimension. 
        ### ONE VERY BIG PROBLEM, What happens if the tiles are not 6? Should I just add zero to the end of it?
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=((self.grid_length * self.grid_length) + self.n_tiles,), dtype=np.float32)
        self.verbose = verbose

    # Anything that part of the RL game is not part of the qwirkle implementation. Like reward, action, 
    
    # Step 2 of conversion, _bag_of_tiles, has been implemented and changed continued. 
    def _generate_new_bag_of_tiles(self):
        self._bag_of_tiles = []

        shapes = [
            SHAPES.CIRCLE,
            SHAPES.DIAMOND,
            SHAPES.SPARKLE,
            SHAPES.SQUARE,
            SHAPES.STAR,
            SHAPES.TRIANGLE
        ]

        colors = [
            COLORS.BLUE,
            COLORS.CYAN,
            COLORS.GREEN,
            COLORS.MAGENTA,
            COLORS.RED,
            COLORS.YELLOW
        ]

        for i in range(3):
            for c in range(len(colors)):
                for s in range(len(shapes)):
                    self._bag_of_tiles.append(Piece(color=colors[c], shape=shapes[s]))    
    

    # step 3 of conversion: add the pick tiles function
    def pick_tiles(self, bag_of_tiles):
        rnd = Random()
        while len(self._tiles) < 6 and len(bag_of_tiles) > 0:
            i = rnd.randint(0, len(bag_of_tiles) - 1)

            self._tiles.append(bag_of_tiles.pop(i))    

    # converts the piece from an object to it's float represetnation. 
            # it checks if the dictionary exists and if so it will look up the color and shape to find the decimal for it
            # if not it creates the dictionary and does that. 
    def piece_to_float_converter(self, piece):
        piece = str(piece)
        if len(self.look_up_dict) == 36:
            piece_in_float = self.look_up_dict[piece]
            return piece_in_float
        else:
            float_auxilary = -0.90
            counter = 0
            shapes = [
                SHAPES.CIRCLE,
                SHAPES.DIAMOND,
                SHAPES.SPARKLE,
                SHAPES.SQUARE,
                SHAPES.STAR,
                SHAPES.TRIANGLE
            ]

            colors = [
                COLORS.BLUE,
                COLORS.CYAN,
                COLORS.GREEN,
                COLORS.MAGENTA,
                COLORS.RED,
                COLORS.YELLOW
            ]

            for c in range(len(colors)):
                for s in range(len(shapes)):
                    zero_check = round(float_auxilary + (counter * 0.05), 2)
                    if zero_check == 0:
                        zero_check = round(zero_check + 0.05, 2)
                        counter += 1
                    self.look_up_dict[str(Piece(color=colors[c], shape=shapes[s]))] = zero_check
                    counter += 1
        
            # # print(look_up_dict)
            # # print(type(piece))
            # # Assuming 'look_up_dict' is your dictionary
            # first_key = next(iter(look_up_dict))
            # print(type(first_key))
            piece_in_float = self.look_up_dict[str(piece)]
            return piece_in_float


    # NEW: I only need tot store the action and the state, I need to call over the other game.  
    #If there are tiles left in the bag, it randomly selects one, removes it from the bag, and adds it to 
    #the list of drawn tiles. If there are no tiles left in the bag, it breaks out of the loop and returns the tiles that were drawn. 
    #This way, the game can continue even if there are no tiles left to draw.          
    # def draw_tiles(self, num_tiles):
    #     tiles = []
    #     for i in range(num_tiles):
    #         if self.bag_of_tiles:
    #             tile = random.choice(self.bag_of_tiles)
    #             self.bag_of_tiles.remove(tile)
    #             tiles.append(tile)
    #         else:
    #             break
    #     return tiles   

    # ATENTION!!!!!#### Step 5 of conversion: get rid of place_tile as it is not needed in this format. 
    # When you progress more you will know how to change it to convert the tile from the class tiles to different floats between -1 and 1.
            

    # def place_tile(self, row, col, color, shape):
    #     # Reset the cell's state
    #     self.board[row, col] = np.zeros(12, dtype=np.int32)

    #     # Set the dimensions corresponding to the tile's color and shape
    #     self.board[row, col, self.color_to_dimension[color]] = 1
    #     self.board[row, col, self.shape_to_dimension[shape]] = 1  


    # @property
    # def observation(self):
    #     if self.players[self.current_player_num].token.number == 1:
    #         # This evaluates moves, so this encodes the entire game state, in qwirkle is different size of board.
    #         # Try a maximum size and a constant sizee(finite number of size)
    #         # Max size of board can be caluclated.  maybe a rectangle 91 by 91. DONE
    #         # Very important function to change
    #         # functionally should be the smae state if tranformation happens. rotation and transformation should not be a problem VERY IMPORTANT.
    #         # You want the state spce should be as small as possible FOR LATER
    #         position = np.array([x.number for x in self.board]).reshape(self.grid_shape)
    #     else:
    #         position = np.array([-x.number for x in self.board]).reshape(self.grid_shape)

    #     la_grid = np.array(self.legal_actions).reshape(self.grid_shape)
    #     out = np.stack([position,la_grid], axis = -1)
    #     return out

    # This is different that observation in init. this runs every round but init runs only once. PS, very useful comment.
    
    # Original one
    # @property
    # def observation(self):
    #     # Get the state of the board
    #     board_state = self.board
        
    #     # So you get the legal actions based on the board state
    #     # Compute the legal actions, 
    #     # legal_actions = self.legal_actions
    
    #     # Change the _tiles to tile_state which will be at most a 6 element list, wiht float numbers encoded. 
    #     tile_state = []
    #     for i in self._tiles:
    #         tile_state.append(self.piece_to_float_converter(i))
        

    #     # Stack the board state and the legal actions along the last dimension
    #     out = np.stack([board_state, tile_state], axis=-1)

    #     return out
    
    @property
    def observation(self):
        # Get the state of the board
        board_state = self.board.flatten()

        # Change the _tiles to tile_state which will be at most a 6 element list, with float numbers encoded. 
        tile_state = []
        for i in self._tiles:
            tile_state.append(self.piece_to_float_converter(i))
        tile_state = np.array(tile_state).flatten()

        # Concatenate the board state and the tile state to create a single 1D array
        out = np.concatenate([board_state, tile_state])

        return out
    

    # Legal actions here goes through every cell on the board and check
    # if it is empty and if it is legal to place a tile there.
    @property
    def legal_actions(self):
        legal_actions = np.zeros((self.grid_length, self.grid_length), dtype=np.float32)
        return legal_actions
        # # Iterate over the cells on the board
        # for i in range(self.board.shape[0]):
        #     for j in range(self.board.shape[1]):
        #         # Check if the cell is empty
        #         #if np.all(self.board[i, j] == np.eye(12)[0]):  # Assuming the first tile is the "empty" tile
        #         if self.board[i, j].all(0):
        #             # Check if placing a tile in this cell would be a legal action
        #             # is_legal_action is a function that you need to write, 
        #             # it should return True if the action is legal and False otherwise
        #             if self.board[i+1, j].any(1) or self.board[i-1, j].any(1) or self.board[i, j+1].any(1) or self.board[i, j-1].any(1): 
        #                 legal_actions[i, j] = 1
        #             # if self.is_legal_action(i, j):
        #             #     legal_actions[i, j] = 1


    # Start what you need and try to figure out how to do it. 

    # def is_legal_action(self, row, col):

    # check where this is coming from. 
    # def square_is_player(self, square, player):
        
    #     return self.board[square].number == self.players[player].token.number

    # again qwirkle implementation would havve this
    def check_game_over(self):
        # Check if the bag of tiles is empty
        if not self._bag_of_tiles:
            # Check if the current player has no tiles left
            if not self._tiles:
                return True
        return False

    # @property
    # # Nothing need to be changed, probably
    # def current_player(self):
    #     return self.players[self.current_player_num]

    # This is interesting part for reward function. You may need functions from qwirkle implementation. No set answer how the reward funtion would work
    # Easy mode assume the reward funtion doesknow everyone's state eventhough imperfect information.
    # Reward funciton:
    # 1. Immediate Reward for Valid Move, 2. Reward Based on Points Scored:
    # 3. Bonus Reward for Qwirkle 4. Penalty for Invalid Move: 
    # 5. End Game Reward/Penalty

    # Inputs of the reward funciton: Current state, action and next state. 
    # meeting : what doe sqwirkle do for illegal move
    # how strong do you want to punish, (maybe end the game) but you need to check the what does it say for illegal.
    # you need to check if you need to end the game or not.
    # for tictactoe it is 1 0 -1 but then the reward for qwirkle is differetn points. 
    ### They have taken the step, after they make a move you day if the move was good or bad.
    # self does contain the board. 

    # For testing purposes.

    # Check if the board is empty or not
    def function_is_board_empty(self):
        if np.all(self.board == float(0)):
            self.flag_is_board_empty = True
            # print(f"The 'if' part of the board, flag is: {self.flag_is_board_empty}")            
        else:
            self.flag_is_board_empty = False
            # print(f"The 'else' part of the board, flag is: {self.flag_is_board_empty}")  
        return self.flag_is_board_empty


    def action_to_indices(self, action):
        tile_index = action % self.n_tiles
        two_d_index = action // self.n_tiles
        col = two_d_index % self.grid_length
        _row = two_d_index // self.grid_length
        return tile_index, col, _row

    def score(self):
        """Return the score for the current turn"""
        if len(self._plays) == 0:
            return 0

        score = 0
        scored_horizontally = []
        scored_vertically = []

        for play in self._plays:
            x, y = play

            min_x = x
            while min_x - 1 >= 0 and self._board[y][min_x - 1] is not None:
                min_x -= 1

            max_x = x
            while max_x + 1 < len(self._board[y]) and self._board[y][max_x + 1] is not None:
                max_x += 1

            if min_x != max_x:
                qwirkle_count = 0
                for t_x in range(min_x, max_x + 1):
                    if (t_x, y) not in scored_horizontally:
                        score += 1
                        qwirkle_count += 1
                        scored_horizontally.append((t_x, y))

                        if (x, y) not in scored_horizontally:
                            score += 1
                            qwirkle_count += 1
                            scored_horizontally.append((x, y))
                    t_x += 1

                if qwirkle_count == 6:
                    score += 6

            min_y = y
            while min_y - 1 >= 0 and self._board[min_y - 1][x] is not None:
                min_y -= 1

            max_y = y
            while max_y + 1 < len(self._board) and self._board[max_y + 1][x] is not None:
                max_y += 1

            if min_y != max_y:
                qwirkle_count = 0
                for t_y in range(min_y, max_y + 1):
                    if (x, t_y) not in scored_vertically:
                        score += 1
                        qwirkle_count += 1
                        scored_vertically.append((x, t_y))

                        if (x, y) not in scored_vertically:
                            score += 1
                            qwirkle_count += 1
                            scored_vertically.append((x, y))
                    t_y += 1

                if qwirkle_count == 6:
                    score += 6

        return score

    def _is_play_valid(self, piece, x, y):
        """Validates a move is within the board, not on the corners, not
           replacing a existing piece, adjacent to an existing tile and valid in
           its row/column"""

        # Make sure the placement is not on a corner and is inside the board
        if x < 0 or x >= len(self._board[0]):
            return False
        if y < 0 or y >= len(self._board):
            return False
        if x == 0 and y == 0:
            return False
        if x == 0 and y == len(self._board) - 1:
            return False
        if x == len(self._board[0]) - 1 and y == len(self._board) - 1:
            return False
        if x == len(self._board[0]) - 1 and y == 0:
            return False

        # Make sure the placement is not already taken
        if self._board[y][x] is not None:
            return False

        # Make sure the placement has at least one adjacent placement
        adjacent_checks = []
        if y - 1 >= 0:
            adjacent_checks.append((self._board[y - 1][x] is None))
        if y + 1 < len(self._board):
            adjacent_checks.append((self._board[y + 1][x] is None))
        if x - 1 >= 0:
            adjacent_checks.append((self._board[y][x - 1] is None))
        if x + 1 < len(self._board[y]):
            adjacent_checks.append((self._board[y][x + 1] is None))

        if all(adjacent_checks):
            return False

        # Validate the play connects to an existing play
        plays = [(play[0], play[1]) for play in self._plays]
        if len(plays) > 0:
            check_horizontal = True
            check_vertical = True
            if len(plays) > 1:
                if plays[0][0] == plays[1][0]:
                    check_horizontal = False
                if plays[0][1] == plays[1][1]:
                    check_vertical = False

            in_plays = False

            if check_horizontal:
                t_x = x
                while t_x - 1 >= 0 and self._board[y][t_x - 1] is not None:
                    t_x -= 1
                    if (t_x, y) in plays:
                        in_plays = True

                t_x = x
                while t_x + 1 < len(self._board[y]) and self._board[y][t_x + 1] is not None:
                    t_x += 1
                    if (t_x, y) in plays:
                        in_plays = True

            if check_vertical:
                t_y = y
                while t_y - 1 >= 0 and self._board[t_y - 1][x] is not None:
                    t_y -= 1
                    if (x, t_y) in plays:
                        in_plays = True

                t_y = y
                while t_y + 1 < len(self._board) and self._board[t_y + 1][x] is not None:
                    t_y += 1
                    if (x, t_y) in plays:
                        in_plays = True

            if not in_plays:
                return False

        # Don't test for piece shape/color if no piece provided
        if piece is None:
            return True

        # Get & Verify all the tiles adjacent horizontally
        row = [piece]
        t_x = x + 1
        while t_x < len(self._board[0]) and self._board[y][t_x] is not None:
            row.append(self._board[y][t_x])
            t_x += 1

        t_x = x - 1
        while t_x >= 0 and self._board[y][t_x] is not None:
            row.append(self._board[y][t_x])
            t_x -= 1

        if not self._is_row_valid(row):
            return False

        # Get & Verify all the tiles adjacent vertically
        row = [piece]
        t_y = y + 1
        while t_y < len(self._board) and self._board[t_y][x] is not None:
            row.append(self._board[t_y][x])
            t_y += 1

        t_y = y - 1
        while t_y >= 0 and self._board[t_y][x] is not None:
            row.append(self._board[t_y][x])
            t_y -= 1

        if not self._is_row_valid(row):
            return False

        return True

    def _is_row_valid(self, row):
        """If all row colors are equal, check each shape shows up at most once.
           If all shapes are equal, check each color shows up at most once.
           Otherwise the row is invalid."""

        if len(row) == 1:
            return True

        if all(row[i].color == row[0].color for i in range(len(row))):
            shapes = []
            for i in range(len(row)):
                if row[i].shape in shapes:
                    return False
                shapes.append(row[i].shape)

        elif all(row[i].shape == row[0].shape for i in range(len(row))):
            colors = []
            for i in range(len(row)):
                if row[i].color in colors:
                    return False
                colors.append(row[i].color)

        else:
            return False

        return True


    def step(self, action):
        # once it took an action what would you do with that just say if it is a good idea or not. 
        # you don't need tunrs_taken. 
        # consider normalising the reward. 
        reward = [0,0]
        tile_index, col, _row = self.action_to_indices(action)

        # the three funcitons will be written down here, checks will be done and if it is true then the "_board" will be updated
        # Here is where the functions go. 

        # you check if the play is valid. 
        bool_valid_play = self._is_play_valid(piece=self._tiles[tile_index], x = col, y = _row)
        
        # checks if the board is empty 
        if self.flag_is_board_empty:
            # All the steps below are mirrored from the else statement.
            self.done = False

            # Board piece assignment.
            self._board[_row][col] = self._tiles[tile_index] 
            self.board[_row][col] = self.piece_to_float_converter(self._tiles[tile_index])
            # print(f"Floating converter: {self.piece_to_float_converter(self._tiles[tile_index])}")
            # print(self.board)
            # Popping the tile
            self._tiles.pop(tile_index)
            self.pick_tiles(self._bag_of_tiles)       

            # function is board empty runs to flip the flag
            self.function_is_board_empty()
            # print(f"the flag: {self.flag_is_board_empty}")
            # print(f"the all zero part {np.all(self.board != 0)}")

            # score, reward and done.
            score = self.score()
            reward[self.current_player_num] = score            
            self.done = self.check_game_over()
            
        elif not bool_valid_play:
            self.done = True
            reward = [10, 10]
            reward[self.current_player_num] = -10
        
        else:
            # Two things need to be done, updating both boards: _board and board. 
            self.done = False
            # The None, and tile_index is done. 
            self._board[_row][col] = self._tiles[tile_index]
            
            # The numerical has been done. 
            self.board[_row][col] = self.piece_to_float_converter(self._tiles[tile_index])

            # Remove the tile from the bag
            self._tiles.pop(tile_index)
            
            # Add a tile to the hand from the bag, WHAT IF? The bag_of_tiles is empty?
            self.pick_tiles(self._bag_of_tiles)
            
            # figuring out some sort of a scoring system. 
            score = self.score()
            reward[self.current_player_num] = score

            # check if the game is over after the action
            self.done = self.check_game_over()

        # I think if the game is done then the player changes to be the other one?
        # what would that indicate though: The other player is the same agent at different point in time. 
        

        if not self.done:
            self.current_player_num = (self.current_player_num + 1) % 2

        
        return self.observation, reward, self.done, {}



    def reset(self):
        self.current_player_num = 1
        self.done = False

        # Initialize the bag of tiles going through each color
        # Step 2 of conversion, _bag_of_tiles, has been implemented and changed
        self._bag_of_tiles = []
        self._generate_new_bag_of_tiles()

        self._tiles = []
        self.pick_tiles(self._bag_of_tiles)
        # print(len(self.bag_of_tiles))

        # Initialize the players' hands
        # self.player_hands = [self.draw_tiles(self.n_tiles) for i in range(self.n_players)]
        # Initialize the board
        # 12 has been gotten rid of as, the only thing on the board would be an integer mapped from the tile to the class. 

        # This board is purly numeric.
        self.board = np.zeros((self.grid_length, self.grid_length), dtype=np.float32)

        # This board is aligned with what the other source code have. 
        self._board = [[None] * self.grid_length for i in range(self.grid_length)]

        # have this flag just to change it to false later on after the first tile is put down
        self.flag_is_board_empty = True

        # It is like a history of plays, in the game this can happen more than once in one round. 
        self._plays = []
        return self.observation
 
    # # Dependent on how you define the board and how to reset this.
    # def reset(self):
    #     # self.board = [Token('.', 0)] * self.num_squares
    #     # self.players = [Player('1', Token('X', 1)), Player('2', Token('O', -1))]

    #     # Initialize the bag of tiles going through each color
    #     self.bag_of_tiles = [(colour, shape) for colour in self.colours for shape in self.shapes for i in range(3)]
    #     print(len(self.bag_of_tiles))
    #     # Initialize the players' hands
    #     self.player_hands = [self.draw_tiles(self.n_tiles) for i in range(self.n_players)]

    #     # Reset the board
    #     self.board = np.zeros((self.grid_length, self.grid_length, 12), dtype=np.int32)

    #     # set current player
    #     self.current_player_num = random.randint(0, 1)
    #     self.turns_taken = 0
    #     self.done = False
    #     #logger.debug(f'\n\n---- NEW GAME ----')
    #     return self.observation

    # Map how it outputs the game on cml

    # def print_board(self, show_valid_placements=False):
    #     if len(self._plays) == 0:
    #         print('The board is empty.')
    #         return

    #     # valid_plays = self.valid_plays()
    #     lines = []
    #     for y in range(len(self._board)):
    #         line = ''
    #         for x in range(len(self._board[y])):
    #             if self._board[y][x] is not None:
    #                 if (x, y) in self._plays:
    #                     line += colored(self._board[y][x].shape + ' ', self._board[y][x].color, 'on_white')
    #                 else:
    #                     line += colored(self._board[y][x].shape + ' ', self._board[y][x].color)
    #             # elif (x, y) in valid_plays and show_valid_placements:
    #             #     line += colored('☐', 'white') + ' '
    #             else:
    #                 line += '  '

    #         lines.append(line)

    #     # add in the top coord line
    #     line = ''.join([chr(65 + i) + ' ' for i in range(len(self._board[0]))])
    #     lines.insert(0, line)
    #     lines.append(line)

    #     for i in range(0, len(lines)):
    #         i_display = str(i).zfill(2) if 0 < i < len(lines) - 1 else '  '
    #         print(i_display, lines[i], i_display)

    def print_board_altered(self, show_valid_placements=False, radius=10):
        if len(self._plays) == 0:
            print('The board is empty.')
            # return

        # Get the coordinates of the last played tile
        last_play = self._plays[-1] if self._plays else (0, 0)

        # Calculate the range of cells to print
        x_start = max(0, last_play[0] - radius)
        x_end = min(len(self._board[0]), last_play[0] + radius)
        y_start = max(0, last_play[1] - radius)
        y_end = min(len(self._board), last_play[1] + radius)

        lines = []
        for y in range(y_start, y_end):
            line = ''
            for x in range(x_start, x_end):
                if self._board[y][x] is not None:
                    if (x, y) in self._plays:
                        line += colored(self._board[y][x].shape + ' ', self._board[y][x].color, 'on_white')
                    else:
                        line += colored(self._board[y][x].shape + ' ', self._board[y][x].color)
                else:
                    line += '  '
            lines.append(line)

        # add in the top coord line
        line = ''.join([chr(65 + i) + ' ' for i in range(x_start, x_end)])
        lines.insert(0, line)
        lines.append(line)

        for i in range(0, len(lines)):
            i_display = str(i + y_start).zfill(2) if 0 < i < len(lines) - 1 else '  '
            print(i_display, lines[i], i_display)


    ### HERE TO UNCOMMENT
    def render(self, mode='human', close=False, verbose = True):
        logger.debug('')
        if close:
            return
        if self.done:
            logger.debug(f'GAME OVER')
        else:
            logger.debug(f"It is Player {self.current_player_num}'s turn to move")
        self.print_board_altered()
            
        # logger.debug(' '.join([x.symbol for x in self.board[:self.grid_length]]))
        # logger.debug(' '.join([x.symbol for x in self.board[self.grid_length:self.grid_length*2]]))
        # logger.debug(' '.join([x.symbol for x in self.board[(self.grid_length*2):(self.grid_length*3)]]))

        # if self.verbose:
        #     logger.debug(f'\nObservation: \n{self.observation}')
        
        # if not self.done:
        #     logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

    # This is to make a move. Given a certain state of the board what are you doing next.
    # Also log to report the what is happening
#     def rules_move(self):
#         if self.current_player.token.number == 1:
#             b = [x.number for x in self.board]
#         else:
#             b = [-x.number for x in self.board]

#         # Check computer win moves
#         for i in range(0, self.num_squares):
#             if b[i] == 0 and testWinMove(b, 1, i):
#                 logger.debug('Winning move')
#                 return self.create_action_probs(i)
#         # Check player win moves
#         for i in range(0, self.num_squares):
#             if b[i] == 0 and testWinMove(b, -1, i):
#                 logger.debug('Block move')
#                 return self.create_action_probs(i)
#         # Check computer fork opportunities
#         for i in range(0, self.num_squares):
#             if b[i] == 0 and testForkMove(b, 1, i):
#                 logger.debug('Create Fork')
#                 return self.create_action_probs(i)
#         # Check player fork opportunities, incl. two forks
#         playerForks = 0
#         for i in range(0, self.num_squares):
#             if b[i] == 0 and testForkMove(b, -1, i):
#                 playerForks += 1
#                 tempMove = i
#         if playerForks == 1:
#             logger.debug('Block One Fork')
#             return self.create_action_probs(tempMove)
#         elif playerForks == 2:
#             for j in [1, 3, 5, 7]:
#                 if b[j] == 0:
#                     logger.debug('Block 2 Forks')
#                     return self.create_action_probs(j)
#         # Play center
#         if b[4] == 0:
#             logger.debug('Play Centre')
#             return self.create_action_probs(4)
#         # Play a corner
#         for i in [0, 2, 6, 8]:
#             if b[i] == 0:
#                 logger.debug('Play Corner')
#                 return self.create_action_probs(i)
#         #Play a side
#         for i in [1, 3, 5, 7]:
#             if b[i] == 0:
#                 logger.debug('Play Side')
#                 return self.create_action_probs(i)


#     def create_action_probs(self, action):
#         action_probs = [0.01] * self.action_space.n
#         action_probs[action] = 0.92
#         return action_probs   

# # helper funciton for isgame over. Check if it is being used from the outside. Just check that if it is NOT called model.py.
# def checkWin(b, m):
#     return ((b[0] == m and b[1] == m and b[2] == m) or  # H top
#             (b[3] == m and b[4] == m and b[5] == m) or  # H mid
#             (b[6] == m and b[7] == m and b[8] == m) or  # H bot
#             (b[0] == m and b[3] == m and b[6] == m) or  # V left
#             (b[1] == m and b[4] == m and b[7] == m) or  # V centre
#             (b[2] == m and b[5] == m and b[8] == m) or  # V right
#             (b[0] == m and b[4] == m and b[8] == m) or  # LR diag
#             (b[2] == m and b[4] == m and b[6] == m))  # RL diag


# def checkDraw(b):
#     return 0 not in b

# # again a helper funciton. probably something similar but the board is different
# def getBoardCopy(b):
#     # Make a duplicate of the board. When testing moves we don't want to 
#     # change the actual board
#     dupeBoard = []
#     for j in b:
#         dupeBoard.append(j)
#     return dupeBoard

# # test functions, what does it test. need to write your own test.
# def testWinMove(b, mark, i):
#     # b = the board
#     # mark = 0 or X
#     # i = the square to check if makes a win 
#     bCopy = getBoardCopy(b)
#     bCopy[i] = mark
#     return checkWin(bCopy, mark)


# def testForkMove(b, mark, i):
    # Determines if a move opens up a fork
    # bCopy = getBoardCopy(b)
    # bCopy[i] = mark
    # winningMoves = 0
    # for j in range(0, 9):
    #     if testWinMove(bCopy, mark, j) and bCopy[j] == 0:
    #         winningMoves += 1
    # return winningMoves >= 2