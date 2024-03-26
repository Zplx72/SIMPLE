# Adapted from https://mblogscode.com/2016/06/03/python-naughts-crossestic-tac-toe-coding-unbeatable-ai/

# Metting 2: observation space, you encoded your own game board and your tiles, and they have (qwrikle game) to link it.
# 
from random import Random
import gym
import numpy as np
import random
from termcolor import colored
#importing the module 
import logging 

#now we will Create and configure logger 
logging.basicConfig(filename="std1.log", format='%(asctime)s %(message)s', filemode='w') 

#Let us Create an object 
logger=logging.getLogger() 

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 

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
    def __init__(self, id, _tiles):
        print(f"class Player, __init__, start")
        self.id = id
        self.score = 0
        self._tiles = _tiles
        print(f"From player, self._tiles: {self._tiles}")
        # self.pick_tiles(self._bag_of_tiles)
        print(f"class Player, __init__, end\n")       
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
        logger.debug("__init__: begining")
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
        # self.current_player_num = 1
        # self.current_player_num = 1

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
        self.grid_length = 30
        self.num_squares = self.grid_length * self.grid_length
        self.grid_shape = (self.grid_length, self.grid_length)

        self.shapes = [SHAPES.CIRCLE, SHAPES.DIAMOND, SHAPES.SPARKLE, SHAPES.SQUARE, SHAPES.STAR, SHAPES.TRIANGLE]
        self.colors = [COLORS.BLUE, COLORS.CYAN, COLORS.GREEN, COLORS.MAGENTA, COLORS.RED, COLORS.YELLOW]

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
        self.look_up_float_to_piece = {}

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
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=((self.grid_length * self.grid_length) + self.n_tiles + (self.grid_length * self.grid_length * self.n_tiles + 1),), dtype=np.float32)
        self.verbose = verbose
        logger.debug("__init__: end\n")
    # Anything that part of the RL game is not part of the qwirkle implementation. Like reward, action, 
    
    # Step 2 of conversion, _bag_of_tiles, has been implemented and changed continued. 
    def _generate_new_bag_of_tiles(self):
        self._bag_of_tiles = []
        for i in range(3):
            for c in range(len(self.colors)):
                for s in range(len(self.shapes)):
                    self._bag_of_tiles.append(Piece(color=self.colors[c], shape=self.shapes[s]))    
    

    # step 3 of conversion: add the pick tiles function
    def pick_tiles_player_specific(self, bag_of_tiles):
        rnd = Random()
        _tiles = []
        while len(_tiles) < 6 and len(bag_of_tiles) > 0:
            i = rnd.randint(0, len(bag_of_tiles) - 1)

            _tiles.append(bag_of_tiles.pop(i))    
        return _tiles    
            
    def _hand_pick_specific_tiles(self, bag_of_tiles):
        _tiles = [Piece(color=self.colors[0], shape=self.shapes[0]), Piece(color=self.colors[3], shape= self.shapes[0]), Piece(color=self.colors[4], shape= self.shapes[4]), Piece(color=self.colors[2], shape= self.shapes[4]), Piece(color=self.colors[3], shape= self.shapes[0]), Piece(color=self.colors[3], shape= self.shapes[2])]
        for tile in _tiles:
            for i, bag_tile in enumerate(bag_of_tiles):
                if bag_tile.color == tile.color and bag_tile.shape == tile.shape:
                    bag_of_tiles.pop(i)
                    break
        return _tiles

    def pick_tiles(self, bag_of_tiles):
        rnd = Random()
        # self._tiles = []
        # self._tiles = []
        while len(self._tiles) < 6 and len(bag_of_tiles) > 0:
            i = rnd.randint(0, len(bag_of_tiles) - 1)

            self._tiles.append(bag_of_tiles.pop(i))    
        # return _tiles
        # return _tiles

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
            for c in range(len(self.colors)):
                for s in range(len(self.shapes)):
                    zero_check = round(float_auxilary + (counter * 0.05), 2)
                    if zero_check == 0:
                        zero_check = round(zero_check + 0.05, 2)
                        counter += 1
                    self.look_up_dict[str(Piece(color=self.colors[c], shape=self.shapes[s]))] = zero_check
                    counter += 1
            # print(self.look_up_dict)
        
            # # print(look_up_dict)
            # # print(type(piece))
            # # Assuming 'look_up_dict' is your dictionary
            # first_key = next(iter(look_up_dict))
            # print(type(first_key))
            piece_in_float = self.look_up_dict[str(piece)]
            return piece_in_float

    def float_to_piece_converter(self, _float):
        if len(self.look_up_float_to_piece) == 36:
            float_in_piece = self.look_up_float_to_piece[_float]
            # print(f"self.look_up_float_to_piece: {self.look_up_float_to_piece}")
            return float_in_piece
        else:
            float_auxilary = -0.90
            counter = 0
            for c in range(len(self.colors)):
                for s in range(len(self.shapes)):
                    zero_check = round(float_auxilary + (counter * 0.05), 2)
                    if zero_check == 0:
                        zero_check = round(zero_check + 0.05, 2)
                        counter += 1
                    self.look_up_float_to_piece[zero_check] = Piece(color=self.colors[c], shape=self.shapes[s])
                    counter += 1
            float_in_piece = self.look_up_float_to_piece[_float]
            # print(f"self.look_up_float_to_piece: {self.look_up_float_to_piece}")
            return float_in_piece

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
    #     legal_actions = self.legal_actions()
    
    #     # Change the _tiles to tile_state which will be at most a 6 element list, wiht float numbers encoded. 
    #     tile_state = []
    #     for i in self._tiles:
    #         tile_state.append(self.piece_to_float_converter(i))
        

    #     # Stack the board state and the legal actions along the last dimension
    #     out = np.stack([board_state, tile_state], axis=-1)

    #     return out
    # his function seems to be swapping a player's current tiles with new ones 
    # from the main bag, and then returning the old tiles to the main bag.
    def swap_tiles(self):
        auxilary_bag = self._tiles.copy()
        self._tiles.clear()

        self.pick_tiles(self._bag_of_tiles)
        
        self._bag_of_tiles.extend(auxilary_bag)
        self.flag_board_zero_check = False
        # print("\nTILES ARE SWAPPED!\n")
        logger.debug("\nTILES ARE SWAPPED!\n")
        return


    @property
    def legal_actions(self):
        logger.debug("def legal_actions(), start:")

        # Make all the actions illegal unless proven otherwise
        
        # counter = 0
        
        # In an instance of the game that the tile need to be put down and _is_play_valid would turn false, then we put all the actions to be equal to 1. Meaning it is fine to take any action it is desired. 
        # After that it will never go trough this if statement ever again
        # print(f"This is self.flag_is_board_empty {self.flag_is_board_empty} Also double check if the board is empty {np.all(self.board == float(0))}")

        # This needed to be added to add functionality, of in case the board is empty then the skipping will not be an option
        if self.flag_is_board_empty:
            logger.debug(f" if self.flag_is_board_empty {self.flag_is_board_empty}")
            # print(f"The board is empty now? {np.all(self.board == float(0))}")
            self.function_is_board_empty()
            first_move_legal_actions = np.ones((self.grid_length*self.grid_length*self.n_tiles + 1), np.float32)
            first_move_legal_actions[-1] = 0
            return first_move_legal_actions

        # 
        n_combinations = self.grid_length*self.grid_length*self.n_tiles
        while True:
            logger.debug(f" while True, start:")
            legal_actions = []
            checked_actions = set()
            # Go through all the possible actions.
            # logger.debug(self.board)

            # logger.debug(self._board) 
            logger.debug(f" self._tiles {self._tiles}")
            for i in range(0, self.grid_length*self.grid_length*self.n_tiles):
                
                # Decode the aciton and check wetheer the action is valid.
                tile_index, col, _row = self.action_to_indices(i)
                checked_actions.add((tile_index, col, _row))
                # logger.debug(f"Checking position ({col}, {_row}) for shape: {self._tiles[tile_index]}")
                # print(f"From the legal actions function: self._tiles {self._tiles}")
                bool_valid_play = self._is_play_valid(piece=self._tiles[tile_index], x = col, y = _row)

                # if the action was valid then turn the index of the legal_actions to 1 indicating that the coresponding aciton is indeed valid. 
                if bool_valid_play:
                    legal_actions.append(1)
                else:
                    legal_actions.append(0)
            
        
            num_ones = np.count_nonzero(np.array(legal_actions) == 1)
            logger.debug(f"     Number of ones {num_ones}")
            # n1 = 0
            # for n in legal_actions:
            #     if n == 1:
            #         n1 += 1
            # logger.debug(f"Number of ones (n1): {n1}")
            # Regardless of whether it will be all zeros or not I will need to append a zero at the end.
            # Just for the dimensions to make sense.
            legal_actions.append(0)

            if np.all(np.array(legal_actions) == 0) and self.flag_board_zero_check:
                # print(f"\nself._tiles before swapping: {self._tiles}")
                self.print_tiles(self._tiles)
                self.swap_tiles()
                # print(f"self._tiles after swapping: {self._tiles}")
                self.print_tiles(self._tiles)
                # print(f"\nThis is the if part of the while loop: flag_board_zero_check: {self.flag_board_zero_check} and is all legal_actions zero ? {np.all(np.array(legal_actions) == 0)}\n")
                logger.debug(f"     If np.all(np.array(legal_actions) == 0) {np.all(np.array(legal_actions) == 0)} and self.flag_board_zero_check {self.flag_board_zero_check}")
            else:
                logger.debug(f"     Else np.all(np.array(legal_actions) == 0) {np.all(np.array(legal_actions) == 0)} and self.flag_board_zero_check {self.flag_board_zero_check}")
                logger.debug(f"     While True, End")
                break            
            
            # This if statment checks if the tile has already been swapped and still there is no legal_actions then it will return a numpy array that only the last element is a legal action for it.
            # This will help 
        # print(f"All actions checked? {len(checked_actions) == n_combinations}")
        logger.debug(f" All actions checked? {len(checked_actions) == n_combinations}")
        if (self.flag_board_zero_check == False) and  np.all(np.array(legal_actions) == 0):
            logger.debug(f" Skip conditional, start : if (self.flag_board_zero_check == False) and  np.all(np.array(legal_actions) == 0), ")
            only_skipping_is_legal = np.zeros((self.grid_length*self.grid_length*self.n_tiles + 1), np.float32)
            only_skipping_is_legal[self.grid_length*self.grid_length*self.n_tiles] = 1
            self.flag_board_zero_check = True
            # print(f"only_skipping_is_legal the length: {len(only_skipping_is_legal)}")
            # print(f"only_skipping_is_legal last elements:  {only_skipping_is_legal[-1]}\n")
            logger.debug(f" Skip conditionnal, End.")
            return only_skipping_is_legal


        logger.debug(f" Is legal_actions all zeros? {np.all(np.array(legal_actions) == 0)}")
        logger.debug(f"def legal_actions() end\n")
        return np.array(legal_actions)
    
    @property
    def observation(self):
        logger.debug("def observation() Start:")
        # Get the state of the board
        board_state = self.board.flatten()

        # Change the _tiles to tile_state which will be at most a 6 element list, with float numbers encoded. 
        tile_state = []
        for i in self._tiles:
            tile_state.append(self.piece_to_float_converter(i))
        tile_state = np.array(tile_state).flatten()

        # Compute the legal actions
        legal_actions = self.legal_actions.flatten()        

        # Concatenate the board state and the tile state to create a single 1D array
        out = np.concatenate([board_state, tile_state, legal_actions])
        logger.debug("def observation() End.\n")
        return out
    # @property
    # def observation(self):
    #     # Get the state of the board
    #     # board_state = np.array(self.board).reshape((self.grid_length, self.grid_length))

    #     # Compute the legal actions
    #     legal_actions = np.array(self.legal_actions()).reshape((self.grid_length, self.grid_length))

    #     # Change the _tiles to tile_state which will be at most a self.n_tiles element list, with float numbers encoded.
    #     tile_state = []
    #     for i in self._tiles:
    #         tile_state.append(self.piece_to_float_converter(i))
    #     tile_state = np.array(tile_state).reshape((self.grid_length, self.grid_length))

    #     # Stack the board state, tile state, and the legal actions along the last dimension
    #     out = np.stack([self.board, tile_state, legal_actions], axis=-1)

    #     return out
    # Legal actions here goes through every cell on the board and check
    # if it is empty and if it is legal to place a tile there.

    def action_to_indices(self, action):
        tile_index = action % self.n_tiles
        two_d_index = action // self.n_tiles
        col = two_d_index % self.grid_length
        _row = two_d_index // self.grid_length
        return tile_index, col, _row
    

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
        logger.debug(f"def check_game_over(): Start")
        if not self._bag_of_tiles:
            # Check if the current player has no tiles left
            if not self._tiles:
                logger.debug(f"def check_game_over(), it is game ove, End.")
                return True
        logger.debug(f"def check_game_over(), not over, End.")
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




    def score(self):
        """Return the score for the current turn"""
        logger.debug("def score(), Start:")
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
        logger.debug(f"def score(), end. {score}")
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
        # logger.debug("passes position check")
        # Make sure the placement is not already taken
        if self._board[y][x] is not None:
            return False
        # logger.debug("is not occupied")
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

        # logger.debug("Has adjacent tile")
        # print(f"Before play for {piece}")
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
            # logger.debug(f"check horizontal done, in_plays : {in_plays}")
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
            # logger.debug(f"check vertical done, in_plays : {in_plays}")
            if not in_plays:
                return False

        # print(F"after play for {piece}")
        # Don't test for piece shape/color if no piece provided
        if piece is None:
            return True

        # print(f"before rows for {piece}")
        # Get & Verify all the tiles adjacent horizontally
        row = [piece]
        # logger.debug(f"{row} before the row check row check")
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
        # logger.debug(f"{row}, passed row check")
        # print(f"after rows for {piece}")
        # Get & Verify all the tiles adjacent vertically
        row = [piece]
        # logger.debug(f"{row} before the row check row check")
        t_y = y + 1
        while t_y < len(self._board) and self._board[t_y][x] is not None:
            row.append(self._board[t_y][x])
            t_y += 1

        t_y = y - 1
        while t_y >= 0 and self._board[t_y][x] is not None:
            row.append(self._board[t_y][x])
            t_y -= 1

        # print(f"before columns {piece}")
        # print(f"{row}, ({x}, {y})")
        if not self._is_row_valid(row):
            return False
        
        # logger.debug(f"passed column check {row}")
        # print(f"after columns for {piece}")
        return True

    def _is_row_valid(self, row):
        """If all row colors are equal, check each shape shows up at most once.
           If all shapes are equal, check each color shows up at most once.
           Otherwise the row is invalid."""
        
        if len(row) == 1:
            # logger.debug(f"inside _is_row_valid len(row) {len(row)}, True return")
            return True
        # logger.debug(f"inside _is_row_valid len(row) {len(row)}")
        # print(f"before same colour check for {row[0]}")
        if all(row[i].color == row[0].color for i in range(len(row))):
            shapes = []
            for i in range(len(row)):
                if row[i].shape in shapes:
                    return False
                shapes.append(row[i].shape)
            
            # print(f"after same colour check for {row[0]}")
            # logger.debug(f"after same colour check for {row[0]}")
            
        elif all(row[i].shape == row[0].shape for i in range(len(row))):
            # logger.debug(F"before same shape for {row[0]}")
            colors = []
            for i in range(len(row)):
                if row[i].color in colors:
                    return False
                colors.append(row[i].color)
            # logger.debug(f"after same shape for {row[0]}")
        else:
            # logger.debug(f"Else: {row}")
            return False

        return True


    def step(self, action):
        logger.debug(f"def step() Start:")
        # once it took an action what would you do with that just say if it is a good idea or not. 
        # you don't need tunrs_taken. 
        # consider normalising the reward. 
        reward = [0,0]
        tile_index, col, _row = self.action_to_indices(action)
        # print(f"action: {action}")
        logger.debug(f" action: {action}")

        # This is the first if statement to check whether the action is to skip or not:
        if action == (self.grid_length * self.grid_length * self.n_tiles):
            logger.debug(f" if statement to skip the turn")
            # Don't see the point to add anything else. as everything remains the same.
            self.done = self.check_game_over()
            return self.observation, reward, self.done, {}


        # the three funcitons will be written down here, checks will be done and if it is true then the "_board" will be updated
        # Here is where the functions go. 
        
        # you check if the play is valid. 
        # print(f"From the step function: self._tiles {self._tiles}")
        # print(f"From the step function: self._tiles {self._tiles}")
        bool_valid_play = self._is_play_valid(piece=self._tiles[tile_index], x = col, y = _row)
        logger.debug(f" self._tiles {self._tiles}, tile_index {tile_index}")
        # checks if the board is empty 
        if self.flag_is_board_empty:
            logger.debug(f"     if self.flag_is_board_empty {self.flag_is_board_empty}")
            # All the steps below are mirrored from the else statement.
            self.done = False

            # Board piece assignment.
            self._board[_row][col] = self._tiles[tile_index] 
            self.board[_row][col] = self.piece_to_float_converter(self._tiles[tile_index])
            self._plays.append((col, _row))
            logger.debug(f"     self._plays: {self._plays}")
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
            self._plays_whole_round.extend(self._plays)
            logger.debug(f"     self.plays {self._plays}")
            logger.debug(f"     self.plays_whole_round {self._plays_whole_round}")
            self._plays = []
            
        elif not bool_valid_play:
            logger.debug(f"     elif not bool_valid_play {not bool_valid_play}")
            self.counter += 1
            self.done = False
            reward = [3, 3]
            reward[self.current_player_num] = -3
            self._plays_whole_round.extend(self._plays)
            logger.debug(f"     self.plays {self._plays}")
            logger.debug(f"     self.plays_whole_round {self._plays_whole_round}")
            self._plays = []
            if self.counter >= 50:
                self.done = True
        else:
            logger.debug(f"     else")
            # Two things need to be done, updating both boards: _board and board. 
            self.done = False
            # The None, and tile_index is done. 
            self._board[_row][col] = self._tiles[tile_index]
            
            # The numerical has been done. 
            self.board[_row][col] = self.piece_to_float_converter(self._tiles[tile_index])

            self._plays.append((col, _row))
            logger.debug(f"     self._plays: {self._plays}")

            # Remove the tile from the bag
            self._tiles.pop(tile_index)
            
            # Add a tile to the hand from the bag, WHAT IF? The bag_of_tiles is empty?
            self.pick_tiles(self._bag_of_tiles)
            
            # figuring out some sort of a scoring system. 
            score = self.score()
            reward[self.current_player_num] = score

            # check if the game is over after the action
            self.done = self.check_game_over()
            self._plays_whole_round.extend(self._plays)
            logger.debug(f"     self.plays {self._plays}")
            logger.debug(f"     self.plays_whole_round {self._plays_whole_round}")
            self._plays = []

        # I think if the game is done then the player changes to be the other one?
        # what would that indicate though: The other player is the same agent at different point in time. 
        

        if not self.done:
            logger.debug(f"     if not self.done {not self.done}")
            self.current_player_num = (self.current_player_num + 1) % 2
            logger.debug(f"    About to switch players now current_player number{self.current_player.id}")
            self._tiles = self.current_player._tiles
            self._plays = []
            # self.switch_player()

        print(f"def step(): end\n")
        return self.observation, reward, self.done, {}
    
    # Swtiches player at then end of the round
    # It is used as a property so no need to be called, it will be called instantly 
    @property
    def current_player(self):
        return self.players[self.current_player_num]


    def reset(self):
        logger.debug("def reset(), start: ")
        self.flag_board_zero_check = True

        self.current_player_num = 0

        self.current_player_num = 0
        self.done = False
        self.counter = 0

        # Initialize the bag of tiles going through each color
        # Step 2 of conversion, _bag_of_tiles, has been implemented and changed
        self._bag_of_tiles = []
        self._generate_new_bag_of_tiles()

        # Instansiate players
        self.players = [Player('1', self._hand_pick_specific_tiles(self._bag_of_tiles)), Player('2', self.pick_tiles_player_specific(self._bag_of_tiles))]
        # self.current_player = self.player1

        self._tiles = self.current_player._tiles
        # self.pick_tiles(self._bag_of_tiles)
        # self.current_player._tiles = self._tiles
        # self.pick_tiles(self._bag_of_tiles)
        # self.current_player._tiles = self._tiles
        # print(len(self.bag_of_tiles))
        # This board is purly numeric.
        self.board = np.zeros((self.grid_length, self.grid_length), dtype=np.float32)

        # This board is aligned with what the other source code have. 
        self._board = [[None] * self.grid_length for i in range(self.grid_length)]


        # It is (x,y) i.e. (column, row)
        self._cooridnates = [(16, 6), (16, 7), (14, 8), (15, 8), (16, 8), (17, 8), (18, 8), (19, 8), (16, 9), (16, 10), (16, 11)]
        self._values = [-0.45, -0.55, -0.45, -0.5, -0.4, -0.35, -0.6, -0.55, -0.5, -0.35, -0.6]

        # Putting the tiles on the board
        for value, coordinate in enumerate(self._cooridnates):
            # self.board[y][x], 
            float_to_piece_rep = self.float_to_piece_converter(float(self._values[value]))
            for i, bag_tile in enumerate(self._bag_of_tiles):
                if bag_tile.color == float_to_piece_rep.color and bag_tile.shape == float_to_piece_rep.shape:
                    self._bag_of_tiles.pop(i)
                    break
            self.board[coordinate[1]][coordinate[0]] = self._values[value]
            self._board[coordinate[1]][coordinate[0]] = float_to_piece_rep

        # Initialize the players' hands
        # self.player_hands = [self.draw_tiles(self.n_tiles) for i in range(self.n_players)]
        # Initialize the board
        # 12 has been gotten rid of as, the only thing on the board would be an integer mapped from the tile to the class. 



        # have this flag just to change it to false later on after the first tile is put down
        self.flag_is_board_empty = False

        # It is like a history of plays, in the game this can happen more than once in one round. 
        self._plays = []
        self._plays_whole_round = []
        logger.debug("def reset : End\n")
        logger.debug(f'---- NEW GAME ----')
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
    def print_tiles(self, tiles):
        logger.debug("def print_tiles(), start: ")
        tiles_output = ''
        for tile in tiles:
            tiles_output += colored(tile.shape, tile.color) + ' '
        logger.debug(' Your Tiles: %s' % tiles_output)
        # logger.debug('              1 2 3 4 5 6\n')
        logger.debug(f" Your Tiles: {tiles}")
        logger.debug("def print_tiles(), end")
        

    def print_board_altered(self, show_valid_placements=False, radius=15):
        logger.debug("  def print_board_altered, start:")
        if len(self._plays_whole_round) == 0:
            print('     The board is empty.')
            logger.debug("  def print_board_altered, end:")
            return

        # Get the coordinates of the last played tile
        last_play = self._plays_whole_round[-1] if self._plays_whole_round else (0, 0)

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
                    if (x, y) in self._plays_whole_round:
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
            # logger.debug(i_display, lines[i], i_display)
            print(i_display, lines[i], i_display)
        # logger.debug(self.board)
        # logger.debug(self._board)
        logger.debug("  def print_board_altered, end")
        # logger.debug(f"\nThe tiles right now {self._tiles}\n ")
        # logger.debug(f"self._plays: {self._plays}")
        # logger.debug("print_board_altered : end")

    ### HERE TO UNCOMMENT
    def render(self, mode='human', close=False, verbose = True):
        logger.debug("def render() start: ")
        # logger.debug('')
        if close:
            return
        if self.done:
            logger.debug(f' GAME OVER')
        else:
            logger.debug(f" It is Player {self.current_player.id}'s turn to move")
        self.print_board_altered()
        self.print_tiles(self._tiles)   
        logger.debug("def render() end.\n")

            
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