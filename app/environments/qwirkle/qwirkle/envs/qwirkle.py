# Adapted from https://mblogscode.com/2016/06/03/python-naughts-crossestic-tac-toe-coding-unbeatable-ai/

# Metting 2: observation space, you encoded your own game board and your tiles, and they have (qwrikle game) to link it.
# 
from random import Random
import gym
import numpy as np


#from stable_baselines import logger

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
    TRIANGLE = 'triangle'
    DIAMOND = 'diamond'
    SQUARE = 'square'
    CIRCLE = 'circle'
    STAR = 'star'
    SPARKLE = 'sparkle'


class Piece:
    def __init__(self, color=None, shape=None):
        self.color = color
        self.shape = shape

    def __str__(self):
        return '%s %s' % (self.color, self.shape)

    def __repr__(self):
        return self.__str__()



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

        # Initialize the bag of tiles going through each color
        # Step 2 of conversion, _bag_of_tiles, has been implemented and changed
        self._bag_of_tiles = []
        self._tiles = []
        # print(len(self.bag_of_tiles))

        # Initialize the players' hands
        # self.player_hands = [self.draw_tiles(self.n_tiles) for i in range(self.n_players)]

        # No need for all the grid stuff perhaps something similar.
        self.grid_length = 91
        self.num_squares = self.grid_length * self.grid_length
        self.grid_shape = (self.grid_length, self.grid_length)
    
        # Initialize the board
        self.board = np.zeros((self.grid_length, self.grid_length, 12), dtype=np.int32)


        #Most importatn one is the action space and observation space
        # The action space is the number of squares on the board, as you can place a token on any square
        # self.action_space = gym.spaces.Discrete(self.num_squares)

        self.action_space = gym.spaces.Tuple((
        # The set of all possible actions an agent could take. In this case is the index of the tiles of the player's hand.
        gym.spaces.Discrete(self.n_tiles*self.grid_length*self.grid_length),  # The player's hand has 6 tiles
        #gym.spaces.MultiBinary(self.n_tiles)
        # gym.spaces.MultiBinary((self.grid_length, self.grid_length))  # The board is grid_length x grid_length
        # Introduce the tile you are placing
        ))

        # For a game like Tic Tac Toe, the observation space might be a 3x3 grid
        # where each cell can be in one of three states (empty, occupied by player 1, or occupied by player 2)
        # self.observation_space = gym.spaces.Box(-1, 1, self.grid_shape+(2,))
        # defines the structure of the observations that the environment provides to the agent.
        # NEW: you need to observe the hand as well
        # NEW: put what the algorithm does as a flowchart. 
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=1, shape=(self.n_tiles, 12), dtype=np.int32),  # Player's hand
            gym.spaces.Box(low=0, high=1, shape=(self.grid_length, self.grid_length, 12), dtype=np.int32)  # Board
        ))     
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

    # NEW: I only need tot store the action and the state, I need to call over the other game.  
    #If there are tiles left in the bag, it randomly selects one, removes it from the bag, and adds it to 
    #the list of drawn tiles. If there are no tiles left in the bag, it breaks out of the loop and returns the tiles that were drawn. 
    #This way, the game can continue even if there are no tiles left to draw.          
    def draw_tiles(self, num_tiles):
        tiles = []
        for i in range(num_tiles):
            if self.bag_of_tiles:
                tile = random.choice(self.bag_of_tiles)
                self.bag_of_tiles.remove(tile)
                tiles.append(tile)
            else:
                break
        return tiles   
        
    def place_tile(self, row, col, color, shape):
        # Reset the cell's state
        self.board[row, col] = np.zeros(12, dtype=np.int32)

        # Set the dimensions corresponding to the tile's color and shape
        self.board[row, col, self.color_to_dimension[color]] = 1
        self.board[row, col, self.shape_to_dimension[shape]] = 1  


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

    # This is different that observation in init. this runs every round but init runs only once.
    @property
    def observation(self):
        # Get the state of the board
        board_state = self.board
        
        # So you get the legal actions based on the board state
        # Compute the legal actions, 
        legal_actions = self.legal_actions

        # Stack the board state and the legal actions along the last dimension
        out = np.stack([board_state, legal_actions], axis=-1)

        return out

    # Legal actions here goes through every cell on the board and check
    # if it is empty and if it is legal to place a tile there.
    @property
    def legal_actions(self):
        legal_actions = np.zeros((self.grid_length, self.grid_length))

        # Iterate over the cells on the board
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                # Check if the cell is empty
                #if np.all(self.board[i, j] == np.eye(12)[0]):  # Assuming the first tile is the "empty" tile
                if self.board[i, j].all(0):
                    # Check if placing a tile in this cell would be a legal action
                    # is_legal_action is a function that you need to write, 
                    # it should return True if the action is legal and False otherwise
                    if self.board[i+1, j].any(1) or self.board[i-1, j].any(1) or self.board[i, j+1].any(1) or self.board[i, j-1].any(1): 
                        legal_actions[i, j] = 1
                    # if self.is_legal_action(i, j):
                    #     legal_actions[i, j] = 1

        return legal_actions
    # Start what you need and try to figure out how to do it. 

    def is_legal_action(self, row, col):

    # check where this is coming from. 
    def square_is_player(self, square, player):
        
        return self.board[square].number == self.players[player].token.number

    # again qwirkle implementation would havve this
    def check_game_over(self):
        # Check if the bag of tiles is empty
        if not self.bag_of_tiles:
            # Check if any player has no tiles left
            for player in self.players:
                if not player.tiles:
                    return True, player
        return False, None

    @property
    # Nothing need to be changed, probably
    def current_player(self):
        return self.players[self.current_player_num]

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
    def step(self, action):
        #once it took an action what would you do with that just say if it is a good idea or not. 
        # you don't need tunrs_taken. 
        # consider normalising the reward. 
        reward = [0,0]

        # Meeting 4: Don't take the location.As it is a bad idea.
        # col = action % self.grid_length
        # row = action // self.grid_length
        # tile = action[1]
        # you need to test the following. 
        tile = action % self.n_tiles
        two_d_index = action // self.n_tiles
        col = two_d_index % self.grid_length
        row = two_d_index // self.grid_length

        # I need to update the state of the game , the board, remove the tile from the hand

        # maximum point can be counted as 432.
        #board = self.board
        # check illegal action
                # Make sure the placement is not on a corner and is inside the board
        if col < 0 or col >= len(self.grid_length):
            done = True
            reward = [1, 1]
            reward[self.current_player_num] = -1
        if row < 0 or row >= len(self.grid_length):
            done = True
            reward = [1, 1]
            reward[self.current_player_num] = -1            
        if col == 0 and row == 0:
            done = True
            reward = [1, 1]
            reward[self.current_player_num] = -1
        if col == 0 and row == len(self.grid_length) - 1:
            done = True
            reward = [1, 1]
            reward[self.current_player_num] = -1
        if col == len(self.grid_length) - 1 and row == len(self.grid_length) - 1:
            done = True
            reward = [1, 1]
            reward[self.current_player_num] = -1
        if col == len(self.grid_length) - 1 and row == 0:
            done = True
            reward = [1, 1]
            reward[self.current_player_num] = -1

        if (self.board[row][col].any(1)):
            done = True
            reward = [1, 1]
            reward[self.current_player_num] = -1
        # check if player can make a move, pass otherwise
        # place all the tiles you can in one step.
            # a while loop to place as many tiles as possible.

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
        
        else:
            self.board[row][col] = tile
            reward[self.current_player_num] = 1
            self.player_hands[self.current_player_num].remove(tile)

            # handle game over
            done = self.check_game_over()
            if not done:
                # Draw a new tile for the player
                self.player_hands[self.current_player_num].append(self.draw_tiles(1))

        self.current_player_num = (self.current_player_num + 1) % 2
        self.turns_taken += 1

        return self.observation, reward, done, {}


    # Dependent on how you define the board and how to reset this.
    def reset(self):
        # self.board = [Token('.', 0)] * self.num_squares
        # self.players = [Player('1', Token('X', 1)), Player('2', Token('O', -1))]

        # Initialize the bag of tiles going through each color
        self.bag_of_tiles = [(colour, shape) for colour in self.colours for shape in self.shapes for i in range(3)]
        print(len(self.bag_of_tiles))
        # Initialize the players' hands
        self.player_hands = [self.draw_tiles(self.n_tiles) for i in range(self.n_players)]

        # Reset the board
        self.board = np.zeros((self.grid_length, self.grid_length, 12), dtype=np.int32)

        # set current player
        self.current_player_num = random.randint(0, 1)
        self.turns_taken = 0
        self.done = False
        #logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation

    # Map how it outputs the game on cml
    def render(self, mode='human', close=False, verbose = True):
        logger.debug('')
        if close:
            return
        if self.done:
            logger.debug(f'GAME OVER')
        else:
            logger.debug(f"It is Player {self.current_player.id}'s turn to move")
            
        logger.debug(' '.join([x.symbol for x in self.board[:self.grid_length]]))
        logger.debug(' '.join([x.symbol for x in self.board[self.grid_length:self.grid_length*2]]))
        logger.debug(' '.join([x.symbol for x in self.board[(self.grid_length*2):(self.grid_length*3)]]))

        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

    # This is to make a move. Given a certain state of the board what are you doing next.
    # Also log to report the what is happening
    def rules_move(self):
        if self.current_player.token.number == 1:
            b = [x.number for x in self.board]
        else:
            b = [-x.number for x in self.board]

        # Check computer win moves
        for i in range(0, self.num_squares):
            if b[i] == 0 and testWinMove(b, 1, i):
                logger.debug('Winning move')
                return self.create_action_probs(i)
        # Check player win moves
        for i in range(0, self.num_squares):
            if b[i] == 0 and testWinMove(b, -1, i):
                logger.debug('Block move')
                return self.create_action_probs(i)
        # Check computer fork opportunities
        for i in range(0, self.num_squares):
            if b[i] == 0 and testForkMove(b, 1, i):
                logger.debug('Create Fork')
                return self.create_action_probs(i)
        # Check player fork opportunities, incl. two forks
        playerForks = 0
        for i in range(0, self.num_squares):
            if b[i] == 0 and testForkMove(b, -1, i):
                playerForks += 1
                tempMove = i
        if playerForks == 1:
            logger.debug('Block One Fork')
            return self.create_action_probs(tempMove)
        elif playerForks == 2:
            for j in [1, 3, 5, 7]:
                if b[j] == 0:
                    logger.debug('Block 2 Forks')
                    return self.create_action_probs(j)
        # Play center
        if b[4] == 0:
            logger.debug('Play Centre')
            return self.create_action_probs(4)
        # Play a corner
        for i in [0, 2, 6, 8]:
            if b[i] == 0:
                logger.debug('Play Corner')
                return self.create_action_probs(i)
        #Play a side
        for i in [1, 3, 5, 7]:
            if b[i] == 0:
                logger.debug('Play Side')
                return self.create_action_probs(i)


    def create_action_probs(self, action):
        action_probs = [0.01] * self.action_space.n
        action_probs[action] = 0.92
        return action_probs   

# helper funciton for isgame over. Check if it is being used from the outside. Just check that if it is NOT called model.py.
def checkWin(b, m):
    return ((b[0] == m and b[1] == m and b[2] == m) or  # H top
            (b[3] == m and b[4] == m and b[5] == m) or  # H mid
            (b[6] == m and b[7] == m and b[8] == m) or  # H bot
            (b[0] == m and b[3] == m and b[6] == m) or  # V left
            (b[1] == m and b[4] == m and b[7] == m) or  # V centre
            (b[2] == m and b[5] == m and b[8] == m) or  # V right
            (b[0] == m and b[4] == m and b[8] == m) or  # LR diag
            (b[2] == m and b[4] == m and b[6] == m))  # RL diag


def checkDraw(b):
    return 0 not in b

# again a helper funciton. probably something similar but the board is different
def getBoardCopy(b):
    # Make a duplicate of the board. When testing moves we don't want to 
    # change the actual board
    dupeBoard = []
    for j in b:
        dupeBoard.append(j)
    return dupeBoard

# test functions, what does it test. need to write your own test.
def testWinMove(b, mark, i):
    # b = the board
    # mark = 0 or X
    # i = the square to check if makes a win 
    bCopy = getBoardCopy(b)
    bCopy[i] = mark
    return checkWin(bCopy, mark)


def testForkMove(b, mark, i):
    # Determines if a move opens up a fork
    bCopy = getBoardCopy(b)
    bCopy[i] = mark
    winningMoves = 0
    for j in range(0, 9):
        if testWinMove(bCopy, mark, j) and bCopy[j] == 0:
            winningMoves += 1
    return winningMoves >= 2