from random import Random
import gym
import numpy as np
import random
from termcolor import colored


from stable_baselines import logger

class COLORS:
    RED = 'red'
    YELLOW = 'yellow'
    GREEN = 'green'
    CYAN = 'cyan'
    MAGENTA = 'magenta'
    BLUE = 'blue'


class SHAPES:
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
    
# class Player():
#     def __init__(self, id, _tiles):
#         self.id = id
#         self.score = 0
#         self._tiles = _tiles
#         print(f"From player, self._tiles: {self._tiles}")
#         # self.pick_tiles(self._bag_of_tiles)       
#         self._tiles = _tiles
#         print(f"From player, self._tiles: {self._tiles}")
#         # self.pick_tiles(self._bag_of_tiles)       
   

class QwirkleEnv(gym.Env):
    def __init__(self, verbose = False, manual = False):
        logger.debug("__init__: begining")
        super(QwirkleEnv, self).__init__()
        self.name = 'qwirkle'
        self.manual = manual
        self.n_tiles = 6
        self.n_players = 2
        self.flag_board_zero_check = True

        self.current_player_num = 0
        # self.done = False
        self.counter = 0

        # Initialize the bag of tiles going through each color
        # Step 2 of conversion, _bag_of_tiles, has been implemented and changed
        self._bag_of_tiles = []
        self._tiles = []
        self._generate_new_bag_of_tiles()

        # Instansiate players
        # self.players = [Player('1', self.pick_tiles_player_specific(self._bag_of_tiles)), Player('2', self.pick_tiles_player_specific(self._bag_of_tiles))]
        # self.current_player = self.player1

        # self._tiles = self.current_player._tiles
        # self.pick_tiles(self._bag_of_tiles)
        # self.current_player._tiles = self._tiles
        # Instansiate players
        # self.players = [Player('1', self.pick_tiles_player_specific(self._bag_of_tiles)), Player('2', self.pick_tiles_player_specific(self._bag_of_tiles))]
        # self.current_player = self.player1
        

        # self.pick_tiles(self._bag_of_tiles)
        # self.current_player._tiles = self._tiles
        # print(len(self.bag_of_tiles))
        # This board is purly numeric.
        self.board = np.zeros((self.grid_length, self.grid_length), dtype=np.float32)

        # Initialize the players' hands
        # self.player_hands = [self.draw_tiles(self.n_tiles) for i in range(self.n_players)]
        # Initialize the board
        # 12 has been gotten rid of as, the only thing on the board would be an integer mapped from the tile to the class. 


        # This board is aligned with what the other source code have. 
        self._board = [[None] * self.grid_length for i in range(self.grid_length)]

        # have this flag just to change it to false later on after the first tile is put down
        self.flag_is_board_empty = True

        # It is like a history of plays, in the game this can happen more than once in one round. 
        self._plays = []       

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
        # [blue ●, magenta ●, red ★, green ★, magenta ●, magenta ❈]
        self._tile = [self._bag_of_tiles.pop(Piece(color=colors[0], shape=shapes[0])), self._bag_of_tiles.pop(Piece(color=colors[3], shape= shapes[0])), self._bag_of_tiles.pop(Piece(color=colors[4], shape= shapes[4])), self._bag_of_tiles.pop(Piece(color=colors[2], shape= shapes[4])), self._bag_of_tiles.pop(Piece(color=colors[3], shape= shapes[0])), self._bag_of_tiles.pop(Piece(color=colors[3], shape= shapes[2]))]

    # step 3 of conversion: add the pick tiles function
    def pick_tiles_player_specific(self, bag_of_tiles):
        rnd = Random()
        _tiles = []
        while len(_tiles) < 6 and len(bag_of_tiles) > 0:
            i = rnd.randint(0, len(bag_of_tiles) - 1)

            _tiles.append(bag_of_tiles.pop(i))    
        return _tiles    

    def pick_tiles(self, bag_of_tiles):
        rnd = Random()
        while len(self._tiles) < 6 and len(bag_of_tiles) > 0:
            i = rnd.randint(0, len(bag_of_tiles) - 1)

            self._tiles.append(bag_of_tiles.pop(i))    
    
    def swap_tiles(self):
        auxilary_bag = self._tiles.copy()
        self._tiles.clear()

        self.pick_tiles(self._bag_of_tiles)
        
        self._bag_of_tiles.extend(auxilary_bag)
        self.flag_board_zero_check = False
        # print("\nTILES ARE SWAPPED!\n")
        logger.debug("\nTILES ARE SWAPPED!\n")
        return

    def legal_actions(self):

        if self.flag_is_board_empty:
            # print(f"The board is empty now? {np.all(self.board == float(0))}")
            self.function_is_board_empty()
            first_move_legal_actions = np.ones((self.grid_length*self.grid_length*self.n_tiles + 1), np.float32)
            first_move_legal_actions[-1] = 0
            return first_move_legal_actions

        # 
        n_combinations = self.grid_length*self.grid_length*self.n_tiles
        while True:
            legal_actions = []
            checked_actions = set()
            # Go through all the possible actions. 
            for i in range(0, self.grid_length*self.grid_length*self.n_tiles):
                
                # Decode the aciton and check wetheer the action is valid.
                tile_index, col, _row = self.action_to_indices(i)
                checked_actions.add((tile_index, col, _row))
                logger.log(f"Checking position ({_row}, {col}) for shape: {self._tiles[tile_index]}")
                # print(f"From the legal actions function: self._tiles {self._tiles}")
                bool_valid_play = self._is_play_valid(piece=self._tiles[tile_index], x = col, y = _row)

                # if the action was valid then turn the index of the legal_actions to 1 indicating that the coresponding aciton is indeed valid. 
                if bool_valid_play:
                    legal_actions.append(1)
                else:
                    legal_actions.append(0)
            
            # Regardless of whether it will be all zeros or not I will need to append a zero at the end.
            # Just for the dimensions to make sense.
            legal_actions.append(0)

            if np.all(np.array(legal_actions) == 0) and self.flag_board_zero_check:
                # print(f"\nself._tiles before swapping: {self._tiles}")
                self.print_tiles(self._tiles)
                self.swap_tiles()
                # print(f"self._tiles after swapping: {self._tiles}")
                self.print_tiles(self._tiles)
                print(f"\nThis is the if part of the while loop: flag_board_zero_check: {self.flag_board_zero_check} and is all legal_actions zero ? {np.all(np.array(legal_actions) == 0)}\n")
            else:
                print(f"\nThis is the else part of the while loop: flag_board_zero_check: {self.flag_board_zero_check} and is all legal_actions zero ? {np.all(np.array(legal_actions) == 0)}\n")
                # self.flag_board_zero_check = True
                break            
            
            # This if statment checks if the tile has already been swapped and still there is no legal_actions then it will return a numpy array that only the last element is a legal action for it.
            # This will help 
        print(f"All actions checked? {len(checked_actions) == n_combinations}")
        if (self.flag_board_zero_check == False) and  np.all(np.array(legal_actions) == 0):
            only_skipping_is_legal = np.zeros((self.grid_length*self.grid_length*self.n_tiles + 1), np.float32)
            only_skipping_is_legal[self.grid_length*self.grid_length*self.n_tiles] = 1
            self.flag_board_zero_check = True
            print(f"only_skipping_is_legal the length: {len(only_skipping_is_legal)}")
            print(f"only_skipping_is_legal last elements:  {only_skipping_is_legal[-1]}\n")
            return only_skipping_is_legal


        print(f"Is legal_actions all zeros? {np.all(np.array(legal_actions) == 0)}")
        # print(legal_actions)
        return np.array(legal_actions)


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
        logger.log("passes position check")
        # Make sure the placement is not already taken
        if self._board[y][x] is not None:
            return False
        logger.log("is not occupied")
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

        logger.log("Has adjacent tile")
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

        # print(F"after play for {piece}")
        # Don't test for piece shape/color if no piece provided
        if piece is None:
            return True

        # print(f"before rows for {piece}")
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
        logger.log(f"{row}, passed row check")
        # print(f"after rows for {piece}")
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

        # print(f"before columns {piece}")
        # print(f"{row}, ({x}, {y})")
        if not self._is_row_valid(row):
            return False
        
        logger.log(f"passed column check {row}")
        # print(f"after columns for {piece}")
        return True

    def _is_row_valid(self, row):
        """If all row colors are equal, check each shape shows up at most once.
           If all shapes are equal, check each color shows up at most once.
           Otherwise the row is invalid."""

        if len(row) == 1:
            return True
        # print(f"before same colour check for {row[0]}")
        if all(row[i].color == row[0].color for i in range(len(row))):
            shapes = []
            for i in range(len(row)):
                if row[i].shape in shapes:
                    return False
                shapes.append(row[i].shape)
            
            # print(f"after same colour check for {row[0]}")
            
        elif all(row[i].shape == row[0].shape for i in range(len(row))):
            print(F"before same shape for {row[0]}")
            colors = []
            for i in range(len(row)):
                if row[i].color in colors:
                    return False
                colors.append(row[i].color)
            # print(f"after same shape for {row[0]}")
        else:
            # print(f"Else: {row}")
            return False

        return True
   
    
