from random import Random
import gym
import numpy as np
import random
from termcolor import colored


# from stable_baselines import logger

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
        # logger.debug("__init__: begining")
        super(QwirkleEnv, self).__init__()

        self.name = 'qwirkle'
        self.manual = manual
        self.n_tiles = 6
        self.n_players = 2
        self.flag_board_zero_check = True

        self.current_player_num = 0
        # self.done = False
        self.counter = 0

        self.look_up_float_to_piece = {}
        self.look_up_dict = {}

        self.shapes = [SHAPES.CIRCLE, SHAPES.DIAMOND, SHAPES.SPARKLE, SHAPES.SQUARE, SHAPES.STAR, SHAPES.TRIANGLE]
        self.colors = [COLORS.BLUE, COLORS.CYAN, COLORS.GREEN, COLORS.MAGENTA, COLORS.RED, COLORS.YELLOW]

        # Initialize the bag of tiles going through each color
        # Step 2 of conversion, _bag_of_tiles, has been implemented and changed
        self._bag_of_tiles = []
        self._tiles = []
        self._generate_new_bag_of_tiles()

        self.grid_length = 30
        self.num_squares = self.grid_length * self.grid_length
        self.grid_shape = (self.grid_length, self.grid_length)

        # The boards
        self.board = np.zeros((self.grid_length, self.grid_length), dtype=np.float32)
        self._board = [[None] * self.grid_length for i in range(self.grid_length)]
        
        # It is (x,y) i.e. (column, row)
        self._cooridnates = [(16, 6), (16, 7), (14, 8), (15, 8), (16, 8), (17, 8), (18, 8), (19, 8), (16, 9), (16, 10), (16, 11)]
        self._values = [-0.45, -0.55, -0.45, -0.5, -0.4, -0.35, -0.6, -0.55, -0.5, -0.35, -0.6]
        
        for value, coordinate in enumerate(self._cooridnates):
            # self.board[y][x], 
            self.board[coordinate[1]][coordinate[0]] = self._values[value]
            self._board[coordinate[1]][coordinate[0]] = self.float_to_piece_converter(float(self._values[value]))


        
        print(self._board)
        self._hand_pick_specific_tiles()

        # have this flag just to change it to false later on after the first tile is put down
        self.flag_is_board_empty = True

        # It is like a history of plays, in the game this can happen more than once in one round. 
        self._plays = []       

    def _hand_pick_specific_tiles(self):
        self._tiles = [Piece(color=self.colors[0], shape=self.shapes[0]), Piece(color=self.colors[3], shape= self.shapes[0]), Piece(color=self.colors[4], shape= self.shapes[4]), Piece(color=self.colors[2], shape= self.shapes[4]), Piece(color=self.colors[3], shape= self.shapes[0]), Piece(color=self.colors[3], shape= self.shapes[2])]
        for tile in self._tiles:
            for i, bag_tile in enumerate(self._bag_of_tiles):
                if bag_tile.color == tile.color and bag_tile.shape == tile.shape:
                    self._bag_of_tiles.pop(i)
                    break
        print(f"self._tiles: {self._tiles}")  
        print(f"len of _bag_of_tiles {len(self._bag_of_tiles)}")      

    def _generate_new_bag_of_tiles(self):
        self._bag_of_tiles = []
        for i in range(3):
            for c in range(len(self.colors)):
                for s in range(len(self.shapes)):
                        self._bag_of_tiles.append(Piece(color=self.colors[c], shape=self.shapes[s]))    
        # [blue ●, magenta ●, red ★, green ★, magenta ●, magenta ❈]

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

    def piece_to_float_converter(self, piece):
        piece = str(piece)
        if len(self.look_up_dict) == 36:
            piece_in_float = self.look_up_dict[piece]
            print(f"self.look_up_dict: {self.look_up_dict}")
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
            piece_in_float = self.look_up_dict[str(piece)]
            print(f"self.look_up_dict: {self.look_up_dict}")
            return piece_in_float
 

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
        # logger.debug("\nTILES ARE SWAPPED!\n")
        return
    
    def function_is_board_empty(self):
        if np.all(self.board == float(0)):
            self.flag_is_board_empty = True
            # print(f"The 'if' part of the board, flag is: {self.flag_is_board_empty}")            
        else:
            self.flag_is_board_empty = False
            # print(f"The 'else' part of the board, flag is: {self.flag_is_board_empty}")  
        return self.flag_is_board_empty
    
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
                # logger.log(f"Checking position ({_row}, {col}) for shape: {self._tiles[tile_index]}")
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
        # logger.log("passes position check")
        # Make sure the placement is not already taken
        if self._board[y][x] is not None:
            return False
        # logger.log("is not occupied")
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

        # logger.log("Has adjacent tile")
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
        # logger.log(f"{row}, passed row check")
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
        
        # logger.log(f"passed column check {row}")
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


def test():
    qwirkle = QwirkleEnv()
    legal_actions = qwirkle.legal_actions()
    return legal_actions
    
test()