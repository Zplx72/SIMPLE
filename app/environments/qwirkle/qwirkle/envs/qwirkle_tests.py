import unittest
from qwirkle import QwirkleEnv  
import numpy as np
class TestQwirkleEnv(unittest.TestCase):
    def setUp(self):
        self.game = QwirkleEnv()  # initialize your game


    # def test_env_initialization(self):
    #     # Create an instance of the environment
    #     env = QwirkleEnv()

    #     # Assert that the bag of tiles is initialized correctly
    #     self.assertEqual(len(env.bag_of_tiles), 96)

    #     # Assert that the player hands are initialized correctly
    #     for hand in env.player_hands:
    #         self.assertEqual(len(hand), 6)

    #     # Assert that the board is initialized correctly
    #     self.assertEqual(env.board.shape, (91, 91, 12))

    
    def test_action_conversion(self):
        # The last element will be 5th tile index, in the 90th col and the 90th row. 
        # That is why 49686 is a valid one. which is multiplied from 91*91*6.
        action = 40000  # replace with a specific action you want to test
        expected_tile_index = action % self.game.n_tiles
        expected_two_d_index = action // self.game.n_tiles
        expected_col = expected_two_d_index % self.game.grid_length
        expected_row = expected_two_d_index // self.game.grid_length

        # Call the action_to_indices function
        tile_index, col, _row = self.game.action_to_indices(action)
        print(f"Action: {action}")
        print(f"Expected: tile_index: {expected_tile_index}, col: {expected_col}, _row: {expected_row}")
        print(f"Actual: tile_index: {tile_index}, col: {col}, _row: {_row}")
        # Check that the calculated values match the expected values
        self.assertEqual(tile_index, expected_tile_index)
        self.assertEqual(col, expected_col)
        self.assertEqual(_row, expected_row)
    
    def test_is_play_valid(self):
        # assuming a valid action
        valid_action = 40000 # replace with a specific action you want to test
        tile_index = valid_action % self.game.n_tiles
        two_d_index = valid_action // self.game.n_tiles
        col = two_d_index % self.game.grid_length
        _row = two_d_index // self.game.grid_length
        does_flag_work =  self.game.flag_is_board_empty
        self.assertEqual(does_flag_work, True)
        # Call the _is_play_valid method
        # Set print options
        np.set_printoptions(threshold=np.inf)
        # self.game.function_is_board_empty()
        self.game.step(valid_action)

        # Check that the method correctly identifies the play as valid
        self.assertEqual(self.game.flag_is_board_empty, False)

    # def test_is_play_invalid(self):
    #     # assuming an invalid action
    #     invalid_action = 40000  # replace with a specific action you want to test
    #     tile_index = invalid_action % self.game.n_tiles
    #     two_d_index = invalid_action // self.game.n_tiles
    #     col = two_d_index % self.game.grid_length
    #     _row = two_d_index // self.game.grid_length

    #     # Call the _is_play_valid method
    #     is_invalid = self.game._is_play_valid(piece=self.game._tiles[tile_index], x=col, y=_row)

    #     # Check that the method correctly identifies the play as invalid
    #     self.assertEqual(is_invalid, False)
    
    def test_step_invalid_play(self):
        # assuming an invalid action
        invalid_action = 1000  # replace with a specific action you want to test
        initial_tiles = self.game._tiles.copy()
        initial_board = self.game._board.copy()
        initial_score = self.game.score()

        # Call the step method
        observation, reward, done, _ = self.game.step(invalid_action)

        # Check that the board was updated correctly
        tile_index = invalid_action % self.game.n_tiles
        two_d_index = invalid_action // self.game.n_tiles
        col = two_d_index % self.game.grid_length
        _row = two_d_index // self.game.grid_length
        self.assertEqual(self.game._board[_row][col], initial_tiles[tile_index])
        print(f"self.game._board[_row][col] : {self.game._board[_row][col]}")

        # Check that the tile was removed from the bag
        self.assertNotIn(initial_tiles[tile_index], self.game._tiles)


        # Check that the score was updated correctly
        self.assertEqual(reward[self.game.current_player_num], self.game.score() - initial_score)
        print(f"reward: {reward}")

        # Check that the game over status is correct
        self.assertEqual(done, self.game.check_game_over())
if __name__ == '__main__':
    unittest.main()