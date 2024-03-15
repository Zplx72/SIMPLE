import unittest
from qwirkle import QwirkleEnv  

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
        action = 49669  # replace with a specific action you want to test
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
    
if __name__ == '__main__':
    unittest.main()