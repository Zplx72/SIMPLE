import unittest
from qwirkle import QwirkleEnv  

class TestQwirkleEnv(unittest.TestCase):
    def test_env_initialization(self):
        # Create an instance of the environment
        env = QwirkleEnv()

        # Assert that the bag of tiles is initialized correctly
        self.assertEqual(len(env.bag_of_tiles), 96)

        # Assert that the player hands are initialized correctly
        for hand in env.player_hands:
            self.assertEqual(len(hand), 6)

        # Assert that the board is initialized correctly
        self.assertEqual(env.board.shape, (91, 91, 12))

    

if __name__ == '__main__':
    unittest.main()