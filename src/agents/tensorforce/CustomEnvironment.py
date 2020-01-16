from tensorforce.environments import Environment
from src.TicTacToe import GameEnvironment


class CustomEnvironment(Environment):

    def __init__(self):
        super().__init__()

        self.game = GameEnvironment()
        self.available_actions = tuple(self.game.get_actions())

    def states(self):
        return dict(type='int', shape=(3, 3), num_states=729)

    def actions(self):
        return dict(type='int', shape=(2,), num_values=9, num_actions=9)

    def close(self):
        self.game = None

    def reset(self):
        self.game.reset()
        return self.game.get_state()

    def execute(self, actions):
        observation, has_ended, reward, was_invalid = self.game.make_move(is_agent=True, move=self.available_actions[actions])
        return observation, has_ended, reward

    def get_states(self):
        return self.game.get_state()
