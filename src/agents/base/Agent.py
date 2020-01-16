from abc import ABC, abstractmethod
from custom_env.tictactoe.envs import TicTacToeEnv


class Agent(ABC):

    def __init__(self):
        self.env = TicTacToeEnv()

    @abstractmethod
    def action(self):
        pass


class RlAgent(Agent):

    def __init__(self):
        super(RlAgent, self).__init__()

    @abstractmethod
    def run(self):
        pass
