from abc import ABC, abstractmethod
from env.tictactoe.envs import TicTacToeEnv


class Agent(ABC):
    """

    The base class. Each agent has to be able to select actions.

    """

    def __init__(self):
        self.env = TicTacToeEnv()

    @abstractmethod
    def action(self, obs, training=False):
        """
        The trained agent chooses the best available action
        TODO consider if it is possible to make also a training mode, where he chooses any kind of action and uses
        TODO the reward from real game as learning signal

        :param obs: the latest observation of the environment
        :param training: True/False training or performance mode
        :return: <action, observation, reward, done, info>
        """
        pass

        """

class RlAgent(Agent):
    """

    The base class of all agents that are willing to learn something (yet nothing is said if the agents actually
    commit to learning or if they just pretend to do that or whether they are successful in their attempt)

    Note: in order to use such an instance for performance mode (sample best action via action()) the training data
    of a previous run has to be loaded first (see method load_agent)
    """

    def __init__(self):
        super(RlAgent, self).__init__()
        self.episodes = 100000
        self.filename = None

    @abstractmethod
    def run(self):
        """
        A training session is run

        :return:
        """
        pass

    @abstractmethod
    def load_agent(self, filename):
        """
        The training data of a previous run are loaded from the given file.
        Has to be evoked in order to perform best available actions!

        :param filename: name of the file that holds the training data
        """
        pass


    def set_episodes(self, episodes):
        """
        Sets the number of episodes this agent is meant to be trained
        """
        self.episodes = episodes


    def set_filename(self, filename):
        """
        Sets a specific filename for the training data thar are saved after a complete training.
        @:param filename of the training data (only leading name) (i.e. the type can not be specified)
        """
        self.filename = filename
