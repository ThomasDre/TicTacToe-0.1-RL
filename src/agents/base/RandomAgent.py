from src.agents.base.Agent import Agent


class RandomAgent(Agent):
    """

    Random Agent Dummy that DOES NOT belong to the family of reinforcement learning agents.

    His solely purpose may is to be used as a simple reactor in games.
    That is, it represents the most basic bot playing a game by choosing a random but valid action.

    """

    def __init__(self):
        super(RandomAgent, self).__init__()
        self.env.reset()

    def action(self, obs, training=False):
        """
        Takes random but valid action independent of the current state of the environment.

        :param obs: current state of environment
        :param training: training: True/False training or performance mode (can be omitted as agent does not care)

        :return: <action, obs, reward, done, info>
        """
        while True:
            action = self.env.get_sampled_action()
            obs, reward, done, info = self.env.take_a_move(action)

            invalid = info['illegal']

            if not invalid:
                break

        return action, obs, reward, done, info
