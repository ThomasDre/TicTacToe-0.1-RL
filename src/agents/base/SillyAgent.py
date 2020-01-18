from src.agents.base.Agent import RlAgent
from env.tictactoe import strategy


class SillyAgent(RlAgent):
    """

    Dummy Class that can be used solely as a placeholder for an RL-Agent.

    That is, it offers all methods, yet DOES NOT perform any actual training
    ==> DO NOT EXPECT NON RANDOM GAMEPLAY

    Why to use:
    If the interaction between environment, game-controller and agent is meant to be tested, you might want to this
    debug run with a class that does not perform time-consuming training.

    """

    def __init__(self):
        super(SillyAgent, self).__init__()
        self.env.reset()
        self.env.set_opponent(strategy.random_bot)

    def run(self):
        for episode in range(100):
            self.env.reset()
            done = False

            while not done:
                # silly agent chooses random actions until he chooses a valid one, and then he hopes for the best
                while True:
                    _, _, done, info = self.env.step(self.env.get_sampled_action())
                    invalid = info['illegal']

                    if not invalid:
                        break

                self.env.render()

        self.env.close()

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

    def load_agent(self, filename):
        # silly agents have no training data
        pass


if __name__ == '__main__':
    agent = SillyAgent()
    agent.run()
