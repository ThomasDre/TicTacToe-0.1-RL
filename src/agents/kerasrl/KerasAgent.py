from src.agents.base.Agent import RlAgent
from custom_env.tictactoe import strategy


class KerasAgent(RlAgent):
    """

    Agent that uses the Keras-RL Library

    """

    def __init__(self):
        super(KerasAgent, self).__init__()
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

    def action(self):
        pass


if __name__ == '__main__':
    agent = KerasAgent()
    agent.run()
    print(agent.env.observation_space)
    print(agent.env.action_space)

