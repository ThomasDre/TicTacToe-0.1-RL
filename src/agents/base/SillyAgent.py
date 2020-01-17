from src.agents.base.Agent import RlAgent
from custom_env.tictactoe import strategy


class SillyAgent(RlAgent):
    """

    Dummy Class that can be used as a opponent for human players.


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

    def action(self):
        while True:
            action = self.env.get_sampled_action()
            obs, reward, done, info = self.env.take_a_move(action)

            print(info)
            invalid = info['illegal']

            if not invalid:
                break

        return action, obs, reward, done, info


if __name__ == '__main__':
    agent = SillyAgent()
    agent.run()
