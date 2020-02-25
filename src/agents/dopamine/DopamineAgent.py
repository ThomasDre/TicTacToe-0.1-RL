from src.agents.base.Agent import RlAgent
from env.tictactoe import strategy
# from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import run_experiment


class DopamineAgent(RlAgent):

    def __init__(self):
        super(DopamineAgent, self).__init__()
        self.env.reset()
        self.env.set_opponent(strategy.random_bot)
        # self.agent = dqn_agent.DQNAgent()

    def run(self):
        # experiment = run_experiment.TrainRunner(create_agent_fn=self.create_agent, create_environment_fn=self.create_env)
        # experiment.run_experiment()
        pass

    def load_agent(self, filename):
        pass

    def action(self, obs, training=False):
        pass

    def create_agent(self):
        return self.agent

    def create_env(self):
        return self.env


if __name__ == '__main__':
    agent = DopamineAgent()
    agent.run()
