from src.agents.base.Agent import RlAgent
from env.tictactoe import strategy

from keras.layers import Dense, Flatten
from keras.models import Sequential
from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy


class KerasAgent(RlAgent):
    """

    Agent that uses the Keras-RL Library

    """

    def __init__(self):
        super(KerasAgent, self).__init__()
        self.env.reset()
        self.env.set_opponent(strategy.random_bot)
        self.agent = None

    def run(self):
        model = KerasAgent.create_model()
        policy = EpsGreedyQPolicy()
        sarsa = KerasAgent.create_agent(model=model, policy=policy)
        sarsa.fit(env=self.env, nb_steps=50000, visualize=False, verbose=1)
        sarsa.save_weights('../../../training/keras/keras-sarsa_base_50000.h5f', True)

    def action(self, obs, training=False):
        # if no agent existent (no training run performed and no existing agent loaded) then create vanilla agent
        if self.agent is None:
            model = KerasAgent.create_model()
            policy = EpsGreedyQPolicy()
            self.agent = KerasAgent.create_agent(model=model, policy=policy)

        # activate training mode of agent
        # BEWARE: class member is changed
        if training:
            self.agent.training = True
        else:
            self.agent.training = False

        return self.agent.forward(obs)

    def load_agent(self, filename):
        model = KerasAgent.create_model()
        policy = EpsGreedyQPolicy()
        self.agent = KerasAgent.create_agent(model=model, policy=policy)
        self.agent.load_weights(filename)

    @staticmethod
    def create_model():
        model = Sequential()
        model.add(Flatten(input_shape=(1, 3, 3)))
        # model.add(Dense(16, activation='relu'))
        model.add(Dense(9, activation='relu'))
        # model.add(Dense(6, activation='relu'))
        model.add(Dense(9, activation='softmax'))
        return model

    @staticmethod
    def create_agent(model, policy):
        new_agent = SARSAAgent(model=model, policy=policy, nb_actions=9)
        new_agent.compile('adam', metrics=['mse'])
        return new_agent


if __name__ == '__main__':
    agent = KerasAgent()
    agent.run()
    print(agent.env.observation_space)
    print(agent.env.action_space)
    print("Samples:")
    print(agent.env.observation_space.sample())
    print(agent.env.action_space.sample())
    print("Diagnostic")
    print(agent.env.observation_space.shape)
    print(agent.env.observation_space.shape[0])
    print(agent.env.action_space)
    print("Look at the model")
