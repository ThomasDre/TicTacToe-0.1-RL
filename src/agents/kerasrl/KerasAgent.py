from src.agents.base.Agent import RlAgent
from custom_env.tictactoe import strategy

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

    def run(self):
        model = self.agent()  # agent(3, 2)
        policy = EpsGreedyQPolicy()
        sarsa = SARSAAgent(model=model, policy=policy, nb_actions=9)
        sarsa.compile('adam', metrics=['mse'])
        sarsa.fit(env=self.env, nb_steps=1000, visualize=False, verbose=1)
        sarsa.save_weights('../../../training/keras/keras-sarsa_base_50000.h5f', True)

    def action(self):
        pass

    @staticmethod
    def agent():
        """

        Consider that this class is currently not completed.
        For any state, action representation coming in, the best available response is picked from the trained
        network.


        :param states: the number of states the environment can be in
        :param actions: the number of available action at each timestep
        :return:
        """
        model = Sequential()
        model.add(Flatten(input_shape=(1, 3, 3)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(9, activation='softmax', input_shape=(1, 3, 3)))

        return model


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



