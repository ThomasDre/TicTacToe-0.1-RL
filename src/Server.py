"""
A simple server that connects the reinforcement learning agent with a hosted (e.g. templates) representation of the
game.

A user can play the game in the browser and can
-) provide samples for learning (TRAIN MODE)
-) play a competitive game on the best level the agent is capable of (EVAL MODE)
-) control and observe offline training behaviour of agent (META-TEACHING MODE)

This server provides simple communication, to get user interactions, commands and to replay the own actions to the
interactive game during a game-session
"""

from flask import Flask, request, render_template, make_response

from env.tictactoe.envs.tictactoe_env import TicTacToeEnv
from src.agents.base.RandomAgent import RandomAgent
from env.tictactoe import strategy
from src.utils import util
import src.agents as agents
import src.exceptions as exceptions


app = Flask(__name__)

game_environment = TicTacToeEnv()
agent = RandomAgent()

first_move = True
response_text_template = '{{\"x\": {}, \"y\": {}, \"end\": \"{}\", \"won\": \"{}\"}}'

# FLAG
_DEBUG = False


ai_agents = ["Random Bot", "Keras RL", "Huskarl"]

trainable_agents = ["Keras RL", "Huskarl"]

agents = {
    "Random Bot": agents.SillyAgent(),
    "Keras RL": agents.KerasAgent(),
    "Huskarl": agents.HuskarlAgent()
}


def agent_name_to_agent(agent_name, for_training):
    if for_training and agent_name not in trainable_agents:
        raise exceptions.AgentNotFoundException()
    if not for_training and agent_name not in ai_agents:
        raise exceptions.AgentNotFoundException

    return agents[agent_name]


@app.route('/')
def output():
    # serve index template
    return render_template('index.html')


@app.route('/reset', methods=['GET'])
def reset():
    game_environment.reset()
    global first_move
    first_move = True
    return make_response('OK', 200)


@app.route('/move/user', methods=['POST'])
def get_user_move():
    game_environment.set_opponent(strategy.human)
    global first_move
    first_move = False
    data = request.get_json(force=True)
    user_move = util.map_cell_to_scalar(data['x'], data['y'])
    obs, reward, done, info = game_environment.take_a_move(user_move)

    if done:
        response_text = response_text_template.format(-1, -1, True, True if reward == 1 else 'Undefined')
        if _DEBUG:
            debug(is_human=True, obs=obs, reward=reward, done=done, response=response_text)
    else:
        action, obs, reward, done, info = agent.action(obs)
        agent_move_x, agent_move_y = util.map_scalar_to_cell(action)

        if done:
            response_text = response_text_template.format(agent_move_x, agent_move_y, True, False if reward == 1 else 'Undefined')
        else:
            response_text = response_text_template.format(agent_move_x, agent_move_y, False, 'Undefined')

        if _DEBUG:
            debug(is_human=False, obs=obs, reward=reward, done=done,response=response_text)

    if done:
        game_environment.reset()

    return make_response(response_text, 200)


@app.route('/move/agent', methods=['GET'])
def force_agent_move():
    game_environment.set_opponent(strategy.human)
    global first_move
    if first_move:
        first_move = False
        obs = game_environment.reset()
        action, obs, reward, done, info = agent.action(obs)
        agent_move_x, agent_move_y = util.map_scalar_to_cell(action)

        if _DEBUG:
            debug(False, obs=obs, reward=reward, done=done)

        response_text = response_text_template.format(agent_move_x, agent_move_y, False, 'Undefined')
        return make_response(response_text, 200)

    return make_response('Failure', 400)


@app.route('/train', methods=['POST'])
def train_agent():
    num_of_episodes = request.form['episodes']
    agent_name = request.form['agent']
    custom_file_name = request.form['fileName']
    global agent

    try:
        agent = agent_name_to_agent(agent_name, for_training=True)
    except exceptions.AgentNotFoundException:
        return make_response("No such agent with name of '" + agent_name + "' available", 400)

    agent.set_episodes(int(num_of_episodes))

    if custom_file_name != '':
        agent.set_filename(custom_file_name)

    agent.run()

    return make_response("ok", 200)


@app.route('/load', methods=['POST'])
def load_agent():
    agent_name = request.form['agent']
    custom_data_file_name = request.form['fileName']

    global agent

    try:
        agent = agent_name_to_agent(agent_name, for_training=False)
    except exceptions.AgentNotFoundException:

        return make_response("No such agent with name of '" + agent_name + ' available', 400)

    if custom_data_file_name != '':
        try:
            agent.load_agent(custom_data_file_name)
        except exceptions.AgentDataNotAvailable:
            make_response("Load agent failed! The given file does not contain valid training data", 400)

    return make_response("ok", 200)


if __name__ == '__main__':
    app.run()


def debug(is_human, obs=None, reward=None, done=None, info=None, response=None, additional=None):
    if is_human:
        print("AI move:")
    else:
        print("User move")
    if obs is not None:
        print("Obs: {}", obs)
    if reward is not None and done is not None:
        print("reward: {}   ---- done: {}".format(reward, done))
    if info is not None:
        print("Info: " + info)
    if response is not None:
        print(response)
    if additional is not None:
        print(additional)
