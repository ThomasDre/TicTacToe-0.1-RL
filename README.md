TicTacToe_RL

This project contains (overview, for more details read the section 'Content') 
a) a custom TicTacToe Environment that extends the OpenAI's Gym Environment
b) a Reinforcement Learning Ecosystem for this TicTacToe Game 


**************************************************************************
*                          RUN THIS PROJECT                              *
**************************************************************************

Preliminaries:
1.) run "pip install -r requirements.txt" from the upper most folder
	(installs all dependencies)
2.) go to the folder custom_env and run "pip install -e ."
	(deploys the custom TicTacToe envioronment so that it can be accesses via the Gym Libraries)

Run the actual program:
3.) Simulate Mode: run the agents by running the respective Agent classes
4.) Interactive Mode: start the web environment by running the 'Server' class



**************************************************************************
*                              CONTENT                                   *
**************************************************************************

************************
* - CUSTOM ENVIRONMENT *
************************

custom_env/tictactoe/envs/tictactoe_env.py
	a custom TicTacToe environment is available that implemenets the OpenAI's Gym Env interface

custom_env/tictactoe/strategy.py
	different opponent behaviours for the game of TicTacToe are provided
	(NOT YET IMPLEMENTED: upon starting an agent the prefered opponent can be specified)
	-) 'human': a opponent mode that tells the agent that he plays against a real client and that its actions come in asynchronously 
	-) 'random': a dummy bot that selects random valid moves
	-) 'bot': NOT YET IMPLEMENTED (a couple of different available TicTacToe bots should be made available)


************************
* - Agents and Server  *
************************

src/Server.py
	starting the server will host a localhost webpage on "http://127.0.0.1:5000/" where the game of TicTacToe is made available
	-> you can play against the agent yourself (performance or training mode either)
	-> you can manage the training of the agents (length, opponent for training, save agent)

	TODO: only playing against random agent currently available

src/agents/
./kerasrl
	to be contiued
	contains implementation of agents using different RL libraries

static/styles/mainpage.css
templates/index.html
	HTML and Styles for the webpage that is run and hosted through python's flask lightweight web application framework

testarea/
	this folder exists solely for the purpose of *guess what* testing things that are so vague that they should not be interfering with real code
	* feel free to abuse this folder, the same way I do BUT NOTE that updates to this folder are excluded from commity by default *




