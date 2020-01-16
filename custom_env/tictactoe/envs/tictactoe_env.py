import gym
from gym import spaces, utils
from gym.error import ResetNeeded
from gym.utils import seeding

import numpy as np
import random


class TicTacToeEnv(gym.Env):
    # unclear what this command does
    metadata = {'render.modes': ['human']}
    reward_range = (-1, 1)

    action_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3)))
    observation_space = spaces.Box(low=0, high=4, shape=(3, 3))

    """
    Global variables
    Have to be consistent for all participants in game
    """
    board = np.zeros((3, 3))
    episodes = 0
    isPlayer1 = True
    newGame = True
    hasEnded = False
    winner = None
    player1_wins = 0
    player2_wins = 0

    isHumanMove = False

    def __init__(self):
        self.player1Sign = 1
        self.player2Sign = 4
        self.gameWonPlayer1 = 3
        self.gameWonPlayer2 = 12
        # strategy holds a callback function (one of the provided in the class strategy, can be set)
        self.strategy = None
        self.info = {'illegal': False, 'human': False}

    """
        -------------------------------------------------------------------------------
        -                                                                             -
        -                                                                             -
        ------------------------- Overwritten / Gym Methods ---------------------------
        -                                                                             -
        -                                                                             -
        -------------------------------------------------------------------------------
        """

    def step(self, action):
        """
        One move on the TicTacToe board is performed

        :type action: tuple (x, y) representing which cell should be marked
        :return <observation, reward, done, info>
        info := {
            'illegal': Boolean,
            'human': Boolean
        }
        """
        row = action[0]
        column = action[1]

        if self.hasEnded:
            raise ResetNeeded()

        # check if cell is already marked
        if self.board[row][column] != 0:
            self.info['illegal'] = True
            return self.board, -1, False, self.info
        else:
            self.info['illegal'] = False

        sign = self.player1Sign if TicTacToeEnv.isPlayer1 else self.player2Sign
        self.board[row][column] = sign
        # switch player
        TicTacToeEnv.isPlayer1 = not TicTacToeEnv.isPlayer1
        TicTacToeEnv.newGame = False

        if self.check_win():
            return self.board, 1, True, self.info
        elif self.check_draw():
            return self.board, 0, True, self.info
        else:
            # game continues: opponent move
            is_human_move, other_won, is_draw = self.strategy(self)

            #  if human move then agent has to ask for result later
            if is_human_move:
                self.info['human'] = True
                TicTacToeEnv.isHumanMove = True
                return self.board, 0, False, self.info
            # if ai move, then result of ais move is determines and returned
            else:
                self.info['human'] = False
                if other_won:
                    return self.board, -1, True, self.info
                elif is_draw:
                    return self.board, 0, True, self.info
                else:
                    return self.board, 0, False, self.info

    def reset(self):
        TicTacToeEnv.board = np.zeros((3, 3))
        TicTacToeEnv.isPlayer1 = True
        TicTacToeEnv.newGame = True
        TicTacToeEnv.hasEnded = False
        TicTacToeEnv.winner = None
        self.info = {'illegal': False, 'human': False}

    def render(self, mode='human'):
        representation = "----------------\n"
        representation += "| {} | {} | {} |\n".format(self.board[0][0], self.board[0][1], self.board[0][2])
        representation += "----------------\n"
        representation += "| {} | {} | {} |\n".format(self.board[1][0], self.board[1][1], self.board[1][2])
        representation += "----------------\n"
        representation += "| {} | {} | {} |\n".format(self.board[2][0], self.board[2][1], self.board[2][2])
        representation += "----------------\n"

        representation = representation.replace(".0", "")
        representation = representation.replace("0", "-")
        representation = representation.replace(str(self.player1Sign), 'X').replace(str(self.player2Sign), 'O')
        print(representation)

        if self.hasEnded:
            print("!!!")
            if TicTacToeEnv.winner is None:
                print(" This game has ended with a draw ")
            elif TicTacToeEnv.winner == self.player1Sign:
                print(" Player 1 has won this game ")
            else:
                print(" Player 2 has won this game ")
            print("!!!\n\n")

            wins1 = TicTacToeEnv.player1_wins
            wins2 = TicTacToeEnv.player2_wins
            print("Player 1: {}\n".format(wins1))
            print("Player 2: {}\n".format(wins2))
            print("Draws: {}\n".format(TicTacToeEnv.episodes - wins1 - wins2))
    """
        -------------------------------------------------------------------------------
        -                                                                             -
        -                                                                             -
        ---------------------- Outside Control Methods  -------------------------------
        -                                                                             -
        -                                                                             -
        -------------------------------------------------------------------------------
        """

    def take_a_move(self, action):
        """
        This method is meant to be used for clients that play against the agent to take their moves.
        Clients (Human-player, bots)

        :param action: tuple(x,y) that represents the cell to be marked
        :return: <observation, reward, done, info>
        """
        row = action[0]
        column = action[1]

        if TicTacToeEnv.hasEnded:
            raise ResetNeeded()

        if self.board[row][column] != 0:
            self.info['illegal'] = True
            return self.board, -1, False, self.info
        else:
            self.info['illegal'] = False

        sign = self.player1Sign if TicTacToeEnv.isPlayer1 else self.player2Sign
        self.board[row][column] = sign
        # switch player
        TicTacToeEnv.isPlayer1 = not TicTacToeEnv.isPlayer1
        TicTacToeEnv.newGame = False

        if self.check_win():
            return self.board, 1, True, self.info
        elif self.check_draw():
            return self.board, 0, True, self.info
        else:
            return self.board, 0, False, self.info

    def set_opponent(self, opponent):
        """
        Specifies whether against whom the agent plays (default bot)

        If bot then after each action taken the result is immediately determines after step() calls.
        If the agent wins he is informed, else the bot move is taken next and the agent is informed if he lost the gane
        (or whether the game continues)

        In human mode:
        after
        :param opponent: callback function that contains the strategy of the opponent (choose from available callback
        methods in strategy.py)
        :return:
        """
        self.strategy = opponent

    @staticmethod
    def get_actions():
        """
        Set of all possible actions that can be made at any time step (not guaranteed that the action is valid in
        respect to current game state)

          0 1 2
        0 A B C
        1 D E F
        2 G H I

        action 'x' defines assigning its own sign ("circle", "cross", respectively encoded as 1 and 4) to corresponding
        slot

        :return: array of possible actions
        """

        return [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]

    @staticmethod
    def get_sampled_action():
        """
        Return an arbitrary action from set of all possible actions (it is not guaranteed that the given action
        is valid in respect to current game state)

        :return: arbitrary action
        """
        index = random.randint(0, 8)
        return TicTacToeEnv.get_actions()[index]
    """
    -------------------------------------------------------------------------------
    -                                                                             -
    -                                                                             -
    -------------------------------- HELPER METHODS -------------------------------
    -                                                                             -
    -                                                                             -
    -------------------------------------------------------------------------------
    """

    def check_draw(self):
        if 0 not in self.board:
            TicTacToeEnv.hasEnded = True
            TicTacToeEnv.episodes += 1
            return True
        else:
            return False

    def check_win(self):
        has_won = False
        winner = None

        diagonal_lr_sum = np.sum(np.diag(self.board))
        diagonal_rl_sum = np.sum(np.diag(np.fliplr(self.board)))

        if diagonal_lr_sum == self.gameWonPlayer1:
            has_won = True
            winner = self.player1Sign
        elif diagonal_lr_sum == self.gameWonPlayer2:
            has_won = True
            winner = self.player2Sign
        if diagonal_rl_sum == self.gameWonPlayer1:
            has_won = True
            winner = self.player1Sign
        elif diagonal_rl_sum == self.gameWonPlayer2:
            has_won = True
            winner = self.player2Sign

        for i in range(3):
            row_sum = np.sum(self.board[i])
            column_sum = np.sum(self.board[:, i])

            if row_sum == self.gameWonPlayer1:
                has_won = True
                winner = self.player1Sign
            elif row_sum == self.gameWonPlayer2:
                has_won = True
                winner = self.player2Sign
            elif column_sum == self.gameWonPlayer1:
                has_won = True
                winner = self.player1Sign
            elif column_sum == self.gameWonPlayer2:
                has_won = True
                winner = self.player2Sign

        if has_won:
            TicTacToeEnv.hasEnded = True
            TicTacToeEnv.episodes += 1

            if winner == self.player1Sign:
                TicTacToeEnv.player1_wins += 1
            else:
                TicTacToeEnv.player2_wins += 1

            TicTacToeEnv.winner = winner

        return has_won

    """
     -------------------------------------------------------------------------------
     -                                                                             -
     -                                                                             -
     ------------------------------ CALLBACK METHODS -------------------------------
     -                                                                             -
     -                                                                             -
     -------------------------------------------------------------------------------"""

