import gym
from gym import spaces
from gym.error import ResetNeeded
from src.utils import util

import numpy as np
import random


class TicTacToeEnv(gym.Env):
    # unclear what this command does
    metadata = {'render.modes': ['human']}
    reward_range = (-1, 1)

    # action_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3)))
    # action_space = spaces.Box(low=0.0, high=2.0, shape=(1, 2), dtype='int')
    # probably bullshit line, i don t know myself what im doing
    action_space = spaces.Discrete(9)
    observation_space = spaces.Box(low=0.0, high=4.0, shape=(3, 3), dtype='int')

    """
    Game Related Variables
    Have to be consistent for all participants in game
    """
    _board = np.zeros((3, 3))
    _episodes = 0
    _isPlayer1 = True
    _newGame = True
    _hasEnded = False
    _winner = None
    _player1_wins = 0
    _player2_wins = 0
    _isHumanMove = False

    def __init__(self):
        self.player1Sign = 1
        self.player2Sign = 4
        self.gameWonPlayer1 = 3
        self.gameWonPlayer2 = 12
        # strategy holds a callback function (one of the provided in the class strategy, can be set)
        self.opp_strategy = None
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

        :type action [0;8] where this corresponds to the cell that should be marked by this move
        :return <observation, reward, done, info>
        info := {
            'illegal': Boolean,
            'human': Boolean
        }
        """
        row, column = util.map_scalar_to_cell(action)

        if TicTacToeEnv._hasEnded:
            raise ResetNeeded()

        # check if cell is already marked
        if TicTacToeEnv._board[row][column] != 0:
            self.info['illegal'] = True
            return TicTacToeEnv._board, -1, False, self.info
        else:
            self.info['illegal'] = False

        sign = self.player1Sign if TicTacToeEnv._isPlayer1 else self.player2Sign
        TicTacToeEnv._board[row][column] = sign
        # switch player
        TicTacToeEnv._isPlayer1 = not TicTacToeEnv._isPlayer1
        TicTacToeEnv._newGame = False

        if self.__check_win():
            return TicTacToeEnv._board, 1, True, self.info
        elif self.__check_draw():
            return TicTacToeEnv._board, 0, True, self.info
        else:
            # game continues: opponent move
            is_human_move, other_won, is_draw = self.opp_strategy(self)

            #  if human move then agent has to ask for result later
            if is_human_move:
                self.info['human'] = True
                TicTacToeEnv._isHumanMove = True
                return TicTacToeEnv._board, 0, False, self.info
            # if ai move, then result of ais move is determines and returned
            else:
                self.info['human'] = False
                if other_won:
                    return TicTacToeEnv._board, -1, True, self.info
                elif is_draw:
                    return TicTacToeEnv._board, 0, True, self.info
                else:
                    return TicTacToeEnv._board, 0, False, self.info

    def reset(self):
        TicTacToeEnv._board = np.zeros((3, 3))
        TicTacToeEnv._isPlayer1 = True
        TicTacToeEnv._newGame = True
        TicTacToeEnv._hasEnded = False
        TicTacToeEnv._winner = None
        self.info = {'illegal': False, 'human': False}
        return TicTacToeEnv._board

    def render(self, mode='human'):
        representation = "----------------\n"
        representation += "| {} | {} | {} |\n".format(TicTacToeEnv._board[0][0], TicTacToeEnv._board[0][1], TicTacToeEnv._board[0][2])
        representation += "----------------\n"
        representation += "| {} | {} | {} |\n".format(TicTacToeEnv._board[1][0], TicTacToeEnv._board[1][1], TicTacToeEnv._board[1][2])
        representation += "----------------\n"
        representation += "| {} | {} | {} |\n".format(TicTacToeEnv._board[2][0], TicTacToeEnv._board[2][1], TicTacToeEnv._board[2][2])
        representation += "----------------\n"

        representation = representation.replace(".0", "")
        representation = representation.replace("0", "-")
        representation = representation.replace(str(self.player1Sign), 'X').replace(str(self.player2Sign), 'O')
        print(representation)

        if TicTacToeEnv._hasEnded:
            print("!!!")
            if TicTacToeEnv._winner is None:
                print(" This game has ended with a draw ")
            elif TicTacToeEnv._winner == self.player1Sign:
                print(" Player 1 has won this game ")
            else:
                print(" Player 2 has won this game ")
            print("!!!\n\n")

            wins1 = TicTacToeEnv._player1_wins
            wins2 = TicTacToeEnv._player2_wins
            print("Player 1: {}\n".format(wins1))
            print("Player 2: {}\n".format(wins2))
            print("Draws: {}\n".format(TicTacToeEnv._episodes - wins1 - wins2))
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

        :type action [0;8] where this corresponds to the cell that should be marked by this move
        :return: <observation, reward, done, info>
        """
        row, column = util.map_scalar_to_cell(action)

        if TicTacToeEnv._hasEnded:
            raise ResetNeeded()

        if TicTacToeEnv._board[row][column] != 0:
            self.info['illegal'] = True
            return TicTacToeEnv._board, -1, False, self.info
        else:
            self.info['illegal'] = False

        sign = self.player1Sign if TicTacToeEnv._isPlayer1 else self.player2Sign
        TicTacToeEnv._board[row][column] = sign
        # switch player
        TicTacToeEnv._isPlayer1 = not TicTacToeEnv._isPlayer1
        TicTacToeEnv._newGame = False

        if self.__check_win():
            return TicTacToeEnv._board, 1, True, self.info
        elif self.__check_draw():
            return TicTacToeEnv._board, 0, True, self.info
        else:
            return TicTacToeEnv._board, 0, False, self.info

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
        self.opp_strategy = opponent

    @staticmethod
    def get_actions():
        """
        Set of all possible actions that can be made at any time step (not guaranteed that the action is valid in
        respect to current game state)

          0 1 2
        0 0 1 2
        1 3 4 5
        2 6 7 8

        :return: array of possible actions
        """

        return [0, 1, 2, 3, 4, 5, 6, 7, 8]

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

    @staticmethod
    def __check_draw():
        if 0 not in TicTacToeEnv._board:
            TicTacToeEnv._hasEnded = True
            TicTacToeEnv._episodes += 1
            return True
        else:
            return False

    def __check_win(self):
        has_won = False
        winner = None

        diagonal_lr_sum = np.sum(np.diag(TicTacToeEnv._board))
        diagonal_rl_sum = np.sum(np.diag(np.fliplr(TicTacToeEnv._board)))

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
            row_sum = np.sum(TicTacToeEnv._board[i])
            column_sum = np.sum(TicTacToeEnv._board[:, i])

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
            TicTacToeEnv._hasEnded = True
            TicTacToeEnv._episodes += 1

            if winner == self.player1Sign:
                TicTacToeEnv._player1_wins += 1
            else:
                TicTacToeEnv._player2_wins += 1

            TicTacToeEnv._winner = winner

        return has_won
