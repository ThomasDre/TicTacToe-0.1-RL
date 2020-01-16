import numpy
import random


class GameEnvironment:
    """
        empty slots of board are represented with '0'
        PC (playable character: simulated or real player) mark moves with '1'
        AI mark move with '4'
        if a row, column, or diagonal sums up to 3 then player 1 has won, if it sums up to 12 Agent has won
    """

    # CONSTANTS
    PC_SIGN = 1
    PC_WIN = 3
    AGENT_SIGN = 4
    AGENT_WIN = 12

    DRAW_REWARD = 0
    WIN_REWARD = 1
    LOSS_REWARD = -1
    ILLEGAL_MOVE_PUNISHMENT = -1

    # module variables
    game_board = numpy.zeros((3, 3))
    episodes = 0
    draws = 0
    agents_wins = 0
    pc_wins = 0
    current_winner = None
    episode_terminated = False

    @staticmethod
    def reset():
        GameEnvironment.game_board = numpy.zeros((3, 3))
        GameEnvironment.current_winner = None
        GameEnvironment.episode_terminated = False

    @staticmethod
    def get_final_reward():
        if GameEnvironment.current_winner == GameEnvironment.AGENT_WIN:
            return GameEnvironment.WIN_REWARD
        elif GameEnvironment.current_winner == GameEnvironment.PC_WIN:
            return GameEnvironment.LOSS_REWARD
        # should a draw be really awarded a 0?
        else:
            return GameEnvironment.DRAW_REWARD

    def make_move(self, is_agent, move):
        """
        Fills a specific cell of the game board with the according sign (1: Agent; 2: PC) and checks validity of move
        and game status thereafter

        :param is_agent: move by agent or opponent
        :param row: selected row
        :param column: selected column
        :return: duple of <'observation','HasGameEnded','Reward', 'InvalidMove'>
        """
        row = move[0]
        column = move[1]

        if self.episode_terminated:
            raise Exception('You have to reset the game environment before continuing with a next move')

        # cell already marked
        if GameEnvironment.game_board[row, column] != 0:
            return GameEnvironment.game_board, False, GameEnvironment.ILLEGAL_MOVE_PUNISHMENT, True

        if is_agent:
            GameEnvironment.game_board[row, column] = GameEnvironment.AGENT_SIGN
        else:
            GameEnvironment.game_board[row, column] = GameEnvironment.PC_SIGN

        return GameEnvironment.check_game_state(GameEnvironment.game_board)

    @staticmethod
    def check_game_state(game_board):
        diagonal_lr_sum = numpy.sum(numpy.diag(game_board))
        diagonal_rl_sum = numpy.sum(numpy.diag(numpy.fliplr(game_board)))

        if diagonal_lr_sum == GameEnvironment.AGENT_WIN:
            GameEnvironment.current_winner = GameEnvironment.AGENT_WIN
        elif diagonal_lr_sum == GameEnvironment.PC_WIN:
            GameEnvironment.current_winner = GameEnvironment.PC_WIN
        if diagonal_rl_sum == GameEnvironment.AGENT_WIN:
            GameEnvironment.current_winner = GameEnvironment.AGENT_WIN
        elif diagonal_rl_sum == GameEnvironment.PC_WIN:
            GameEnvironment.current_winner = GameEnvironment.PC_WIN

        for i in range(3):
            row_sum = numpy.sum(game_board[i])
            column_sum = numpy.sum(game_board[:, i])

            if row_sum == GameEnvironment.AGENT_WIN:
                GameEnvironment.current_winner = GameEnvironment.AGENT_WIN
            elif column_sum == GameEnvironment.AGENT_WIN:
                GameEnvironment.current_winner = GameEnvironment.AGENT_WIN
            elif row_sum == GameEnvironment.PC_WIN:
                GameEnvironment.current_winner = GameEnvironment.PC_WIN
            elif column_sum == GameEnvironment.PC_WIN:
                GameEnvironment.current_winner = GameEnvironment.PC_WIN

        # someone has one
        if GameEnvironment.current_winner is not None:
            reward = GameEnvironment.WIN_REWARD if GameEnvironment.current_winner == GameEnvironment.AGENT_WIN else GameEnvironment.LOSS_REWARD
            GameEnvironment.terminate_episode()
            return GameEnvironment.game_board, True, reward, False
        # no winner, but all moves were made
        elif 0 not in GameEnvironment.game_board:
            GameEnvironment.terminate_episode()
            return GameEnvironment.game_board,True, GameEnvironment.DRAW_REWARD, False
        # no winner, still moves to be made
        else:
            return GameEnvironment.game_board, False, 0, False

    @staticmethod
    def terminate_episode():
        GameEnvironment.episode_terminated = True
        GameEnvironment.episodes += 1
        if GameEnvironment.current_winner == GameEnvironment.AGENT_WIN:
            GameEnvironment.agents_wins += 1
        elif GameEnvironment.current_winner == GameEnvironment.PC_WIN:
            GameEnvironment.pc_wins += 1
        elif GameEnvironment.current_winner is None:
            GameEnvironment.draws += 1

    @staticmethod
    def has_ended():
        return GameEnvironment.episode_terminated

    @staticmethod
    def get_state():
        return GameEnvironment.game_board

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
        return GameEnvironment.get_actions()[index]
