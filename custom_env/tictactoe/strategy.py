from custom_env.tictactoe.envs.tictactoe_env import TicTacToeEnv


"""
How to use this class:
All different kind of opponents are implemented in this file.
Simply add an additional method performing the particular behaviour.

These methods can be selected as needed as the opponent against the agent should play.
Upon interacting with the environment 'env.set_opponent(strategy)' has to be called,
where strategy is a function callback representing one of the methods in this class

I.e. env.set_opponent(human)  (beware not to call human as a function -> i.e NOT "human()")



Implementation Guide: strategy_name(environment: TicTacToeEnv)

- each method performs the strategy on its own (additional classes and library supports are of course possible)
- upon interacting with the environment (i.e. send action, get available actions etc) use the callback handler
  which has to be given as a parameter
- return is <isHuman, hasWon, isDraw>
"""


def human(environment: TicTacToeEnv):
    """
    Tells the environment that a human player takes the next move. (I.e. No move is taken at this point)
    :param environment: callback handler to environment
    :return: <isHuman, hasWon, isDraw> (True, None, None)

    None specifies that nothing can be said at this time about these properties
    """
    return True, None, None


def random_bot(environment: TicTacToeEnv):
    """
    Simulates the move by a random bot. (Chooses any valid action)
    :param environment: callback handler to environment
    :return: <isHuman, hasWon, isDraw> (False, ?, ?)
    """
    while True:
        action = environment.get_sampled_action()
        _, reward, done, info = environment.take_a_move(action)

        invalid = info["illegal"]

        if not invalid:
            break

    if reward == 1:
        return False, True, False
    elif done:
        return False, False, True
    else:
        return False, False, False
