from gym.envs.registration import register

register(
    id='tictactoe-v0',
    entry_point='custom_env.tictactoe.envs:TicTacToeEnv',
)
