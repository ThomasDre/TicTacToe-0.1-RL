"""

Test area means test area, nothing here is intended to make sense!

"""

import gym
from custom_env import tictactoe

env = gym.make('tictactoe-v0')

for i in range(10):
    env.step((1,2))

    if i % 2 == 0:
        env.reset()

test = {}
print(test)
test['att1'] = 'Nice'
print(test)

x = True
y = not x
z = not y

print(x)
print(y)
print(z)
