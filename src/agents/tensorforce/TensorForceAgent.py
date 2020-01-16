from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner

import tensorforce
import numpy as np


def main():
    environment = Environment.create(
        environment='src.tensorforce.CustomEnvironment.CustomEnvironment',  max_episode_timesteps=10
    )
    #environment = Environment.create(
    #  environment='gym', level='CartPole', max_episode_timesteps=500, visualize=False
    #)
    agent = Agent.create(
        agent='tensorforce', environment=environment, update=64, objective='policy_gradient', reward_estimation=dict(horizon=20),
    )
    runner = Runner(
        agent=agent,
    )

    runner.run(num_episodes=200)
    runner.run(num_episodes=100)
    runner.close()


if __name__ == '__main__':
    print(tensorforce.__version__)
    main()


"""

write an app, that searches all  stores and checks for thing that are in retail, check for bonuses 

--> basically what platforms like triavago do, but with focus on eveyday stuff

that means, it focuses on necessary products of everyday life, food (foremost)
could be extednded for things that are bought vonsitely too (but less frequentl, car oil, clothing, ....)

what app does, it offers possibility to create list of products that are wanted in general, add products, and offer
subtypes of one product, e.g. not only milk, but bio milk, lactose free milk, etc

then create default lists
keep track of what runs out, and then update that this products needs to be bought
(also create auto mechanisms, like milk every 2 days, or usage 12 apples per week)

app then searches for all the products that are needed and checks only lists of supermarktest for 
special offers, it then selects the supermarket (near to you) which offers best prices for all of the
things

also notifies you when specific products are currently much cheaper than else
and it manages your purchase list in a smart way, so that it suggest to buy a reserve of one product if it
is currently very cheap

also add a dashboard, tha keeps track of your "saving succeses" and also include a ethical guide
"you accomplished 80 % biological and environment frienly purhase", suggest to improve this behaviour
if wanted (dont dorce to) and show development
also show statistic of weekly/monthly spendings and (motivate to user by showing how much money
they left on the street by not be smart , show them the equivalence of hat they left pout
"you could have traveled 3 days to barcelona but you are a dumpass!!! xd")

-----------------------------------------------
what is required: so that it works
----------------------------------------------
-) website crawler (move over websites of different vendors (billa, spar, merkur, hofer, etc))
-) such lists and prices and information about special offer need to be available?!?!?!
-) algorithmic component (easy one: get all to minimum price)
   mot that easy: smart adjustments (app needs to know what is the all time average of prices, how
   to make msart suggestion, only save thing that con not spoil
-) server, that crawls, and collects data, calculates, notofoes clients about offers, and adjust buyin
bahiviour
-) client app --- collects personal statistics, spending behaivour, types of what is being bought by
user, display this data



"""