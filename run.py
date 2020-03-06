from Agent import *
from Epsilon import *
from Game import *
from Logger import *
import numpy as np
import matplotlib.pyplot as plt

play = False #play the agent at the end?
TRAINING_STEPS = 10

''' Initialise everything '''
RL = Agent(connect4())
RL.load()
eps = Epsilon(EPSILON_MAX = 1, EPSILON_MIN = 0.1, DECAY = 0.9997)
loss = [0]
logger = Logger(load=True)



for run in range(TRAINING_STEPS):
    game = connect4()   #make new game
    while not game.Done:
        for player in [0,1]:

            # Get action based on epsilon greedy
            move = RL.get_action(game, player, 0.2)
            game.place(move, player)

            win = game.check_win(player)
            draw = game.check_draw()

            # record actions, rewards, and observations
            logger.log(move=move, player=player, win=win, board_before = game.board_before, board_after = game.board, draw=draw)

            # finish game
            if game.Done:
                break
    # make the observations into a form where the agent can learn from them - don't think this works as intended
    logger.convert()
    loss.append(RL.train(logger.x_train, logger.y_train))

    # Save the network weights intermittently
    if np.mod(run, 100) == 0:
        RL.save()

    print(run, '/', TRAINING_STEPS, '  {:.2f}%  loss = {:.2f}'.format(100 * run / TRAINING_STEPS, loss[-1]), end='\r')

# show whether or not it worked, probably not.
plt.plot(loss)
plt.show()

''' Play against the RL agent'''
if play:
    # blank game
    game = connect4()
    while not game.Done:
        for player in [0,1]:
            # show board
            game.print_board()

            if player == 0:
                # RL agents move
                move = RL.get_action(game, player, 0)
            else:
                # Your move
                move = game.human_move(player)
            game.place(move, player)

            # check if the game has been won or drawn
            win = game.check_win(player, do_print=True)
            draw = game.check_draw()

            # Save observations as before
            logger.log(move=move, player=player, win=win, board_before = game.board_before, board_after = game.board, draw=draw)

            # Finish game
            if game.Done:
                break
    # make the observations into a form where the agent can learn from them - don't think this works as intended
    logger.convert()

# save the replay buffer
logger.save()