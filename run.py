from Agent import *
from Epsilon import *
from Game import *
from Logger import *
import numpy as np
import matplotlib.pyplot as plt


play = True
TRAINING_STEPS = 25000
RL = Agent(connect4())
RL.load()
eps = Epsilon(EPSILON_MAX = 1, EPSILON_MIN = 0.1, DECAY = 0.9997)
loss = [0]
logger = Logger()
for run in range(TRAINING_STEPS):
    print(run, '/', TRAINING_STEPS, '  {:.2f}%  loss = {:.2f}'.format(100 * run / TRAINING_STEPS, loss[-1]), end='\r')
    game = connect4()
    while not game.Done:
        for player in [0,1]:
            move = RL.get_action(game, player, 0.2)

            game.place(move, player)

            win = game.check_win(player)
            draw = game.check_draw()

            logger.log(move=move, player=player, win=win, board_before = game.board_before, board_after = game.board, draw=draw)

            if game.Done:
                break
    logger.convert()
    loss.append(RL.train(logger.x_train, logger.y_train))

    if np.mod(run, 100) == 0:
        # plt.plot(loss)
        # plt.show()
        RL.save()

plt.plot(loss)
plt.show()

if play:
    game = connect4()
    logger = Logger()
    while not game.Done:
        for player in [0,1]:
            game.print_board()
            if player == 0:
                move = RL.get_action(game, player, 0)
            else:
                move = game.human_move(player)
            game.place(move, player)

            win = game.check_win(player, do_print=True)
            draw = game.check_draw()

            logger.log(move=move, player=player, win=win, board_before = game.board_before, board_after = game.board, draw=draw)

            if game.Done:
                break



