from game import connect4
from my_functs import *
import numpy as np
import matplotlib.pyplot as plt

EPSILON_MIN = 0.01
EPSILON_MAX = 0.02
DECAY = 0.9999

epsilon = EPSILON_MAX

epsilon = epsilon * DECAY
if epsilon <= EPSILON_MIN:
    epsilon = EPSILON_MIN

''' GAME '''
model = get_model()
loaded = False
player = 0
players = [0, 1]
MAX_EXP = 10000
TRAINING_STEPS = 1000
loss = []


for run in range(TRAINING_STEPS):
    print(run, '/', TRAINING_STEPS, '  ', '{:.2f}%'.format(100 * run / TRAINING_STEPS), end='\r')
    experience1 = []
    experience2 = []
    done = False
    game = connect4(player1=-1, player2=1)
    while not done:
        for player in players:
            # game.print_board()
            # show board to agent and get move (must be valid)

            ''' GET EPSILON'''
            epsilon = epsilon * DECAY
            if epsilon <= EPSILON_MIN:
                epsilon = EPSILON_MIN

            # play move, get reward, experience is board, move, reward
            # move = game.move(player, return_move = True, do_move = False)

            move = get_action(board = game.board, flip = player, epsilon = epsilon, game=game, model = model)

            exp1, exp2, done = game.RL_play(player, move)
            experience1.append(exp1)
            experience2.append(exp2)
            '''player sees itself as p1 always'''

            if done:
                break

            ''' have observations, moves and rewards 
            learn rewards from obs and then choose move based on best future observation? '''

    if 'x_train' not in locals():
        x_train, y_train = modify_rewards(experience1)


    x, y = modify_rewards(experience1)
    x_train = np.concatenate([x_train, x])
    y_train = np.concatenate([y_train, y])

    x, y = modify_rewards(experience2)
    x_train = np.concatenate([x_train, x])
    y_train = np.concatenate([y_train, y])

    if x_train.shape[0] > MAX_EXP:
        x_train = x_train[-MAX_EXP:,:]
        y_train = y_train[-MAX_EXP:, :]

    selection = np.random.randint(0, x_train.shape[0]-1 , min(x_train.shape[0], 128))

    if not loaded:
        model.load_weights('agent.hdf5')
        loaded = True

    loss.append(np.mean(np.square(model(x_train[selection, :]) - y_train[selection, :])))
    model.fit(x_train[selection,:], y_train[selection,:], batch_size = 32, verbose=False)



plt.plot(loss)
plt.savefig('loss.png')
model.save_weights('agent.hdf5')



