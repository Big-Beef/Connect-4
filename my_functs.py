import numpy as np
import tensorflow as tf

def get_model():
    loss = tf.keras.losses.MeanSquaredError()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dense(7))

    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9),
                  loss = loss)
    return(model)

def get_action(board, flip, epsilon, game, model):
    mult = 1
    if flip:
        mult = -1
    if (np.random.rand() < epsilon):
        valid = False
        while not valid:
            move = np.random.randint(7)
            valid = game.check_valid(move)


    predictions = model((board.flatten() * mult)[np.newaxis, :])
    predictions = np.squeeze(np.flip(np.argsort(predictions)))
    valid = False
    i = 0
    while not valid:
        move = predictions[i]
        valid = game.check_valid(move)
        i += 1

    return (move)


def one_hot(a, num_classes=7):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def modify_rewards(exp, gamma = 0.9):
    rewards = []
    observation = []
    move = []
    n_values = 7

    for i in reversed(exp):
        rewards.append(i['R'])
        observation.append(i['observation'])
        move.append(i['move'])


    for idx in range(len(rewards)-1):
        rewards[idx+1] = rewards[idx+1] + rewards[idx] * gamma
    rewards = np.flip(np.asarray(rewards))

    observation = np.asarray(observation)

    move = np.asarray(move)
    move = one_hot(move)

    rewards = np.transpose(np.tile(rewards, (7, 1)))

    y_train = rewards * move

    return observation, y_train