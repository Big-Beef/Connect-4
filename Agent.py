import tensorflow as tf
import numpy as np
import copy
import os

def loss(y_pred, y_true):
    loss = tf.math.abs(y_true - y_pred)

    loss = 10*tf.math.reduce_mean(loss[tf.not_equal(y_true, 0)])
    return loss

class Agent:
    def __init__(self, game):
        Input = tf.keras.layers.Input(shape=42)
        Dense1 = tf.keras.layers.Dense(42, activation='softplus')(Input)
        Dense2 = tf.keras.layers.Dense(512, activation='relu')(Dense1)
        Dense3 = tf.keras.layers.Dense(128, activation='relu')(Dense2)
        Output = tf.keras.layers.Dense(7)(Dense3)



        self.model = tf.keras.models.Model(inputs=Input, outputs=Output)

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.9),
                      loss=loss)

    def load(self):
        try:
            self.model.load_weights(os.getcwd() + '/save' + 'agent.hdf5')
        except:
            print('Unable to load model')
            pass

    def save(self):
        self.model.save_weights(os.getcwd() + '/save' + '/agent.hdf5')

    def get_action(self, game, player, epsilon):
        if epsilon > np.random.rand():
            while(1):
                move = np.random.randint(7)
                if game.check_valid(move):
                    return move

        flip = 1
        if player == 1:
            flip = -1
        flattened_input = flip*copy.copy(game.board.flatten()[np.newaxis, :])
        actions = np.argsort(self.model(flattened_input))

        for i in actions[0,:]:
            if game.check_valid(i):
                return i

    def train(self, x_train, y_train):
        random_selection = np.random.randint(0,x_train.shape[0] - 1 , min(x_train.shape[0], 128))
        self.model.fit(x_train[random_selection], y_train[random_selection], batch_size = 32, epochs = 5, verbose = False)

        random_selection2 = np.random.randint(0, x_train.shape[0] - 1, min(x_train.shape[0], 64))
        y_pred = self.model.predict(x_train[random_selection2])
        return loss(y_pred, y_train[random_selection2])






