import numpy as np
import os
class Logger:
    def __init__(self, load = False):
        self.max_len = 10000
        self.board_before = []
        self.board_after = []
        self.move = []
        self.player = []
        self.win = []
        self.draw = []
        self.x_train = np.zeros((1,42))
        self.y_train = np.zeros((1,7))

        self.cwd = os.getcwd()
        self.save_dir = self.cwd+'/save'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if load:
            self.load()

    def log(self, move, player, win, board_before, board_after, draw):
        self.board_before.append(board_before)
        self.board_after.append(board_after)
        self.move.append(move)
        self.player.append(player)
        self.win.append(win)
        self.draw.append(draw)

    def Discount_Reward(self, reward, gamma=0.5, lim_ahead=None):
        reward = np.flip(reward)
        for idx in range(len(reward) - 1):
            reward[idx + 1] = reward[idx + 1] + reward[idx] * gamma
        reward = np.flip(reward)
        return reward

    def one_hot(self, a, num_classes=7):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    def save(self):
        os.chdir(self.save_dir)
        np.save('observations.npy', self.x_train)
        np.save('rewards.npy', self.y_train)
        os.chdir(self.cwd)

    def load(self):
        os.chdir(self.save_dir)
        try:
            self.x_train = np.load('observations.npy')
            self.y_train = np.load('rewards.npy')
        except:
            print('no replay buffer found')
        os.chdir(self.cwd)


    def convert(self):
        if self.win[-1]:
            self.win.append(-1)
            self.draw.append(0)
        if self.draw[-1]:
            self.win.append(0)
            self.draw.append(1)

        self.player.append(self.player[-2])
        self.player = np.asarray(self.player)

        self.board_before.append(self.board_after[-1])
        self.board_before = np.asarray(self.board_before)

        board_p0 = np.asarray(self.board_before)[self.player == 0]
        board_p1 = np.asarray(self.board_before)[self.player == 1] * -1

        reward_p0 = np.asarray(self.win).astype(int)[self.player == 0] + np.asarray(self.draw).astype(int)[self.player == 0] * -0.1
        reward_p1 = np.asarray(self.win).astype(int)[self.player == 1] + np.asarray(self.draw).astype(int)[self.player == 1] * -0.1

        reward_p0 = self.Discount_Reward(reward_p0)
        reward_p1 = self.Discount_Reward(reward_p1)

        move_p0 = np.asarray(self.move)[self.player[:-1] == 0]
        move_p1 = np.asarray(self.move)[self.player[:-1] == 1]

        len_p0 = move_p0.shape[0]
        len_p1 = move_p1.shape[0]

        board_p0 = board_p0[:len_p0, :, :].reshape((len_p0, -1))
        board_p1 = board_p1[:len_p1, :, :].reshape((len_p1, -1))

        reward_p0 = reward_p0[-len_p0:]
        reward_p1 = reward_p1[-len_p1:]

        reward_p0 = self.one_hot(move_p0) * np.transpose(np.tile(reward_p0, (7, 1)))
        reward_p1 = self.one_hot(move_p1) * np.transpose(np.tile(reward_p1, (7, 1)))

        x = np.append(board_p0, board_p1, 0)
        y = np.append(reward_p0, reward_p1, 0)

        self.x_train = np.append(self.x_train, x, 0)
        self.y_train = np.append(self.y_train, y, 0)

        if self.x_train.shape[0] > self.max_len:
            self.x_train = self.x_train[-self.max_len:]
            self.y_train = self.y_train[-self.max_len:]

        self.board_before = []
        self.board_after = []
        self.move = []
        self.player = []
        self.win = []
        self.draw = []








