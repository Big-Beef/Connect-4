from game import connect4
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt




class DeepModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(DeepModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in range(hidden_units):
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='sigmoid', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


# class ConnectX(gym.Env):
#     def __init__(self, switch_prob=0.5):
#         # self.env = make('connectx', debug=False)
#         # self.env =
#         self.pair = [None, 'random']
#         self.trainer = self.env.train(self.pair)
#         self.switch_prob = switch_prob
#
#         # Define required gym fields (examples):
#         config = self.env.configuration
#         self.action_space = gym.spaces.Discrete(config.columns)
#         self.observation_space = gym.spaces.Discrete(config.columns * config.rows)
#
#     def switch_trainer(self):
#         self.pair = self.pair[::-1]
#         self.trainer = self.env.train(self.pair)
#
#     def step(self, action):
#         return self.trainer.step(action)
#
#     def reset(self):
#         if np.random.random() < self.switch_prob:
#             self.switch_trainer()
#         return self.trainer.reset()
#
#     def render(self, **kwargs):
#         return self.env.render(**kwargs)

class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences = 0, batch_size = 8, ):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam()
        self.gamma = gamma
        self.model = DeepModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}  # The buffer
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    @tf.function
    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            # Only start the training process when we have enough experiences in the buffer
            return 0

        # Randomly select n experience in the buffer, n is batch-size
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.preprocess(self.experience['s'][i]) for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])

        # Prepare labels for training process
        states_next = np.asarray([self.preprocess(self.experience['s2'][i]) for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    # Get an action by using epsilon-greedy
    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(7)
            # return int(np.random.choice([c for c in range(self.num_actions) if state.board[c] == 0]))
        else:
            prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].numpy()
            # for i in range(self.num_actions):
            #     if state.board[i] != 0:
            #         prediction[i] = -1e7
            return int(np.argmax(prediction))

    # Method used to manage the buffer
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        ref_model = tf.keras.Sequential()

        ref_model.add(self.model.input_layer)
        for layer in self.model.hidden_layers:
            ref_model.add(layer)
        ref_model.add(self.model.output_layer)

        ref_model.load_weights(path)

    # Each state will consist of the board and the mark
    # in the observations
    def preprocess(self, state):
        # result = state.board[:]
        result = state
        # result.append(state.mark)

        return result




def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    while not done:
        # Using epsilon-greedy to get an action
        action = TrainNet.get_action(observations, epsilon)

        # Caching the information of current state
        prev_observations = observations

        # Take action
        observations, reward, done, _ = env.step(action)

        # Apply new rules
        if done:
            if reward == 1: # Won
                reward = 20
            elif reward == 0: # Lost
                reward = -20
            else: # Draw
                reward = 10
        else:
            reward = -0.05 # Try to prevent the agent from taking a long move

        rewards += reward

        # Adding experience into buffer
        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)

        # Train the training model by using experiences in buffer and the target model
        TrainNet.train(TargetNet)
        iter += 1
        if iter % copy_step == 0:
            # Update the weights of the target model when reaching enough "copy step"
            TargetNet.copy_weights(TrainNet)
    return rewards


player1 = DQN(7*6, 7, 2, gamma = 0.99, max_experiences=10000, min_experiences = 100, batch_size = 32)
player2 = DQN(7*6, 7, 2, gamma = 0.99, max_experiences=10000, min_experiences = 100, batch_size = 32)
TargetNet = DQN(7*6, 7, 2, gamma = 0.99, max_experiences=10000, min_experiences = 100, batch_size = 32)
#
# game.print_board()

# player = 1
# # env = ConnectX()
# while(1):
#     game.move(player)
#     game.print_board()
#     if game.check_win(player):
#         break
#
#     player = player * -1


rewards = 0
iter = 0
done = False

min_epsilon = 0.1
decay = 0.99999
player = 1
epsilon = 0.99
action = None
maxiter = 10
player1.load_weights('weights.hdf5')
player2.load_weights('weights.hdf5')
for i in range(maxiter):
    print(i, '/', maxiter, '  ', '{:.2f}%'.format(100*i/maxiter), end ='\r')
    game = connect4(7, 6)
    count = 0

    while(1):
        rewardp1 = 0
        rewardp2 = 0
        reward = 0
        epsilon = max(min_epsilon, epsilon * decay)
        prev_observations = game.board
        t_count = 0
        while(1):
            t_count = t_count+1

            if player == 1:
                action = player1.get_action(np.reshape(game.board, -1),epsilon)+1
            elif player == -1:
                action = player2.get_action(np.reshape(game.board, -1),epsilon)+1


            if t_count > 10:
                action = np.random.randint(7)
            if game.check_valid(action):
                game.place(action, player)
                break
            else:
                reward -= 0.1

        # game.print_board()

        if player == 1:
            rewardp1 = reward
        else:
            rewardp2 = reward
        done = False
        if game.check_win(player):
            done = True
            if player == 1:
                rewardp1 = 1
                rewardp2 = -1

            elif player == -1:
                rewardp2 = 1
                rewardp1 = -1
            break
        if game.check_draw():
            rewardp1 = -0.1
            rewardp2 = -0.1
            done = True
            break

        expp1 = {'s': prev_observations, 'a': action, 'r': rewardp1, 's2': game.board, 'done': done}
        expp2 = {'s': prev_observations, 'a': action, 'r': rewardp2, 's2': game.board, 'done': done}

        player1.add_experience(expp1)
        player2.add_experience(expp2)
        player = player * -1
        count = count + 1
    # Train the training model by using experiences in buffer and the target model
    player1.train(TargetNet)
    player2.train(TargetNet)
    iter += 1

    # Update the weights of the target model when reaching enough "copy step"
    TargetNet.copy_weights(player1)

player1.save_weights('weights.hdf5')
game = connect4(7, 6)
count = 0




rewardp1 = 0
reward = 0
epsilon = max(min_epsilon, epsilon * decay)
prev_observations = game.board
t_count = 0
while(1):
    t_count = t_count+1
    game.print_board()
    if player == -1:
        action = player1.get_action(np.reshape(game.board, -1),epsilon)+1

        if t_count > 10:
            action = np.random.randint(7)
        if game.check_valid(action):
            game.place(action, player)
            break
        else:
            reward -= 0.1

            # game.print_board()

            if player == 1:
                rewardp1 = reward
            done = False
            if game.check_win(player):
                done = True
                if player == 1:
                    rewardp1 = 1
                elif player == -1:
                    rewardp1 = -1
                break
            if game.check_draw():
                rewardp1 = -0.1
                done = True
                break

            expp1 = {'s': prev_observations, 'a': action, 'r': rewardp1, 's2': game.board, 'done': done}

            player1.add_experience(expp1)

    else:
        game.move(player)

    player = player * -1
    count = count + 1

    
# Train the training model by using experiences in buffer and the target model
player1.train(TargetNet)
player2.train(TargetNet)
iter += 1

# Update the weights of the target model when reaching enough "copy step"
TargetNet.copy_weights(player1)

# while not done:
#     # Using epsilon-greedy to get an action
#
#     action = TrainNet.get_action(observations, epsilon)
#
#     # Caching the information of current state
#     prev_observations = observations
#
#     # Take action
#     observations, reward, done, _ = env.step(action)
#
#     # Apply new rules
#     if done:
#         if reward == 1: # Won
#             reward = 20
#         elif reward == 0: # Lost
#             reward = -20
#         else: # Draw
#             reward = 10
#     else:
#         reward = -0.05 # Try to prevent the agent from taking a long move
#
#     rewards += reward
#
#     # Adding experience into buffer
#     exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
#     TrainNet.add_experience(exp)
#
#     # Train the training model by using experiences in buffer and the target model
#     TrainNet.train(TargetNet)
#     iter += 1
#     if iter % copy_step == 0:
#         # Update the weights of the target model when reaching enough "copy step"
#         TargetNet.copy_weights(TrainNet)
# return rewards