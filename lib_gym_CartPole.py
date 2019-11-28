import os
import numpy as np
import gym
import random
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input
from collections import deque
from  matplotlib import pyplot as plt
import pickle
import time

"""remove message:  I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports
instructions that this TensorFlow binary was not compiled to use: AVX2"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Deep_Neural_Network:
    """ create the structure of the DEEP NEURAL NETWORK that will be used to calculate the Q Value"""

    def __init__(self, state_size, action_size, fc1=64, fc2=64):

        self.state_size = state_size                       # number of continuous features on the State
        self.action_size = action_size                     # number of actions
        self.fc1 = fc1                                     # number of neurons on the 1st fully connected layer
        self.fc2 = fc2                                     # number of neurons on the 2nd fully connected layer

        # Create Deep Neural Network Model
        self.X_input = Input(shape=(self.state_size,))
        self.X = Dense(units=self.fc1, activation='relu')(self.X_input)
        self.X = Dense(units=self.fc2, activation='relu')(self.X)
        self.X = Dense(units=self.action_size, activation='linear')(self.X)
        self.model = Model(inputs=self.X_input, outputs=self.X)


class Agent:
    """ Create an Agent that will interact with the environment and decide the best action to take in order to receive
    more rewards. """

    def __init__(self, env, epsilon, eps_min, eps_decay, tau, learning_rate, gamma,  batch_size, update_every):

        self.env = env                                    # instance of an environment of Ai Gym
        self.action_size = env.action_space.n             # number of action
        self.state_size = env.observation_space.shape[0]  # state size (input of the DQN - number of features)
        self.epsilon = epsilon                            # exploration rate
        self.eps_min = eps_min                            # min exploration rate
        self.eps_decay = eps_decay                        # factor to decrease the exploration rate at each episode
        self.tau = tau                                    # learning rate to update the parameters of the Next State DQN
        self.learning_rate = learning_rate                # learning rate to train the State DQN
        self.gamma = gamma                                # discount rate of the future rewards
        self.memory = []                                  # list with some episodes (S, A, R, Next Sate)
        self.batch_size = batch_size                      # number of examples the DQN_State will train at the same time
        self.update_every = update_every                  # number of time steps we need to take,  before update DQN_State and DQN_Next
        self.count = 0                                    # count number of times steps

    def create_DQN(self):

        # DQN STATE -- used to predict the Q(state).  Input = State / Output = Q value for all Actions
        self.DQN_State = Deep_Neural_Network(state_size=self.state_size, action_size=self.action_size)
        self.DQN_State.model.compile(optimizer='adam', loss='MSE', metrics=['accuracy'])

        # DQN NEXT -- used to predict the Q(Next State) and used to calculate the "Target" to train the DQN STATE
        self.DQN_Next = Deep_Neural_Network(state_size=self.state_size, action_size=self.action_size)

        # inicialize DQN Next with the same random parameters used to inicialize DQN State
        self.DQN_Next.model.set_weights(self.DQN_State.model.get_weights())

        self.DQN_State.model.summary()


    def update_memory(self, episode):
        self.memory.append(episode)


    def train_DQN_State(self):

        # Suffle the Memory -- break the correlation that exists btw state and actions that are close to each other
        random.shuffle(self.memory)

        states, actions, rewards, next_states, dones = list(zip(*self.memory[:self.batch_size]))
        state_idx = np.arange(self.batch_size)
        ones = np.ones(self.batch_size)
        states = np.array(states)
        next_states = np.array(next_states)

        # calculate the Target values that will be used to train the DQN_State.
        target = self.DQN_State.model.predict(states)
        Q_Next = self.DQN_Next.model.predict(next_states)
        # For each episode (m) we will change only 1 value related to the pair (State, Action) that we want to update.
        # If the Next State = Terminal State,(done = True),  Q(Next State, for all Actions)=0  ==> (ones-dones)
        target[state_idx, actions] = rewards + self.gamma * np.max(Q_Next, axis=1) * (ones - dones)

        # # train DQN_State -- update parameters
        self.DQN_State.model.fit(x=states, y=target, batch_size=self.batch_size, epochs=1, verbose=0)


    def update_parameters_DQN_Next(self):

        # after x-times Q_State is trained, we update the parameters of Q_Next using the updated parameters of Q_State
        self.theta = self.DQN_State.model.get_weights()
        self.target_theta = self.DQN_Next.model.get_weights()

        # soft update using the learning rate tau
        self.target_theta = self.tau * np.array(self.theta) + (1-self.tau)*np.array(self.target_theta)

        # update parameters of DQN_Next
        self.DQN_Next.model.set_weights(self.target_theta)


    def acc(self, state):

        if np.random.rand() < self.epsilon:
            action = env.action_space.sample()
            return action
        else:
            state = state.reshape(1, 4)
            action = np.argmax(self.DQN_State.model.predict(state))
            return action


    def update_epsilon(self):

        # after each episode the exploration rate should decrease and the agent will exploit more the best actions
        self.epsilon = max(self.eps_min, self.eps_decay*self.epsilon)


    def step(self, state, action, reward, next_state, done):

        self.update_memory((state, action, reward, next_state, done))

        self.count = self.count + 1

        if len(self.memory) > self.batch_size:
            if self.count % self.update_every == 0:
                self.train_DQN_State()
                self.update_parameters_DQN_Next()


def save_results(object):
    object = object
    file_out = open('C:\\PY4E\\files_test\CartPole.pickle', 'wb')
    pickle.dump(object, file_out)
    file_out.close()


def DQNModel(env, num_episodes):

    scores = deque(maxlen=100)
    rewards = []

    agent = Agent(env, epsilon=1, eps_min=0.01, eps_decay=0.995, tau=0.1, learning_rate=0.1, gamma=0.99, batch_size=64, update_every=4)

    agent.create_DQN()

    for i in range(1, num_episodes+1):

        score = 0
        state = env.reset()

        while True:
            action = agent.acc(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score = score + reward
            rewards.append(score)
            if done:
                scores.append(score)
                if i >= 100:
                    rewards.append(np.average(scores))
                    print('Episode:{}......Average Score:{}......Epsilon:{:.2f}'.format(i, np.average(scores), agent.epsilon))
                break

        if np.average(scores) >= 195:
            print('Congratulations, problem solved!!!')
            save_results(agent.DQN_State.model.get_weights())
            break

        agent.update_epsilon()

    return rewards



env = gym.make('CartPole-v1')

print('observation space', env.observation_space)
print('action space', env.action_space)
print('observation space low', env.observation_space.low)
print('observation space high', env.observation_space.high)

# TRAIN THE DQN
# rewards = DQNModel(env, num_episodes=250)

# PLOT THE REWARDS
# plt.plot(rewards)
# plt.show()



# Play the game using the trained DQN

# load the trained parameters
file_in = open('C:\\PY4E\\files_test\CartPole.pickle', 'rb')
object = pickle.load(file_in)
DQN_State_Weights = object

# create an agent and the DQN
agent = Agent(env, epsilon=1, eps_min=0.01, eps_decay=0.995, tau=0.1, learning_rate=0.1, gamma=0.99, batch_size=64, update_every=4)
agent.create_DQN()
agent.DQN_State.model.set_weights(DQN_State_Weights)

# play the game
for i in range(1, 6):

    state = env.reset()
    step = 0
    score = 0
    while step < 201:
        step = step + 1
        img = plt.imshow(env.render(mode='rgb_array'))
        # action = env.action_space.sample()
        action = np.argmax(agent.DQN_State.model.predict(state.reshape(1, env.observation_space.shape[0])))
        next_state, reward, done, info = env.step(action)
        state = next_state
        img.set_data(env.render(mode='rgb_array'))
        score = score + reward

        if done:
            print('episode', i,'score', score)
            time.sleep(1)
            print(state)
            break

        if step == 200:
            print('episode', i, 'score', score)
            time.sleep(1)

env.close()



