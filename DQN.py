import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys

from utils import *
from Envs import *
from Agents import *

# Generate experience from first visit MC control
initial_smiles = 'c1ccccc1'
limit_step = 15
atom_types = ['C', 'O', 'N']
env = Env(initial_smiles, limit_step, atom_types)
policy, increment_Q, states_history, best_reward = mc_control(env, 10000, 0.02, eps_start=0.4)

# Train the DQN from experience
epochs = 200
teacher_Q = increment_Q
agent = RLearnAgent(1024)
loss_history = []
for n, states in enumerate(states_history):
    if len(states) >= 3:
        for i in range(len(states)):
            previous_state = states[i]
            state = states[i+1]
            next_state = states[i+2]
            loss = agent.train(previous_state, state, next_state, teacher_Q)
            loss_history.append(loss)
            if (i+4) > len(states):
                break
    if True:
        print ('\rTraining epoch: %d/%d' % (n+1, len(states_history)), end="")
        sys.stdout.flush()

print ('Plotting DQN training history')
plt.plot(loss_history)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
plt.close()

# Generate optimized molecule from DQN
initial_smiles = 'c1ccccc1'
limit_step = 15
atom_types = ['C', 'O', 'N']
state_size = 1024
env = DeepEnv(initial_smiles, limit_step, atom_types)
agent = DQNAgent(state_size)
done = False
batch_size = 32
EPISODES = 200
R = defaultdict(dict)
ep_score_history = np.array([])

for e in range(EPISODES):
    state = env.reset()
    score = 0
    for n in range(500):
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions)
        next_state, reward, done = env.step(valid_actions, action)
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        score += reward
        if done:
            print (state)
            print (score)
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    if e % 10 == 0:
        print ('episode:', e)
