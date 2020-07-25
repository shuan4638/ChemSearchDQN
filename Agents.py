import random
import numpy as np
from collections import deque

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import QED
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from utils import *
class RLearnAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95       # discount rate
        self.epsilon = 1.0      # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                        optimizer=Adam(lr=self.learning_rate))
        return model

    def get_next_state(self, Q, state):
        next_states = list(Q[state].keys())
        act_dim = len(next_states)
        return next_states[np.random.randint(act_dim)]

    def train(self, previous_state, state, next_state, Q):
        V1 = Q[previous_state][state]
        V2 = Q[state][next_state]
        R = V1 - V2
        R = np.array([R])
        fp1 = self.smiles2fps(state)
        fp2 = self.smiles2fps(next_state)
        fp_diff = fp2 - fp1
        history = self.model.fit(fp_diff, R, epochs=1, verbose=0)
        return history.history['loss'][0]

    def smiles2fps(self, smiles):
        arr = np.zeros((1,))
        mol = Chem.MolFromSmiles(smiles)
        mol = AllChem.AddHs(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits = self.state_size)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return np.array([arr])
    
    
class DQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95       # discount rate
        self.epsilon = 1.0      # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning model
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def smiles2fps(self, smiles):
        arr = np.zeros((1,))
        mol = Chem.MolFromSmiles(smiles)
        mol = AllChem.AddHs(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits = self.state_size)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return np.array([arr])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions): ### epslon greedy policy ###
        action_size = len(valid_actions)
        if np.random.rand() <= self.epsilon:
            return random.randrange(action_size)
        fp = self.smiles2fps(state)
        act_values = self.model.predict(fp)
        return np.argmax(act_values[0])     # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            fp1 = self.smiles2fps(state)
            fp2 = self.smiles2fps(next_state)
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(fp2)[0]))
                
            target_f = self.model.predict(fp1)[0]
            
            self.model.fit(fp1, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)