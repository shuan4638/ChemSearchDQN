import numpy as np
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import QED
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole


from utils import *

class Env():
    def __init__(self, initial_smiles, limit_step, atom_types, random_stop_rate = 0.01):
        self.limit_step = limit_step
        self.atom_types = atom_types
        self.random_stop_rate = random_stop_rate
        self.initial_smiles = initial_smiles
    def reset(self):
        self.smiles = self.initial_smiles
        self.time = 0
        return self.smiles

    def step(self, valid_actions, action):
        orinigal_smiles = self.smiles
        next_smiles = self.get_state(valid_actions, action)
        self.time = self.time + 1
        self.smiles = next_smiles
        done = self.complete(orinigal_smiles)
        if done:
            reward = self.finish_reward()
        else:
            reward = 0
        return next_smiles, reward, done

    def get_state(self, valid_actions, action):
        return list(valid_actions)[action]

    def complete(self, orinigal_smiles):
        t = self.time
        atom_types = self.atom_types
        limit_step = self.limit_step
        actions = self._get_valid_actions()
        return t >= limit_step or len(actions) == 0 or np.random.random() < self.random_stop_rate or orinigal_smiles == self.smiles

    def finish_reward(self):
        smiles = self.smiles
        mol = Chem.MolFromSmiles(smiles)
        qed_value = QED.qed(mol)
        return qed_value


    def get_valid_actions(self):
        smiles = self.smiles
        atom_types = self.atom_types
        mol = Chem.MolFromSmiles(smiles)
        atom_valences, atoms_with_free_valence = get_atoms_with_free_valence(mol, atom_types)
        valid_actions = set()
        valid_actions.update(
            _atom_addition(
                mol,
                atom_types=atom_types,
                atom_valences=atom_valences,
                atoms_with_free_valence=atoms_with_free_valence))
        valid_actions.update(
            _bond_addition(
                mol,
                atoms_with_free_valence=atoms_with_free_valence,
                allowed_ring_sizes=[4, 5, 6],
                allow_bonds_between_rings=False))
        valid_actions.add(smiles)
        return valid_actions

    def show_mol(self):
        mol = Chem.MolFromSmiles(self.smiles)
        return Draw.MolToImage(mol)
        
    def save_mol(self, file_name = None):
        mol = Chem.MolFromSmiles(self.smiles)
        img = Draw.MolToImage(mol)
        if file_name != None:
            img.save('%s.png' % file_name)
        else:
            img.save('%s.png' % self.smiles)
            
            
class DeepEnv():
    def __init__(self, initial_smiles, limit_step, atom_types, model, state_size = 1024, random_stop_rate = 0.01):
        self.limit_step = limit_step
        self.atom_types = atom_types
        self.random_stop_rate = random_stop_rate
        self.Rewards = defaultdict(dict)
        self.model = model
        self.state_size = state_size
        self.initial_smiles = initial_smiles

    def reset(self):
        self.smiles = self.initial_smiles
        self.time = 0
        return self.smiles

    def step(self, valid_actions, action):
        orinigal_smiles = self.smiles
        next_smiles = self.get_state(valid_actions, action)
        self.time = self.time + 1
        done = self.complete(next_smiles)
        reward = self.get_reward(next_smiles)
        self.smiles = next_smiles
        return next_smiles, reward, done

    def get_state(self, valid_actions, action):
        return list(valid_actions)[action]

    def complete(self, next_smiles):
        t = self.time
        atom_types = self.atom_types
        limit_step = self.limit_step
        actions = self.get_valid_actions()
        return t >= limit_step or len(actions) == 0 or np.random.random() < self.random_stop_rate or next_smiles == self.smiles

    def get_reward(self, next_smiles):
        fp1 = self.smiles2fps(self.smiles)
        fp2 = self.smiles2fps(next_smiles)
        fp_diff = fp2 - fp1
        reward = self.model.predict(fp_diff)[0]
        return reward

    def smiles2fps(self, smiles):
        arr = np.zeros((1,))
        mol = Chem.MolFromSmiles(smiles)
        mol = AllChem.AddHs(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits = self.state_size)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return np.array([arr])

    def get_valid_actions(self):
        smiles = self.smiles
        atom_types = self.atom_types
        mol = Chem.MolFromSmiles(smiles)
        atom_valences, atoms_with_free_valence = get_atoms_with_free_valence(mol, atom_types)
        valid_actions = set()
        valid_actions.update(
            _atom_addition(
                mol,
                atom_types=atom_types,
                atom_valences=atom_valences,
                atoms_with_free_valence=atoms_with_free_valence))
        valid_actions.update(
            _bond_addition(
                mol,
                atoms_with_free_valence=atoms_with_free_valence,
                allowed_ring_sizes=[4, 5, 6],
                allow_bonds_between_rings=False))
        valid_actions.add(smiles)
        return valid_actions
        
    def get_QED(self):
        smiles = self.smiles
        mol = Chem.MolFromSmiles(smiles)
        qed_value = QED.qed(mol)
        return qed_value

    def show_mol(self):
        mol = Chem.MolFromSmiles(self.smiles)
        return Draw.MolToImage(mol)
        
    def save_mol(self, file_name = None):
        mol = Chem.MolFromSmiles(self.smiles)
        img = Draw.MolToImage(mol)
        if file_name != None:
            img.save('%s.png' % file_name)
        else:
            img.save('%s.png' % self.smiles)
            