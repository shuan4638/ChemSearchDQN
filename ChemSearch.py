import numpy as np
import itertools
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import QED
from rdkit.Chem import Draw
import time

def get_atoms_with_free_valence(state, atom_types):
    def atom_valences(atom_types): 
        periodic_table = Chem.GetPeriodicTable()
        return [max(list(periodic_table.GetValenceList(atom_type))) for atom_type in atom_types]
    atom_valences = {atom_type: atom_valences([atom_type])[0] for atom_type in atom_types}
    atoms_with_free_valence = {}
    for i in range(1, max(atom_valences.values())):
        atoms_with_free_valence[i] = [atom.GetIdx() for atom in state.GetAtoms() if atom.GetNumImplicitHs() >= i]
    return atom_valences, atoms_with_free_valence
    
def _atom_addition(state, atom_types, atom_valences, atoms_with_free_valence):
  bond_order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE }
  atom_addition = set()
  for i in bond_order:
    for atom in atoms_with_free_valence[i]:
      for element in atom_types:
        if atom_valences[element] >= i:
          new_state = Chem.RWMol(state)
          idx = new_state.AddAtom(Chem.Atom(element))
          new_state.AddBond(atom, idx, bond_order[i])
          sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
          # When sanitization fails
          if sanitization_result:
            continue
          atom_addition.add(Chem.MolToSmiles(new_state))
  return atom_addition

def check_ring_violate(mol):
    ring_info = mol.GetRingInfo().AtomRings()
    ring_violate = False
    for ring1, ring2 in itertools.combinations(ring_info, 2):
        ask = ring1 + ring2
        cnt = 0
        for atom in ring1:
            if ask.count(atom) == 2:
                cnt += 1
                if cnt > 2:
                    ring_violate = True
                    break
    return ring_violate

def _bond_addition(state, atoms_with_free_valence, allowed_ring_sizes,
                   allow_bonds_between_rings): 

  allowed_ring_sizes = [5, 6]
  bond_orders = [
      Chem.BondType.SINGLE,
      Chem.BondType.DOUBLE,
      Chem.BondType.TRIPLE]

  free_bonds = {}
  bond_addition = set()

  for bond in range(len(bond_orders)):
    free_bonds[bond] = set()
    for valence, atoms in atoms_with_free_valence.items():
      if bond < valence:
         free_bonds[bond].update(set(atoms))
  for bond, atoms in free_bonds.items():
    for atom1, atom2 in itertools.combinations(atoms, 2): 
      if not Chem.Mol(state).GetBondBetweenAtoms(atom1, atom2) and not (state.GetAtomWithIdx(atom1).IsInRing() and state.GetAtomWithIdx(atom2).IsInRing()):
          if len(Chem.rdmolops.GetShortestPath(state, atom1, atom2)) in allowed_ring_sizes:
            new_state = Chem.RWMol(state)
            Chem.Kekulize(new_state, clearAromaticFlags=True)
            new_state.AddBond(atom1, atom2, bond_orders[bond])

            sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
            ring_violate = check_ring_violate(new_state)
            # When sanitization fails or ring violate
            if sanitization_result or ring_violate:
                continue
            bond_addition.add(Chem.MolToSmiles(new_state))
  return bond_addition

class Env():
    def __init__(self, initial_smiles, limit_step, atom_types, random_stop_rate = 0.01):
        self.initial_smiles = initial_smiles
        self.limit_step = limit_step
        self.atom_types = atom_types
        self.random_stop_rate = random_stop_rate

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
        limit_step = self.limit_step
        actions = self._get_valid_actions()
        return t >= limit_step or len(actions) == 0 or np.random.random() < self.random_stop_rate or orinigal_smiles == self.smiles

    def finish_reward(self):
        smiles = self.smiles
        mol = Chem.MolFromSmiles(smiles)
        qed_value = QED.qed(mol)
        return qed_value

    def _get_valid_actions(self):
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
        return valid_actions

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

def get_atoms_info(mol):
    atom_types = ['C', 'O', 'N']
    periodic_table = Chem.GetPeriodicTable()
    AtomNums = [periodic_table.GetAtomicNumber(atom) for atom in atom_types]
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atoms = np.array([atoms.count(x) for x in AtomNums]) 
    return atoms

def get_bonds_info(mol):
    bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    bonds =  [bond.GetBondType() for bond in list(mol.GetBonds())]
    bonds =  [bonds.count(x) for x in bond_types]
    bonds = np.array([bonds[0]+ bonds[3]/2, bonds[1]+ bonds[3]/2, bonds[2]]) 
    return bonds

def get_C_env(mol):
    Cs = []
    for i,atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == 'C':
            Cs.append(i)       
    C_dict = defaultdict(list)
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in Cs:
            C_dict[bond.GetBeginAtomIdx()].append((bond.GetBondType(), bond.GetEndAtom().GetSymbol()))
        if bond.GetEndAtomIdx() in Cs:
            C_dict[bond.GetEndAtomIdx()].append((bond.GetBondType(), bond.GetBeginAtom().GetSymbol()))
    bond_type = set()
    for env in C_dict.values():
        bond_type.update(env)

    bond_dict = {}
    bond_idx = 0
    for atom in ['C', 'O', 'N']:
        for bond in [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]:
            bond_dict[(bond, atom)] = bond_idx
            bond_idx += 1
    
    C_envs = []
    for values in C_dict.values():
        C_envs.append([bond_dict[bond] for bond in values])
    return C_envs


def get_mol_infos(smiles, radius):
    mol = Chem.MolFromSmiles(smiles)
    fps = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits = 1024)
    atoms = get_atoms_info(mol)
    bonds = get_bonds_info(mol)
    C_envs = get_C_env(mol)
    return fps, atoms, bonds, C_envs

def mol_violation(atoms1, bonds1, C_envs1, atoms2, bonds2, C_envs2):
    violate = False
    for i in range(len(atoms1)):
        if atoms1[i] > atoms2[i]:
            violate = True
            break
    for i in range(len(bonds1)):
        if bonds1[i] > bonds2[i]:
            violate = True
            break
        
    def check_intersection(a, b):
        temp = []
        for bond in a: 
            if bond in b:
                temp.append(bond)
            else:
                return False
            if a.count(bond) > b.count(bond):
                return False
        return True
    
    all_bonds = []
    for bonds in C_envs2:
        all_bonds += bonds

    for C_env1_ in C_envs1:
        check_holder = False
        for C_env2_ in C_envs2:
            if check_intersection(C_env1_, C_env2_):
                check_holder = True
#                print (C_env1_)
                for bond in C_env1_:
                    try:
                        all_bonds.remove(bond)
#                        print (all_bonds)
                    except:
                        violate = True
#                        print ('no bond')
                break
        if check_holder == False:
            violate = True
            break
        
    return violate

def filter_actions(smiles, valid_actions, target_fps, target_atoms, target_bonds, target_C_envs, radius):
    filter_actions = []
    reach = False
    mol1 = Chem.MolFromSmiles(smiles)
    fps1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius = radius, nBits = 1024)
    base_similarity = DataStructs.FingerprintSimilarity(fps1, target_fps)
    for next_smiles in valid_actions:
        fps2, atoms2, bonds2, C_envs2 = get_mol_infos(next_smiles, radius)
#        print (all(elem in target_C_envs for elem in C_envs2))
        next_similarity = DataStructs.FingerprintSimilarity(fps2, target_fps)
        if next_similarity > base_similarity and not mol_violation(atoms2, bonds2, C_envs2, target_atoms, target_bonds, target_C_envs):
#            base_similarity = next_similarity ## Accelerate
#            print (next_smiles)
#            print ('target', target_C_envs)
#            print ('next', C_envs2)
            filter_actions.append(next_smiles)
#            print (next_smiles, next_similarity)
        if next_similarity == 1:
            reach = True
            filter_actions = [next_smiles]
            break
    return filter_actions, reach

def chemical_random_episode(env, search_dict, target_fps, target_atoms, target_bonds, target_C_envs, radius):
    initial_state = env.reset()
    state = initial_state
    pre_state = initial_state
    episode = [state]
    reach = False
    while True:            
        if state not in search_dict:
            valid_actions = env._get_valid_actions()
            valid_actions, reach = filter_actions(state, valid_actions, target_fps, target_atoms, target_bonds, target_C_envs, radius)  # filter actions
            search_dict[state] = valid_actions # first meet state, record possible actions
        elif search_dict == {initial_state : []}:
           search_dict = 'terminate'
           break
        else:
            valid_actions = search_dict[state] # load updated actions
            valid_actions, reach = filter_actions(state, valid_actions, target_fps, target_atoms, target_bonds, target_C_envs, radius) ##filter again
#        print (valid_actions)
        nA = len(valid_actions)
        if nA == 0: # if len(valid_actions) == 0, fail and remove this state from dictionary and never add back
            search_dict.pop(state) # if state has no action left, delete from dictionary
            search_dict[pre_state].remove(state)
            mol1 = Chem.MolFromSmiles(state)
            fps1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=radius)
            print ('No action space, Last action: %s, similarity: %.3f' % (state, DataStructs.FingerprintSimilarity(fps1, target_fps)))
            break
        action = np.random.randint(nA)
        next_state, reward, done = env.step(valid_actions, action)
        episode.append(next_state)

        if reach == True:
            search_dict[state].remove(next_state)
            mol2 = Chem.MolFromSmiles(next_state)
            fps2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=radius)
            print ('Reach, last action: %s, similarity: %.3f' % (next_state, DataStructs.FingerprintSimilarity(fps2, target_fps)))
            break
        pre_state = state
        state = next_state
        
    return episode, reach, search_dict

def chemical_random_search(env, target_smiles, radius):
    target_fps, target_atoms, target_bonds, target_C_envs = get_mol_infos(target_smiles, radius)
    reach = False
    search_dict = {}
    search_times = 0
    reach_episodes = []
    time_start = time.time()
    while search_dict != 'terminate':
        episode, reach, search_dict = chemical_random_episode(env, search_dict, target_fps, target_atoms, target_bonds, target_C_envs, radius)
        if reach == False:
            search_times += 1
            print ('fail', len(episode), 'search time:', search_times)

        elif reach == True:
            print (episode, '\nsearch time:', search_times)
            reach_episodes.append(episode)
            reach = False
    print ('Search time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))
    return reach_episodes

initial_smiles = 'CCCC'
limit_step = 20
atom_types = ['C', 'O', 'N']
env = Env(initial_smiles, limit_step, atom_types)
target_smiles = 'NC(=N)NCCCC(N)C(=O)O'
reach_episodes = chemical_random_search(env, target_smiles, serach_radius = 3)