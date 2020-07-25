import numpy as np
import itertools
from rdkit import Chem
from collections import defaultdict
import sys

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
            

def behavior_policy(state, actions, Q):
    if state in Q: 
        next_values = []
        for next_state in actions:
            if next_state in Q[state]:
                # print (Q[state][next_state])
                next_values.append(Q[state][next_state])
            else:
                next_values.append(-100)
        return np.argmax(next_values)
    else:
        return np.random.randint(len(actions))

def epsGreedyPolicy(env, Q, eps):
    state = env.smiles
    actions = env._get_valid_actions()
    if np.random.random() < eps:
        action = np.random.randint(len(actions))
    else:
        action = behavior_policy(state, actions, Q)
    return action

def generate_episode(policy, env):
    episode = []
    state = env.reset()
    while True:
        valid_actions = env._get_valid_actions() 
        action = policy()
        next_state, reward, done = env.step(valid_actions, action)
        episode.append((state, next_state, reward))
        if done:
            break
        state = next_state
    return episode

def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0, eps = 0.1):
    returns_sum = defaultdict(dict)
    N = defaultdict(dict)
    Q = defaultdict(dict)
    best_reward = 0
    reward_history = []
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        policy = lambda: epsGreedyPolicy(env, Q, eps)
        episode = generate_episode(policy, env)
        states, actions, rewards = zip(*episode)
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            if state not in returns_sum or actions[i] not in returns_sum[state]:
                returns_sum[state][actions[i]] = sum(rewards[i:]*discounts[:-(1+i)])
                N[state][actions[i]] = 1.0
                Q[state][actions[i]] = returns_sum[state][actions[i]]/N[state][actions[i]]
            else:    
                returns_sum[state][actions[i]] += sum(rewards[i:]*discounts[:-(1+i)])
                N[state][actions[i]] += 1.0
                Q[state][actions[i]] = returns_sum[state][actions[i]]/N[state][actions[i]]
        if rewards[-1] > best_reward:
            best_reward = rewards[-1]
        reward_history.append(rewards[-1])
    return Q, best_reward, reward_history

def generate_episode_from_Q(env, Q, epsilon):
    """ generates an episode from following the epsilon-greedy policy """
    episode = []
    state = env.reset()
    while True:
        valid_actions = env._get_valid_actions()
        nA = len(valid_actions)
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                    if state in Q else np.random.randint(nA)
        next_state, reward, done = env.step(valid_actions, action)
        episode.append((state, next_state, reward))
        if done or next_state == state:
            break
        state = next_state
    return episode

def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + epsilon / nA
    return policy_s

def update_Q(env, episode, Q, alpha, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        if state in Q and actions[i] in Q[state]:
            old_Q = Q[state][actions[i]]
            Q[state][actions[i]] += alpha*(sum(rewards[i:]*discounts[:-(i+1)]) - old_Q)
        else:
            old_Q = 0
            Q[state][actions[i]] = alpha*(sum(rewards[i:]*discounts[:-(i+1)]) - old_Q)
    return Q

def mc_control(env, num_episodes, alpha, gamma=0.8, eps_start=0.7, eps_decay=0.999, eps_min=0.05):
    # initialize empty dictionary of arrays
    Q = defaultdict(dict)
    epsilon = eps_start
    best_reward = 0
    states_history = []
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % (100) == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # set the value of epsilon
        epsilon = max(epsilon*eps_decay, eps_min)
        # generate an episode by following epsilon-greedy policy
        episode = generate_episode_from_Q(env, Q, epsilon) 
        # update the action-value function estimate using the episode
        Q = update_Q(env, episode, Q, alpha, gamma)
        states, actions, rewards = zip(*episode)
        states_history.append(states)
        if rewards[-1] > best_reward:
            best_reward = rewards[-1]
    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k,np.argmax(list(v.values()))) for k, v in Q.items())
    return policy, Q, states_history, best_reward
