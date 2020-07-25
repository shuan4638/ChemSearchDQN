# ChemSearch DQN

In MolDQN, the Agent learn the property in the chemical space continuously. However in the real world, the chemicals with knwon property are very few (sparse).

Considering this issue, ChemSearch DQN aims to learns the property only at the chemical point with kwnown property.

For easy demonstration, this work assumed the chemical property can be found after a few steps of chemical contruction.
<img src="https://github.com/shuan4638/ChemSearchDQN/figs/blob/master/Sparsedata.png">

## Learn from terminate state

To make this more practical, an algorithm that find all the possible path from initial molecule to target molecules was proposed.

See ChemSearch.py to discover the constructing pathway from C4 to Arginine.

After finding the pathways of chemical construction, DQN could learn from this pathways and be applied to real world dataset.

<img src="https://github.com/shuan4638/ChemSearchDQN/blob/master/Targetsearch.png">

## Installation

All the chemical decription were done by rdkit. Please use conda to install rdkit.

```bash
conda install -c conda-forge rdkit
```

## Authors
This code was written by Shuan Chen (PhD candidate of KAIST CBE) in 2020 for DQN practice.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

This work was inspired by :

## Credit

"Optimization of molecules via deep reinforcement learning", arXiv preprint arXiv:1810.08678, 2018

and github code from https://github.com/aksub99/MolDQN-pytorch
