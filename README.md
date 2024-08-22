# Self-assembling peptides

The project's objective is to identify the optimal arrangement of 20 natural peptides within sequences of 3 or 5, facilitating seamless self-assembly for protein formation. To achieve this goal, a Reinforcement Learning model is trained through the creation of a customized environment using OpenAI's Gym library.

## Idea Behind Code:

Given a specific length of the protein chain, the environment randomly selects a position and an amino acid, chosen from the pool of 20 natural amino acids. Subsequently, the selected amino acid is placed at the chosen position within the protein chain. Following this modification, a reward from [1, 0, -1] is assigned based on the calculated hydrophobicity value of the altered protein chain. The cumulative rewards are then aggregated, and the objective is to maximize this score by employing the Proximal Policy Optimization Algorithm over a specified number of total timesteps.  

## Code Explanation:

### <u>Reward Scheme:</u>

```python
# Modules
from rdkit import Chem
from rdkit.Chem import Crippen
```

First, the integer array specifying the amino acid at a position of the protein chain is converted from integers to strings.

```python
def indices_to_peptide(indices):
    amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    peptide_sequence = '-'.join([amino_acids[i] for i in indices])
    return peptide_sequence
```

Then, this string sequence is passed to a function to get the logP value, specifying hydrophobicity.

```python
def calculate_logP(peptide_sequence):

    # Create an RDKit molecule object
    mol = Chem.MolFromSequence(peptide_sequence)

    # Check if the molecule is valid
    if mol is not None:
        logP = Crippen.MolLogP(mol)
        return logP
    else:
        # Return a placeholder value
        return -1000000000.0
```
<br>

#### <u>Illustration:</u> 

Taking randomized protein chain of length 5.

```python
peptide = MultiDiscrete([20] * len_peptide).sample()

# Convert indices to a tripeptide sequence
sequence = indices_to_peptide(peptide)

hydrophobicity = calculate_logP(sequence)

print('Protein chain as interger: {}\nProtein chain as string: {}\nHydrophobicity value for protein chain: {}'.format(peptide, sequence, hydrophobicity))
```

```python
Output: 
Protein chain as interger: [18 13 17 17 12]
Protein chain as string: TYR-PHE-TRP-TRP-MET
Hydrophobicity value for protein chain: -9.586089999999931
```
<br>

### <u>Custom Environment:</u>

Custom Environment named `peptideEnv` is made using `Env module of the gym library`. The standard environment made using OpenAI's gym library must have four functions defined:

```bash
class peptideEnv(Env)
├── def __init__(self)
├── def step(self, action)
├── def render(self)
└── def reset(self)
```

```python
# Installing gym and stable-baselines3
!pip install stable-baselines3[extra]
!pip install gym

# get gym library path and add to Python interpreter's search path
!pip show gym
import sys
sys.path.append('c:\\users\\hp\\desktop\\projectpeptide\\summer_internship_2023\\projectpeptide\\lib\\site-packages')
```

```python
# Importing modules
import gym 
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
```

- `__init__(self)`
    - Initialized the action space, observation space, state and other critical values for the model.
    - Action space randomly choose an amino acid from the 20 standard acids and a position in the chain.
    - Observation space defines the total possible outcomes.
    - Initialized state as a random protein sequence of the given length using `MultiDiscrete` space.

```python
    def __init__(self):
    
    self.action_space = MultiDiscrete([20, len_peptide])
    self.observation_space = MultiDiscrete([20] * len_peptide)
    self.state = MultiDiscrete([20] * len_peptide).sample()

    # critical values
    self.critical_logP = -1000000000.0
    self.numAction = 100
```

- `step(self, action)`
    - Takes an action and update the state variable.
    - Assigns reward of 1 if logP value increases from previous iteration, 0 if it is equal and -1 otherwise.

```python
    def step(self, action):
    
    # Extract amino acid and position from the action
    amino_acid = action // len_peptide
    position = action % len_peptide

    # Update the state based on the action
    self.state[position] = amino_acid
    self.numAction -=1
    
    # Convert indices to a peptide sequence and calculate logP 
    # for the peptide
    sequence = indices_to_peptide(self.state)
    logP_value = calculate_logP(sequence)
    
    # Calculate reward
    if logP_value > self.critical_logP: 
        reward = 1 
        self.critical_logP = logP_value
    elif logP_value == self.critical_logP:
        reward = 0
    else:
        reward = -1 
    
    # Check if is done
    if self.numAction <= 0: 
        done = True
    else:
        done = False
    
    info = {}
    return self.state, reward, done, info
```

- `render(self)` 
    - Optional and is used for visualizing the environment.

- `reset(self)`

```python
def reset(self):
    # Reset the peptide to a new randomized state
    self.state = MultiDiscrete([20] * len_peptide).sample()
    
    # Reset critical_logP at the beginning of each episode
    self.critical_logP = -1000000000.0
    self.numAction = 100
    return self.state
```

### <u>Training, Saving and Evaluating the model:</u>

- Model trained using PPO algorithm, Mlpolicy with 500000 timesteps.

```python
# Importing modules
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
```

```python
# train
env = peptideEnv()
log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log = log_path)
model.learn(total_timesteps= 500000, log_interval= 10)

# saved as PPO
model.save('PPO')

#evaluate
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
```

## Graphical Results:

![Results](https://github.com/ajiteshshree/summer_internship_2023-24/blob/main/Extras/Graph.png)

### Inferences:
- Average Episode Reward increases with the iterations.
- Loss decreases with more iterations.

## References:

- <a href="https://towardsdatascience.com/policy-gradient-methods-104c783251e0"> Policy Gradient Method Explanation-1</a><br>
- <a href="https://huggingface.co/learn/deep-rl-course/unit4/policy-gradient?fw=pt"> Policy Gradient Method Explanation-2</a><br>
- <a href="https://drive.google.com/file/d/1kmlz5kaJnC4FblE5oBK-0NnJRRSLm_pI/view?usp=drive_link"> ML solving Peptide self-assembly paper</a>
- <a href="https://youtu.be/Mut_u40Sqz4?si=AwunkKbTvgTM1PvI"> Reinforcement Learning Full Course by Nicholas Renotte</a><br>
- <a href="https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html"> Proximal Policy Optimization Algorithm documentation Stable-Baselines3</a><br>

