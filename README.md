# -NIPS-2017-Learning-to-Run
Reinforcement learning environments with musculoskeletal models   
https://www.crowdai.org/challenges/nips-2017-learning-to-run  
  
CANDIDATE ALGORITHMS:  

Depp Deterministic Policy Gradient---DDPG(https://arxiv.org/abs/1509.02971)  
Recurrent Deterministic Policy Gradient---RDPG(https://arxiv.org/pdf/1512.04455.pdf)  
Trust  
MOTIVATION:  
In this case, we are manipulating the muscles rather than the velocity of the body parts, i.e., if I understood the problem correctly, we are changing the acceleration, which should be considered a second-order markov chain. LSTM and recurrent network might be able to capture the long short term dependencies.  
  
FOlDERS AND FILES:  
DDPG: The standard implementation of ddpg in tensorflow, following the pesudocode in the paper.  
RDPG: The recurrent version of the ddpg in tensorflow(There are complications that have not been resolved)
  
DEPENDENCIES:
Tensorflow
numpy
math
