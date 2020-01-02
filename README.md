# Adaptive Network

reinforcement learning based adaptive network 


**See branch DQN-pytorch for Deep RL codes with pytorch.**


You should install following python modules.
**Dependency: pyyaml, numpy, matplotlib, networkx, pytorch**

You can install pyyaml, numpy, matplotlib, networkx library with pip install or conda install command.
You can install pytorch by using the instruction at https://pytorch.org/.

Main codes consists of four .py files.
1. main.py: main function - load hyperparameter settings from config/*.yaml file and run the main loop and plot graphs
2. env.py: network environment code
3. DQNAgent.py: The agent choose actions with regard to the environment using deep neural network
4. DQN.py: deep neural network code

