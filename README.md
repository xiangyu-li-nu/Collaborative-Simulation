# Collaborative Simulation

The code repository for the paper *"Collaborative Traffic Signal Control and Path Recommendations Under PM2.5 Exposure Based on Reinforcement Learning."* The project explores reinforcement learning (RL) algorithms to optimize urban traffic signal control, comparing their performance against traditional methods like the Max Pressure (MP) algorithm.

This repository includes implementations of the Deep Q-Learning Network (DQN) and Multi-Agent Advantage Actor-Critic (MA2C) algorithms in traffic simulation environments. It aims to reduce traffic congestion, average waiting times, and CO2 emissions using reinforcement learning techniques.

## Authors

- **Xiangyu Li**, a second-year Ph.D. student in Transportation Engineering at Northwestern University.  
  **Email:** [xiangyuli2027@u.northwestern.edu](mailto:xiangyuli2027@u.northwestern.edu)

## Repository Structure

### **DQN**

This folder contains the implementation of the Deep Q-Learning Network for traffic signal control:

- `cs.py`: Core script for running simulations and configuring the DQN agents.
- `main.py`: The main script to start and manage the traffic simulation.
- `map_config.py`: Defines the traffic network and simulation configuration.
- `multi_signal.py`: Handles multi-signal interaction for traffic control.
- `rewards.py`: Defines the reward function for the DQN agents.
- `signal_config.py`: Configures traffic signal settings.
- `states.py`: Manages state representation for reinforcement learning.
- `traffic_signal.py`: Implements the logic for controlling traffic signals in SUMO.
- `Evaluation.py`: Evaluates the performance of the trained DQN model.
- `Data recording.csv`: Example output data recorded from simulations.

### **ma2c**

This folder contains the implementation of the Multi-Agent Advantage Actor-Critic (MA2C) algorithm:

- `agents/`: Implements multi-agent learning logic.
- `environments/cologne3/`: Contains the SUMO traffic environment setup.
- `logs/`: Stores training and evaluation logs.
- `Evaluation.py`: Evaluates the performance of the trained MA2C model.
- `agent_config.py`: Configures MA2C agent parameters.
