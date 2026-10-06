# Dynamic Traffic Light Control with Multi‑Agent RL

## Overview
This repository implements a **multi‑agent reinforcement learning (RL)** framework for adaptive traffic signal control in simulated road networks.  
Each intersection is modeled as an independent agent that learns to minimize vehicle waiting time, queue length, and overall travel time using **Proximal Policy Optimization (PPO)**. The environment is built on **Gymnasium** and follows the OpenAI Gym API, making it easy to plug in alternative agents or RL algorithms.

Key highlights:
- **Decentralized control**: each traffic light runs its own policy network while sharing a common replay buffer for coordinated learning.
- **Scalable simulation**: supports arbitrary grid sizes and custom traffic demand patterns.
- **Extensive visualisation**: Matplotlib‑based plots and optional animation of vehicle movements.
- **Fully reproducible**: seeds, logging, and checkpointing are handled out‑of‑the‑box.

## Tech Stack
- **Python 3.9+**
- **PyTorch** – neural network training
- **Gymnasium** – environment interface
- **NumPy** – numerical operations
- **Matplotlib** – plotting and animation
- **tqdm** – progress bars
- **tensorboard** – training diagnostics (optional)

## Project Structure
```
dynamic-traffic-light-rl/
├── env/                     # Custom Gymnasium environment
│   ├── __init__.py
│   └── traffic_env.py
├── agents/                  # Multi‑agent PPO implementation
│   ├── __init__.py
│   ├── ppo_agent.py
│   └── buffer.py
├── utils/                   # Helper functions (visualisation, logging)
│   ├── __init__.py
│   └── plot.py
├── configs/                 # YAML configuration files
│   └── default.yaml
├── scripts/                 # CLI entry points
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── requirements.txt
├── README.md                # ← you are here
└── LICENSE
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dynamic-traffic-light-rl.git
   cd dynamic-traffic-light-rl
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import gymnasium, torch, numpy; print('All core packages imported successfully')"
   ```

## Quick Start

### 1. Train a multi‑agent PPO controller
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --seed 42 \
    --log_dir runs/experiment_01
```
- `--config` points to a YAML file with hyper‑parameters (learning rate, gamma, network sizes, etc.).
- Training checkpoints are saved in `runs/experiment_01/checkpoints/`.
- TensorBoard logs are written to `runs/experiment_01/tb/`.

### 2. Evaluate a trained policy
```bash
python scripts/evaluate.py \
    --checkpoint runs/experiment_01/checkpoints/ckpt_500.pth \
    --episodes 20 \
    --render
```
- `--render` opens a Matplotlib window that animates traffic flow for each episode.
- The script prints average episode reward, mean waiting time, and average queue length.

### 3. Visualise training curves
```bash
python scripts/visualize.py \
    --log_dir runs/experiment_01/tb/ \
    --output plots/training_curves.png
```
The generated PNG contains reward, loss, and entropy trajectories for all agents.

## Detailed Usage

### Configuration (`configs/default.yaml`)
```yaml
env:
  grid_size: [4, 4]          # rows x columns of intersections
  episode_length: 3600       # steps per episode (seconds)
  traffic_pattern: "rush_hour"
  seed: 12345

ppo:
  num_agents: 16
  hidden_sizes: [128, 128]
  lr: 3e-4
  gamma: 0.99
  lam: 0.95
  clip_eps: 0.2
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 0.5
  rollout_len: 2048
  epochs: 10
  minibatch_size: 256

training:
  total_timesteps: 2_000_000
  log_interval: 1000
  checkpoint_interval: 50000
```
Adjust any field via the command line, e.g., `--config configs/default.yaml --env.grid_size 6 6`.

### Environment API
```python
import gymnasium as gym
from env.traffic_env import TrafficLightEnv

env = TrafficLightEnv(grid_size=(4, 4), episode_length=3600, traffic_pattern="random")
obs, info = env.reset(seed=42)

# Step loop
action = env.action_space.sample()   # replace with agent policy
next_obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```
- **Observation**: a dictionary containing per‑intersection vehicle counts, current phase, and time‑to‑green.
- **Action**: a vector of phase indices (one per intersection).

### Training Loop (high‑level)
```python
from agents.ppo_agent import MultiAgentPPO
from utils.plot import plot_training

ppo = MultiAgentPPO(env, config_path="configs/default.yaml")
ppo.train(total_timesteps=2_000_000)

plot_training(ppo.logger, save_path="plots/training_curves.png")
```
All heavy lifting (advantage estimation, policy/value updates, gradient clipping) lives inside `MultiAgentPPO`.

## Logging & Monitoring
- **TensorBoard**: `tensorboard --logdir runs/experiment_01/tb/`
- **CSV logger**: `ppo.logger.save_csv("runs/experiment_01/metrics.csv")`
- **Checkpoints**: automatically stored as `ckpt_{step}.pth`.

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/new‑algorithm`).
3. Ensure code passes linting (`flake8 .`) and tests (`pytest -q`).
4. Submit a pull request with a clear description of changes.

## License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.

## References
- Schulman, J. et al., *Proximal Policy Optimization Algorithms*, 2017.  
- Krajzewicz, D. et al., *SUMO – Simulation of Urban MObility*, 2012.  
- OpenAI Gymnasium documentation: https://gymnasium.farama.org  

---  

*Happy coding! 🚦*