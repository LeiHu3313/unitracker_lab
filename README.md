# UniTracker Lab

[中文说明](README_CN.md)

Isaac Lab project for  UniTracker environments and learning algorithms.

## Environment

- Linux
- NVIDIA GPU and a driver compatible with Isaac Sim
- Python 3.11 recommended (Python 3.10 or newer is supported)
- Isaac Sim 5.1.0
- Isaac Lab v2.3.2
- RSL-RL v3.1.2 (included as a Git submodule)

## Setup

Clone the project with its RSL-RL submodule:

```bash
git clone --recurse-submodules https://github.com/LeiHu3313/unitracker_lab.git
cd unitracker_lab
```

If the project was cloned without submodules:

```bash
git submodule update --init --recursive
```

Activate the Python environment used by Isaac Lab, then install the project and its local RSL-RL package:

```bash
python -m pip install -e source/unitracker_lab
python -m pip install -e rsl_rl
```
