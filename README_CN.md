# UniTracker Lab

[English](README.md)

用于实现 UniTracker 环境与学习算法的 Isaac Lab 项目。

## 环境要求

- Linux
- NVIDIA GPU，以及与 Isaac Sim 兼容的驱动
- 推荐 Python 3.11（支持 Python 3.10 及以上版本）
- Isaac Sim 5.1.0
- Isaac Lab v2.3.2
- RSL-RL v3.1.2（以 Git submodule 引入）

## 环境配置

克隆项目及 RSL-RL submodule：

```bash
git clone --recurse-submodules https://github.com/LeiHu3313/unitracker_lab.git
cd unitracker_lab
```

如果克隆时没有拉取 submodule：

```bash
git submodule update --init --recursive
```

激活 Isaac Lab 使用的 Python 环境，然后安装本项目和本地 RSL-RL：

```bash
python -m pip install -e source/unitracker_lab
python -m pip install -e rsl_rl
```
