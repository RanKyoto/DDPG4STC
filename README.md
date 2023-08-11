# DDPG4STC
Ran Wang 2023/02/08 Kyoto University

Contains supplementary simulation code for the work:
```
Ran Wang & Kenji Kashima (2023)
Deep reinforcement learning for continuous-time self-triggered control with experimental evaluation,
Advanced Robotics, DOI: 10.1080/01691864.2023.2229886
```
## Bibtex
```
@article{ranDDPG4STC2023,
author = {Ran Wang and Kenji Kashima},
title = {Deep reinforcement learning for continuous-time self-triggered control with experimental evaluation},
journal = {Advanced Robotics},
volume = {37},
number = {16},
pages = {1012-1024},
year  = {2023},
publisher = {Taylor & Francis},
doi = {10.1080/01691864.2023.2229886},
URL = {https://doi.org/10.1080/01691864.2023.2229886}
}
```

# Setup
## DownloadZip or use git conmand:
```
git clone https://github.com/RanKyoto/DDPG4STC.git 
```
## Install required python packages
```
pip install panda-gym
pip install gym==0.21.0
pip install numpy=1.23.5 --upgrade
pip install tensorflow-gpu
pip install matplotlab
```

# Play with DDPG4STC
## DDPG4STC with a rotary inverted pendulum
See "main.py"

## DDPG4STC with panda-gym
See "main_panda.py"
