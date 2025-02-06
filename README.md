<h1>SGFT</span></h1>

This repo contains the code for:

Rapidly Adapting Policies to the Real-World via Simulation-Guided Fine-Tuning. Patrick Yin*, Tyler Westenbroek*, Simran Bagaria, Kevin Huang, Ching-An Cheng, Andrey Kolobov, Abhishek Gupta. arXiV preprint 2025.

Project Page: [https://weirdlabuw.github.io/sgft/](https://weirdlabuw.github.io/sgft/).

----


## Overview

SGFT accelerates real-world finetuning by simply relabeling the reward with potential-based shaping using the value function learned in simulation. It is a simple change on top of your favorite off-policy RL method. In this codebase, we make a simple change on top of TDMPC2 and demonstrate our method on a sim2sim DMC task.

----

## Getting started

Install dependencies via `conda` by running the following command:

```
conda env create -f docker/environment.yaml
pip install gym==0.21.0
```

### Training and Finetuning

```
$ python tdmpc2/train.py
$ python tdmpc2/train.py --config-name config_finetune sgft_checkpoint=/path/to/checkpoint
```

----

## Acknowledgments
This codebase is built upon the original work by Nicklas Hansen on TD-MPC2: Scalable, Robust World Models for Continuous Control. The original can be found at: https://github.com/nicklashansen/tdmpc2.

If you use this code, please also cite the original authors as specified in their repository:

```
@inproceedings{hansen2024tdmpc2,
  title={TD-MPC2: Scalable, Robust World Models for Continuous Control}, 
  author={Nicklas Hansen and Hao Su and Xiaolong Wang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## Citations

If you find our work useful, please consider citing our paper as follows:

```
@inproceedings{yin2025sgft,
  author    = {Yin, Patrick and Westenbroek, Tyler and Bagaria, Simran and Huang, Kevin and Cheng, Ching-An and Kolobov, Andrey and Gupta, Abhishek},
  title     = {Rapidly Adapting Policies to the Real-World via Simulation-Guided Fine-Tuning},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
```

## License

This project is derviedf from TD-MPC2, which is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.
