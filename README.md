# Learning State Representations via Retracing in Reinforcement Learning

Python code base for the **Cycle-Consistency World Model** agent in our [paper](https://arxiv.org/abs/2111.12600).

The code is developed based on the TF2 code of [Dreamer v1](https://github.com/danijar/dreamer).

If you find the code and our paper useful, please cite us in the following format:
```
@inproceedings{yu2022learning,
    title={Learning State Representations via Retracing in Reinforcement Learning},
    author={Changmin Yu and Dong Li and Jianye Hao and Jun Wang and Neil Burgess},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=CLpxpXqqBV}
}
```

### Installation and Dependencies

The code is based on python 3.7, the necessary dependencies can be installed by running:

```
conda env create -f environment.yml
```

### Training

```
CUDA_VIDIBLE_DEVICES=0 python ccwm.py --task dmc_walker_walk
```

### Contact

Please feel free to use/extend this code for your own research. Please send any enquiry to <changmin.yu.19@ucl.ac.uk> or simply open an issue.