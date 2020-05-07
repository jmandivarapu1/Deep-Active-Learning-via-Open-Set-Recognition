"""
Stand alone evaluation script for open set recognition and plotting of different datasets

Uses the same command line parser as main.py

The attributes that need to be specified are the number of variational samples (should be greater than one if prediction
uncertainties are supposed to be calculated and compared), the architecture type and the resume flag pointing to a model
checkpoint file.
Other parameters like open set distance function etc. are optional.

example usage:
--resume /path/checkpoint.pth.tar --var-samples 100 -a MLP
"""

import collections