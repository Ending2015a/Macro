import os
import sys
import time
import logging

import gym

from stable_baselines import PPO2
from unstable_baselines import Macro, MacroWrapper



env = gym.make('SeaquestNoFrameSkip-v4')

def 

model = Macro(PPO2, )
