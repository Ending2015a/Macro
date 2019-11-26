import os
import sys
import time
import logging

import gym
from unstable_baselines import Macro
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy

sys.path.append(os.path.abspath("./stable-baselines/stable_baselines"))
sys.path.append(os.path.abspath("../../lib"))

from env_wrapper import SkillWrapper, ActionRemapWrapper
from stable_baselines.common.vec_env import VecFrameStack
from cmd_util import make_atari_env


ENV_ID = "SeaquestNoFrameskip-v4"
SEED = 1000
TEMP_LOGDIR = "./"
env_id = ENV_ID
skills = [[2,2,2],[1,1,1]]
env_creator_ = lambda env:ActionRemapWrapper(env)
env_creator = lambda env:SkillWrapper(env_creator_(env), skills=skills)
env = VecFrameStack(make_atari_env(env_id, 1, SEED, extra_wrapper_func=env_creator, logdir=TEMP_LOGDIR, wrapper_kwargs={"episode_life":False, "clip_rewards":False}), 4)


model = Macro(PPO2, CnnPolicy, env, verbose=1, macro_length=3, macro_num=None)

# model.act_model.set_clip_action_mask([1,1,1,1,1,0,0,0])
obs = env.reset()

model.learn(100000, eval_env=env.envs[0], eval_timesteps=2000, timesteps_per_epoch=1000)
for steps in range(5):
    action, _states = model.predict(obs)
    
    obs, rewards, dones, info = env.step(action)
    print(action)
    if bool(dones[0]) is True:
        break
