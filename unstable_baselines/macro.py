# --- built in ---
import os
import gym
import sys
import time
import math
import logging
import inspect

# --- 3rd party ---
import numpy as np

# --- my module ---
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.schedules import LinearSchedule


def lcs(s1, s2):

    table = {}

    for i in range(0, len(s1)):
        for j in range(0, len(s2)):
            if s1[i] == s2[j]:
                table[(i, j)] = table.get((i-1, j-1), 0) + 1
            else:
                table[(i, j)] = max(table.get((i-1, j), 0), table.get((i, j-1), 0))

    return table[(len(s1)-1, len(s2)-1)]



def pick_macros(A, l, omega, max_num):
    '''
    pick macros

    Args:
        A: (list) action trajectory
        l: (int) macro length
        omega: (float) ratio
        max_num: (int) the max number of M
    '''

    # occurrence
    O = {}

    # count occurrence
    for i in range(len(A)-l):
        O[ tuple(A[i:i+l]) ] = O.get( tuple(A[i:i+l]), 0 ) + 1
    
    # rank
    O_sorted = sorted(O.keys(), key=lambda x: O[x], reverse=True)
    # initialize M
    M = [O_sorted[0]]

    for o in O_sorted[1:]:
        # exceed max number
        if len(M) >= max_num: break

        if all(lcs(o, m) < omega * l for m in M):
            M.append(o)

    return M



def evaluate(model, env, timesteps):
    obs = env.reset()

    scores = []
    current_score = 0
    traj = []
    for i in range(timesteps):
        act, _ = model.predict(obs)
        obs, reward, done, info = env.step(act, decay=False)
        
        current_score += reward
        traj.append(act)

        if done: 
            scores.append(current_score)
            current_score = 0
            obs = env.reset()

    return scores, traj


def extend_traj(model, traj):
    macro = model.env.M

    full_traj = []

    for act in traj:
        if act < model.primitive_action_space_size:
            full_traj.append(act)
        else:
            full_traj.extend(macro[action - model.primitive_action_space_size])

    return full_traj



def _callback(model, eval_env, update_macros, sub_callback=None):

    def callback(l, g):
        '''
        stable baselines learning callback

        Args:
            l: locals() from stable baselines
            g: globals() from stable baselines
        
        Return:
            (bool) whether to interrupt learning process
        '''

        current_timesteps = l['_']

        if current_timesteps in update_macros:

            # evaluating model
            
            scores, traj = evaluate(eval_env, model.eval_timesteps)
            full_traj = extend_traj(model, traj)

            # genereate new macros

            macros = pick_macros(full_traj, self.macro_length, self.omega, self.macro_num)

            # update macros
            model.env.update_macros(macros)

            if hasattr(model, 'exploration'):
                if isinstance(model.exploration, LinearSchedule):
                    model.exploration.reset_epsilon(current_timesteps)

        if sub_callback is not None:
            if sub_callback(locals(), globals()) is False:
                return False

        return True

    return callback


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == 'update_macros:'
                env.skills = data
                remote.send('OK')
            else:
                raise NotImplementedError
        except EOFError:
            break


class MySubprocVecEnv(SubprocVecEnv):
    def update_macros(self, M):
        self.M = M
        for remote in self.remotes:
            remote.send(('update_macros', M))

        for remote in self.remotes:
            remote.recv()


class MyDummyVecEnv(DummyVecEnv):
    def __init__(self, *args, **kwargs):
        super(DummyVecEnv, self).__init__(*args, **kwargs)

        self.M = self.envs[0].skills

    def update_macros(self, M):
        self.M = M
        for env in self.envs:
            env.skills = M

class MyLinearSchedule(MyLinearSchedule):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.start_step = 0

    def reset_epsilon(self, step):
        self.start_step = step

    def value(self, step):
        fraction = min(float(step-self.start_step) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


# class MacroWrapper(gym.Wrapper):

#     _Macro = None

#     def __init__(self, env, macro_num=None, gamma=1.0):


#         self._old_action_space = env.action_space
#         self._gamma = gamma

#         assert isinstance(self._old_action_space, gym.spaces.Discrete), 'action space must be Discrerte'

#         if macro_num is None:
#             self.macro_num = self._old_action_space.n

#         type(self)._Macro = [[0] for _ in self.macro_num]

#         self.action_space = gym.spaces.Discrete(self._old_action_space.n + self.macro_num)


#         super().__init__(env)

#     @classmethod
#     def update_macros(cls, M):
#         cls._Macro = M

#     def step(self, action, decay=True):
#         if action < self._old_action_space.n:
#             return self.env.step(action)
#         else:
#             macro = type(self)._Macro[action - self._old_action_space.n]
#             total_reward = 0.0
#             for idx, act in enumerate(macro):
#                 obs, reward, done, info = self.env.step(act)
#                 _decay = math.pow(self._gamma, idx) if decay else 1.0
#                 total_reward += _decay * reward
#                 if done:
#                     break

#             return obs, total_reward, done, info


#     def extend_traj(self, traj):
#         full_traj = []

#         for act in traj:
#             if act < self._old_action_space.n:
#                 full_traj.append(act)
#             else:
#                 full_traj.extend(type(self)._Macro[action - self._old_action_space.n])

#         return full_traj
    



def Macro(algo, *args, macro_length=5, macro_num=None, **kwargs):

    # create attribute
    if not hasattr(Macro, 'setup_stable_baselines'):
        Macro.setup_stable_baselines = False

    # initialize stable baselines
    if not Macro.setup_stable_baselines:
        import stable_baselines as stb

        stb.common.vec_env.subproc_vec_env._worker = _worker
        stb.common.vec_env.DummyVecEnv = MyDummyVecEnv
        stb.common.vec_env.SubprocVecEnv = MySubprocVecEnv
        stb.common.schedules.LinearSchedule = MyLinearSchedule

        Macro.setup_stable_baselines = True


    class _Macro(algo):
        def __init__(self, *_args, **_kwargs):
            
            super(_Macro, self).__init__(*_args, **_kwargs)

            self.macro_length = macro_length
            self.macro_num = self.action_space.skill_n if macro_num is None else macro_num
            self.primitive_action_space_size = self.action_space.primitive_action_n
            

        def learn(self, *_args, eval_env=None, 
                               eval_timesteps=200000, 
                               timesteps_per_epoch=1000000, 
                               omega=0.8, 
                               update_macros=[6, 13, 25, 50], **_kwargs):

            self.omega = omega
            self.eval_timesteps = eval_timesteps
            self.update_macros = update_macros

            my_callback = _callback(self, eval_env, [i*timesteps_per_epoch for i in update_macros])

            super(_Macro, self).learn(*_args, callback=my_callback, **_kwargs)

            return self

    return _Macro(*args, **kwargs)




if __name__ == '__main__':
    assert lcs('567567avbhjgcutydiyte111222f', 'abcpopopopodpopef666') == 6
