#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym, sys
import numpy as np

from agent import Agent

def main(env_name, render=False, monitor=True, load=False, seed=0):

    env = gym.make(env_name)
    view_path = "./video/" + env_name
    model_path = "./model/" + env_name + "_"

    n_st = env.observation_space.shape[0]
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        # CartPole-v0, Acrobot-v0, MountainCar-v0
        n_act = env.action_space.n
        action_list = range(0, n_act)
    elif type(env.action_space) == gym.spaces.box.Box:
        # Pendulum-v0
        action_list = [np.array([a]) for a in [-2.0, 2.0]]
        n_act = len(action_list)

    agent = Agent(n_st, n_act, seed)
    if load:
        agent.load_model(model_path)

    if monitor:
        env.monitor.start(view_path, video_callable=None, force=True, seed=seed)
    for i_episode in xrange(1000):
        observation = env.reset()
        r_sum = 0
        q_list = []
        for t in xrange(200):
            if render:
                env.render()
            state = observation.astype(np.float32).reshape((1,n_st))
            act_i, q = agent.get_action(state)
            q_list.append(q)
            action = action_list[act_i]
            observation, reward, ep_end, _ = env.step(action)
            state_dash = observation.astype(np.float32).reshape((1,n_st))
            agent.stock_experience(state, act_i, reward, state_dash, ep_end)
            agent.train()
            r_sum += reward
            if ep_end:
                break
        print "\t".join(map(str,[i_episode, r_sum, agent.epsilon, agent.loss, sum(q_list)/float(t+1) ,agent.step]))
        agent.save_model(model_path)
    if monitor:
        env.monitor.close()

if __name__=="__main__":
    env_name = sys.argv[1]
    main(env_name)
