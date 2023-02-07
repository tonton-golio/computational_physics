from utils.utils_global import *
from celluloid import Camera
import gymnasium as gym
import math
import random
import matplotlib


from collections import namedtuple, deque
from itertools import count

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from torchvision import transforms as T
from PIL import Image
from pathlib import Path
from collections import deque
import random, datetime, os, copy



def explore(restaurants):
    r = np.random.choice(list(restaurants.keys()))
    
    (mu, sig) = restaurants[r]
    score = np.random.normal(mu, sig, None)
    return r, score

def exploit(restaurants, score_table, method='standard'):
    def where2go(score_table, method='standard'):
        if method == 'standard':
            mean_score = np.array([(key , score_table[key]['sum']/score_table[key]['count']) for key in score_table])

            r = mean_score[np.argmax(mean_score[:, 1].astype(float)),0]

        elif method == 'UCB':
            t = sum(score_table[key]['count'] for key in score_table.keys())
            #print('t=',t)
            mean_score = np.array([(key , 
                score_table[key]['sum']/score_table[key]['count'] + np.sqrt(t/score_table[key]['count'])) for key in score_table])
            
            r = mean_score[np.argmax(mean_score[:, 1].astype(float)),0]
        else:
            raise ValueError('method not recognized')
        return r

    if len(score_table) == 0: return explore(restaurants)
    elif len(score_table) == 1: r = list(score_table.keys())[0]
    else: r = where2go(score_table, method)
        
    (mu, sig) = restaurants[r]
    score = np.random.normal(mu, sig, None)
    return r, score

def bandit(restaurants, n=300, epsilon=10, method='standard'):
    score_table = {}
    for i in range(n):
        rand_num = np.random.uniform(0,1)
        if rand_num < epsilon/100:

            r, score = explore(restaurants)
            if r in score_table:
                score_table[r]['count'] += 1
                score_table[r]['sum']   += score
            else:
                score_table[r] = {'count' : 1, 'sum': score}

        else:
            r, score = exploit(restaurants, score_table, method)

            if r in score_table:
                score_table[r]['count'] += 1
                score_table[r]['sum']   += score
            else:
                score_table[r] = {'count' : 1, 'sum': score}

    total_score = sum(score_table[key]['sum'] for key in score_table.keys())
    return total_score

def many_bandit_runs(restaurants, n=300, n_epsilons=10, n_exp=2,  method='standard'):
    scores = []
    epls = np.logspace(0,2,n_epsilons, dtype=int)
    for epl in tqdm(epls):
        scores_tmp = []
        for i in range(n_exp):
            scores_tmp.append(bandit(restaurants, n, epl, method))
        scores.append(scores_tmp)
    return np.array(scores), epls

def show_bandit_scores(scores, epls):
    fig = plt.figure(1)
    plt.errorbar(epls, np.mean(scores, axis=1), np.std(scores, axis=1), lw=0, elinewidth=1)
    plt.xscale('log')
    plt.xlabel('epsilon')
    plt.ylabel('total score')
    plt.title(r'$\epsilon$-greedy; striking a balance between exploration and exploitation')
    plt.show()
    plt.close()
    return fig
