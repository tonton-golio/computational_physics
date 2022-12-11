import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd    
from time import time
from time import sleep
import matplotlib as mpl
from scipy.constants import gravitational_constant


def set_rcParams():
	"""
		setting matplotlib style
	"""
	# extra function
	mpl.rcParams['patch.facecolor'] = (0.04, 0.065, 0.03)
	mpl.rcParams['axes.facecolor'] = 'grey'
	mpl.rcParams['figure.facecolor'] = (0.04, 0.065, 0.03)
	mpl.rcParams['xtick.color'] = 'white'
	mpl.rcParams['ytick.color'] = 'white'
	# mpl.rcParams['axes.grid'] = True  # should we?
	mpl.rcParams['figure.autolayout'] = True  # 'tight_layout'

set_rcParams()

# General

text_path = 'assets/inverse_problems/text/'

st.set_page_config(page_title="plt contribute", 
    page_icon="🧊", 
    #layout="wide", 
    initial_sidebar_state="collapsed", 
    menu_items=None)

def getText_prep(filename = text_path+'week1.md', split_level = 2):
    with open(filename,'r', encoding='utf8') as f:
        file = f.read()
    level_topics = file.split('\n'+"#"*split_level+' ')
    text_dict = {i.split("\n")[0].replace('### ','') : 
                "\n".join(i.split("\n")[1:]) for i in level_topics}
    
    return text_dict  

def template():
    st.title('')

    # Main text
    text_dict = getText_prep(filename = text_path+'bounding_errors.md', split_level = 2)

    st.markdown(text_dict["Header 1"])
    st.markdown(text_dict["Header 2"])
    with st.expander('Go deeper', expanded=False):
        st.markdown(text_dict["Example"])


# Week 1

def G_ij(zi, xj): 
    return gravitational_constant * np.log( ((zi+1)**2 + xj**2) / ( zi**2 + xj**2 ) )

def G_matrix(xs, zs):
    G = np.array([[G_ij(z, x) for z in zs] for x in xs])
    return G


def contour_of_G(G, xlabel='$x$-position', ylabel='depth'):
    fig, ax = plt.subplots( figsize=(5,3))
    CS = ax.contourf(G, 10, cmap=plt.cm.bone)

    CS2 = ax.contour(CS, levels=CS.levels[::2], colors='r')

    ax.set_title(r'$G$')
    ax.set_ylabel(ylabel, color='black')
    ax.set_xlabel(xlabel, color='black')

    cbar = fig.colorbar(CS)
    cbar.add_lines(CS2)
    plt.gca().invert_yaxis()

    fig.set_facecolor('lightgray')
    plt.tight_layout()
    plt.close()
    return fig 

def getParams(G, d_obs, eps_space = np.logspace(-12, -10, 200)):
    ms = []
    
    for epsilon in eps_space:
        m_e = np.linalg.inv(G.T@G + epsilon**2 * np.eye(100) ) @  (G.T @d_obs)

        ms.append(m_e)

    return np.array(ms)


def find_minimum(G, ms, d_obs, 

    eps_space,
    data_error = [10**(-9)] * 18,
    data_error1 = [10**(-8)] * 18):

    result = [ abs( np.linalg.norm(G @ m - d_obs) -
                 np.linalg.norm(data_error) 
               ) for m in ms ]

    result1 = [ abs( np.linalg.norm(G @ m - d_obs) -
                 np.linalg.norm(data_error1) 
               ) for m in ms ]

    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(eps_space, result, label='data error = 10e-9')
    ax.plot(eps_space, result1, label='data error = 10e-8')

    min_eps = eps_space[np.argmin(result)]
    min_eps1 = eps_space[np.argmin(result1)]
    ax.axvline(min_eps,ls='--',label=f'minimum = {round(min_eps,14)}')
    ax.axvline(min_eps1,ls='--',c='orange', label=f'minimum = {round(min_eps1,14)}')

    ax.set_xlabel(r'$\epsilon$', color='black')
    ax.set_ylabel(r'Deviation', color='black')

    ax.set(
            xscale='log', yscale='log', 
            #xlim=(1e-15, 1e-11)
        )

    ax.legend(facecolor='beige')
    ax.set_facecolor('lightgray')
    fig.set_facecolor('lightgray')

    plt.close()
    return fig


