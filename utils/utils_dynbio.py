from time import time
from time import sleep

import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd    
from scipy.integrate import odeint

# General

text_path = 'assets/dynamical_models/text/'

# setting matplotlib style:
#mpl.rcParams['patch.facecolor'] = (0.04, 0.065, 0.03)
#mpl.rcParams['axes.facecolor'] = (0.04, 0.065, 0.03)
#mpl.rcParams['figure.facecolor'] = 'gray'#(0.04, 0.065, 0.03)
#mpl.rcParams['xtick.color'] = 'white'
#mpl.rcParams['ytick.color'] = 'white'
#mpl.rcParams['figure.autolayout'] = True  # 'tight_layout'
#mpl.rcParams['axes.grid'] = True  # should we?


st.set_page_config(page_title="Dynamical Models", 
    page_icon="🧊", 
    layout="wide", 
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


def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    
    if isinstance(value, (float, np.float)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None

###########################################
# week 1

# week 2
## plot function for Hill function
def plot_hill_function(threshold, coeff, activation=True):
    x = np.linspace(0, 2, 1000)

    if activation:
        y = 1.0 / (1.0 + (x/threshold)**coeff)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.axhline(0.5, c='gray', ls='--')
        ax.axvline(threshold, c='gray', ls='--')
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        ax.set_title("Hill function for repression")
        ax.set_xlabel("Concentration of TF $c_\mathrm{TF}$")
        ax.set_ylabel("Value of Hill function")

    else :
        y = (x/threshold)**coeff / (1.0 + (x/threshold)**coeff)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.axhline(0.5, c='gray', ls='--')
        ax.axvline(threshold, c='gray', ls='--')
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        ax.set_title("Hill function for activation")
        ax.set_xlabel("Concentration of TF $c_\mathrm{TF}$")
        ax.set_ylabel("Value of Hill function")
    sns.despine()
    return fig

## Solving ODE
def srna_simulation():
    def model(x, t, k_mRNA, g_mRNA, k_pro, g_pro, k_sRNA, g_sRNA, delta):
        # x[0] is the concentration of mRNA,m, and 
        # x[1] is the concentration of protein, p.
        # x[2] is the concentration of sRNA, s.
        dmdt = k_mRNA - g_mRNA*x[0] - delta*x[0]*x[2]
        dpdt = k_pro*x[0] - g_pro*x[1]
        dsdt = k_sRNA - g_sRNA*x[2] - delta*x[0]*x[2]
        dxdt = [dmdt, dpdt, dsdt]
        return dxdt
        
    # parameters
    k_mRNA  = 1.
    g_mRNA  = 5.
    k_pro   = 50.
    g_pro   = 1.
    k_sRNA   = 0.01
    g_sRNA   = 1.
    delta = 100.

    # simulation length
    t_final=10
    # data point
    t = np.linspace(0, t_final, 100)

    # initial state
    mRNA_initial    = 0.
    protein_initial = 0.
    sRNA_initial = 0.


    # initail state as array
    x_initial = [mRNA_initial, protein_initial, sRNA_initial]

n
# integrate the ordingary differential equation, it returns the array of solution in x_solution
    x_solution = odeint(model, x_initial, t, 
                        args=(k_mRNA, g_mRNA, k_pro, g_pro, k_sRNA, g_sRNA, delta))

n
# plot results
    fig, ax = plt.subplots()
    ax.plot(t, x_solution[:,0], ls = '-',label=r'mRNA')
    ax.plot(t, x_solution[:,1], ls = '--',label=r'protein')
    ax.plot(t, x_solution[:,2], ls = '--',label=r'sRNA')
    ax.xlabel('time')
    ax.ylabel('response')
    ax.legend(frameon=False)
    sns.despine()
    return fig






