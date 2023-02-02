from time import time, sleep
import psutil
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import psutil
import matplotlib as mpl
import streamlit_toggle as tog
import string
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from scipy.constants import gravitational_constant
from scipy.optimize import curve_fit
import streamlit_toggle as tog
import string


##### DEVMOD ##########     this determines whether
devmod = False #########     or not we profile, i.e 
############# DEVMOD ##    measure func performance 

def function_profiler_on(f, *args, **kwargs):
    def wrapper(*args, **kwargs):
        start_time = time()
        start_cpu = psutil.cpu_percent()
        start_ram = psutil.virtual_memory()

        result = f(*args, **kwargs)
        
        end_time = time()
        end_cpu = psutil.cpu_percent()
        end_ram = psutil.virtual_memory()
        print('CPU usage: ' + str(end_cpu - start_cpu) + '%')
        print('RAM usage: ' + str(end_ram.percent - start_ram.percent) + '%')
        #print('GPU usage: ' + str(end_gpu - start_gpu) + ' bytes')
        print('Time: ' + str(end_time - start_time) + ' seconds')

        # I wanna save or add this data to a csv file
        
        data = {
            'datetime' : datetime.today(),
            'func_name' : str(f).split()[1],
            'run_time' : end_time - start_time,
            #'start_ram' : start_ram._asdict(),
            #'end_ram' : end_ram._asdict(),
            'start_cpu' : [start_cpu],
            'end_cpu': [end_cpu],
        }
        for key, val in start_ram._asdict().items():
            data['start_ram_' + key] = [val]

        for key, val in end_ram._asdict().items():
            data['end_ram_' + key] = [val]


        try:            
            new_line = pd.DataFrame.from_dict(data)
            df = pd.read_csv('profiling_data.csv', index_col=0)
            df = pd.concat([df, new_line], axis=0)
            df.to_csv('profiling_data.csv')

        except:
            print('profiling logger failed')
            df = pd.DataFrame.from_dict(data)
            df.to_csv('profiling_data.csv')

        
        return result
    return wrapper

def function_profiler_off(f, *args, **kwargs):
    def wrapper(*args, **kwargs):
        
        result = f(*args, **kwargs)
        
        
        return result
    return wrapper

if devmod:
    function_profiler = function_profiler_on
else:
    function_profiler = function_profiler_off


# fig counter
fig_counter = np.array([0])
fig_counter[0] = 0


def set_rcParams(style_dict = {
        'patch.facecolor' : (0.40, 0.65, .1),
        'axes.facecolor' : (0.04, 0.065, 0.03),
        'figure.facecolor' : (.2, .05, 0.3),
        'xtick.color' : 'white',
        'ytick.color' : 'white',
        'text.color' : 'white',
        # 'axes.grid' : True,  # should we?,
        'figure.autolayout' : True,  # 'tight_layout',
        'axes.labelcolor' :  "lightgreen",
                    }):
	"""
		setting matplotlib style
	"""
	# extra function
	for key, val in style_dict.items():

		mpl.rcParams[key] = val

def strip_leading_spaces(x):
    x_stripped = x
    if len(x)<2: return x
    for i in range(12):
        try:
            if x_stripped[0] == ' ':
                x_stripped = x_stripped[1:]
            else:
                break
        except:
            break
    return x_stripped

def strip_lines(text):
    return'\n'.join([strip_leading_spaces(x) for x in text.split('\n')])


def wrapfig(width=200, text='aaa',src='', st=st):
        
    
    HEAD = """<!DOCTYPE html>
        <html>    
        <head><style>
            img {
            float: right;
            margin: 5px;
            }
        </style></head>
        """
    
    BODY = """
        <body>
        <div class="square">
            <div>
            <img src="{}"
                width = {}
                alt="Longtail boat in Thailand">
            </div>
        <p>{}</p>
        </div>
        </body></html>
        """.format(src, width, text)

    str = HEAD+BODY
    str = strip_lines(str)
    st.markdown(str,  unsafe_allow_html=True
    )
    
def gauss_pdf_N(x, mu, sigma):
    """Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)


def getText_prep(filename = 'assets/inverse_problems/text/'+'week1.md', split_level = 2):
    with open(filename,'r', encoding='utf8') as f:
        file = f.read()
    level_topics = file.split('\n'+"#"*split_level+' ')
    text_dict = {i.split("\n")[0].replace('### ','') : 
                "\n".join(i.split("\n")[1:]) for i in level_topics}
    
    return text_dict  


def getText_prep_new(filename = 'assets/applied_statistics/text/'+'week3.md'):
    with open(filename) as f:
        file = f.read().split('KEY: ')[1:]
    d = {}
    for i in file:
        key = i.split('\n')[0]
        d[key] = i[len(key)+1:]       
    return d





def  makeFunc_dict(filename='utils/utils_appstat.py'):
    with open(filename) as f:
        lines = f.read().split('\n')
    # start marked by def
    # end marked by no indent
    func_now = False
    key = None
    func_dict = {}
    for line in lines:#[150:220]:
        #st.write(func_now, line)
        
        if ("def" == line[:3]):# and (func_now == False):
            #st.write(line)
            # look for def to start function
            func_now = True
            if key != None:
                # stitch
                func_dict[key] = '\n'.join(func_dict[key])
            key = line.split('def')[1].split("(")[0]
            func_dict[key] = [line]

        elif (func_now == True) and (key != None ):
            if len(line.strip())==0:
                # if line is empty, append it
                func_dict[key].append(line)

            elif (line[:1] == '\t' or line[:4] == ' '*len(line[:4])) :
                func_dict[key].append(line)
            
            else:
                
                # end func
                func_now = False

        else: pass
            #print(line[:2])
    func_dict[key] = '\n'.join(func_dict[key])
    func_dict_core = {}
    func_dict_extras = {}
    for func in func_dict:
        #st.code(func_dict[func])
        if '# extra function' in func_dict[func]: func_dict_extras[func] = func_dict[func]
        else: func_dict_core[func] = func_dict[func]

    # omg, this was a hassel
    return func_dict_core, func_dict_extras


def  figNum():
    # extra function
    fig_counter[0] += 1
    return f"Figure {fig_counter[0]}: "


def  caption_figure(text, st=st):
    # extra function
    st.caption('<div align="center">' +figNum()+ text+'</div>' , unsafe_allow_html=True)

## from teachers

def  format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    # extra function
    if isinstance(value, (float, np.float)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'

 
def  values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    # extra function
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res

 
def  len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    # extra function
    return len(max(s, key=len))

 
def  nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    # extra function
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def  add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    # extra function
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None

