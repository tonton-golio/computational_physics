import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd    
from time import time
import seaborn as sns
from time import sleep
import matplotlib as mpl

# General

text_path = 'assets/applied_statistics/text/'

# setting matplotlib style:
mpl.rcParams['patch.facecolor'] = (0.04, 0.065, 0.03)
mpl.rcParams['axes.facecolor'] = (0.04, 0.065, 0.03)
mpl.rcParams['figure.facecolor'] = 'gray'#(0.04, 0.065, 0.03)
mpl.rcParams['xtick.color'] = 'white'
mpl.rcParams['ytick.color'] = 'white'
mpl.rcParams['figure.autolayout'] = True  # 'tight_layout'
# mpl.rcParams['axes.grid'] = True  # should we?


st.set_page_config(page_title="Applied Statistics", 
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

#
def means(arr, truncate=1):
	"""
	determines mean of sorts = ['Geometric mean', 'Arithmetic mean', 'Median',  'Mode', 'Harmonic', 'Truncated']
	"""
	arr = arr.copy().astype(float)
	n = len(arr)
	arr = np.sort(arr) # for median
	def geometric(arr):
		return np.prod(arr)**(1/n)
	def arithmetic(arr):
		return np.sum(arr) / n
	def median(arr):
		return arr[n//2] if n%2==0 else arr[n//2+1]
	def mode(arr):
		unique, counts = np.unique(arr, return_counts=True)
		return arithmetic(unique[np.argwhere(counts==np.max(counts))])
	def harmonic(arr):
		return (np.sum( arr**(-1) ) / n)**(-1)
	def truncated(arr):
		arr = arr[truncate:-truncate]
		return arithmetic(arr)


	return {
			'Geometric mean' : geometric(arr),
			'Arithmetic mean' :  arithmetic(arr), 
			'Median': median(arr),  
			#'Mode': mode(arr),
			'Harmonic': harmonic(arr), 
			'Truncated' : truncated(arr)
			}

def showMeans(arr, truncate=1):
	'''
		on a hist, shows means
	'''
	fig, ax = plt.subplots(figsize=(6,2))
	arr = np.sort(arr)
	d = means(arr, truncate=truncate)
	arr_truncated = arr[truncate:-truncate]
	counts, bins = np.histogram(arr_truncated)
	
	ax.hist(arr_truncated, bins=len(arr)//50)
	colors = sns.color_palette("hls", 8)
	for i, c in zip(d, colors):
		ax.axvline(d[i], label=i, c=c)
	#ax.legend(facecolor='beige')
	
	#ax.set()##xscale='log')#, yscale='log')
	_ = [plt.text(d[key], (idx+1)*max(counts)//9 , s = key, color='white') for idx, key in enumerate(d)]
	plt.close()

	return fig


def std_calculations(n=400):
	fig, ax = plt.subplots(3,1, figsize=(8,4), sharex=True, sharey=True)
	for idx, N in enumerate([5,10,20]):
		res = {'N': [], 'N-1':[]}
		for i in range(n):
			x = np.random.randn(N)
			std_N   = ( 1/(N)   * np.sum( (x-np.mean(x))**2) )**.5
			std_N_1 = ( 1/(N-1) * np.sum( (x-np.mean(x))**2) )**.5
			res['N'].append(std_N)
			res['N-1'].append(std_N_1)

		minmin = min([min(res['N']), min(res['N-1'])])
		maxmax = max([max(res['N']), max(res['N-1'])])
		shared_bins = np.linspace(minmin,maxmax,20)
		

		ax[idx].hist(res['N'], bins=shared_bins, label='N', fill=True, alpha=.6,	facecolor='pink')
		ax[idx].hist(res['N-1'],bins=shared_bins, label='N-1', fill=True, alpha=.6,		facecolor='yellow')
		ax[idx].set_title(f'N={N}', color="white")
        
		ax[idx].set(xticks=[], yticks=[])
	ax[0].legend(facecolor='beige')
	ax[idx].set_xlabel(r'standard devitation, $\sigma$', color='beige')
	fig.suptitle(f'Distributions of standard deviations, n={n}', color='beige')
	plt.tight_layout()
	plt.close()
	return fig


def roll_a_die(num=100):
    x = np.random.randint(1,7,num)
    fig, ax = plt.subplots(figsize=(5,4))
    means = [np.mean(x[:n]) for n in range(2, num)]
    
    ax.scatter(range(2, num), means, c='pink', alpha=.7)
    ax.set(xlabel='number of rolls in mean', ylabel='mean')
    plt.close()
    return fig

def roll_dice(rolls=200):
    data = {}
    nums = np.arange(2,40, dtype=int)
    for num in nums:
        data[num] = []
        for n in range(rolls):
            x = np.random.randint(1,7,num)
            data[num].append(np.mean(x))

    
    y = [np.mean(data[num]) for num in nums]
    yerr = [np.std(data[num]) for num in nums]

    fig, ax = plt.subplots(figsize=(5,4))
    ax.errorbar(nums, y, yerr, c='white')
    ax.set(xlabel='number of rolls in mean', ylabel='mean')
    plt.close()
    return fig

def makeDistibutions(n_experi = 1000,
					N_uni= 1000, 
					N_exp= 1000, 
					N_cauchy = 1000):
	# input distributions
	uni = np.random.uniform(-1,1, size=(n_experi, N_uni))*12**.5

	exp = np.random.exponential(size=(n_experi,N_exp))-1

	c = np.random.standard_cauchy(size=(n_experi,N_cauchy))
	c = np.sort(c)
	c = c[:,N_cauchy//10:-N_cauchy//10]
	
	# combined dist
	concat = np.hstack([uni.copy(), exp.copy(), c.copy()])
	
	# sums
	N = [N_uni, N_exp, N_cauchy]; N.append(sum(N))
	sums = [np.sum(dist, axis=1) / n**.5 for dist, n in zip([c,uni,exp,concat], N)]
	
	
	return sums


def plotdists(sums, N_bins = 100):
	fig, ax = plt.subplots(1,4,figsize = (8,4))
	for i, (s, name) in enumerate(zip(sums, ['uni', 'exp', 'chauchy','combined'])):
		ax[i].hist( s, bins=N_bins , histtype='step')
		ax[i].set_title(name)
		ax[i].set(yticks=[], ylim=(0,ax[i].get_ylim()[1]*1.2 ))
		
		text = {'mean':s.mean(),
			'std' : s.std(ddof=1)}
		text = nice_string_output(text, extra_spacing=2, decimals=3)
		add_text_to_ax(0.1, 0.97, text, ax[i], fontsize=12, color='white')
		
	N_scale = (max(s)-min(s)) / N_bins            # The scale factor between histogram and the fit. Takes e.g. bin width into account.
	x_gauss = np.linspace(min(s), max(s), 1000)   # Create the x-axis for the plot of the fitted function
	y_gauss = N_scale*gauss_extended(x_gauss, len(s), 0, 1)                   # Unit Gaussian
	ax[3].plot(x_gauss, y_gauss, '-', color='blue', label='Unit Gauss (no fit)') 

	plt.tight_layout()
	plt.close()
	return fig



def demoArea():
            "Demo"
            cols = st.columns(4)

            W = cols[0].slider('W', 1, 10, 5)
            L = cols[2].slider('L', 1, 10, 5)
            sig_W = cols[1].slider('\sigma_W', 0., 1., .05)
            sig_L = cols[3].slider('\sigma_L', 0., 1., .05)

            fig = plt.figure()
            
            
            fig, ax = plt.subplots(1,2,figsize=(7,3))

            ax[0].axvline(0, color="white")
            ax[0].axhline(0, color="white")

            # W
            size = 1000
            Ws = np.random.normal(loc= W, scale=sig_W, size=size)
            Ls = np.random.normal(loc= L, scale=sig_L, size=size)
            ax[0].scatter(np.random.uniform(0,W, size=size),
                        Ls,
                        color='yellow', alpha=.2)

            ax[0].scatter(Ws, np.random.uniform(0,L, size=size),
                            color='yellow', alpha=.2)
            ax[0].set_xlabel('Width', color='black', fontsize=15)
            ax[0].set_ylabel('Length', color='black', fontsize=15)

            ax[1].set_title('AREA dist')
            ax[1].hist(Ws*Ls)
            st.pyplot(fig)



### ERROR PROP FROM TEACHERS
from IPython.core.display import Latex

# Define your PDF / model 
def gauss_pdf(x, mu, sigma):
	"""Normalized Gaussian"""
	return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

def gauss_extended(x, N, mu, sigma):
	"""Non-normalized Gaussian"""
	return N * gauss_pdf(x, mu, sigma)

# Week 2
def PDFs(size = 1000):
	x_uniform = np.random.rand(size)
	x_normal = np.random.randn(size)
	x_binomial = np.random.binomial(10, .3, size)
	x_poisson = np.random.randn(size)
	x_gaussian = np.random.randn(size)

	fig, ax = plt.subplots(1,5, figsize=(12,3))

	counts, bins = np.histogram(x_uniform)

	ax[0].hist(x_uniform)
	ax[0].set_title('Uniform', color='white')
	ax[1].hist(x_normal)
	ax[1].set_title('Normal', color='white')
	ax[2].hist(x_poisson)
	ax[2].set_title('Poisson', color='white')
	ax[3].hist(x_binomial)
	ax[3].set_title('Binomial', color='white')
	ax[4].hist(x_gaussian)
	ax[4].set_title('Gaussian', color='white')
	plt.close()

	return fig


def fitSimple(size =  100, nsteps = 100):

	x = np.random.rand(size)
	noise = lambda: np.random.randn()
	f = lambda x, a, b=0: a*x**2 + b*noise()
	y = f(x, 4, 0)

	# fitting


	loss = lambda a: sum((y - f(x,a))**2)


	delta = 0.1
	a0 = 1

	current_loss = loss(a0)
	for i in range(nsteps):
		pos_loss = loss(a0+delta)
		neg_loss = loss(a0-delta)
		if pos_loss < current_loss:
			current_loss = pos_loss
			a0 += delta
		elif neg_loss < current_loss:
			current_loss = neg_loss
			a0 -= delta
		else:
			delta *= 0.9

	fig, ax = plt.subplots(figsize=(8,3))
	ax.scatter(x,y, label='data')
	x_plot = np.linspace(min(x), max(x), max([100,size]))
	ax.plot(x_plot, f(x_plot, a0), ls='--', c='r', label='fit')
	plt.legend(facecolor='beige')
	plt.close()
	return fig


# Week 6
def makeBlobs(size=100):
	'''
	Makes 2d random blobs
	'''
	X = np.random.rand(size, 2)

	X[0][X[0]<0.5] -= .5
	X[1][X[1]<0.5] -= .5

	noise = np.random.randn(size, 2)*.1
	return X+noise

def scatter(x,y):
	fig, ax = plt.subplots()
	ax.scatter(x, y)
	plt.close()
	return fig





