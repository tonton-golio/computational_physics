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





