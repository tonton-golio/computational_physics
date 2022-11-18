import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd    
from time import time
from time import sleep


# General

text_path = 'assets/applied_statistics/text/'

st.set_page_config(page_title="Applied Statistics", 
    page_icon="🧊", 
    layout="wide", 
    initial_sidebar_state="collapsed", 
    menu_items=None)

def getText_prep(filename = text_path+'week1.md', split_level = 2):
    with open(filename,'r' ) as f:
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





