import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd    
from time import time
import seaborn as sns
from time import sleep
from scipy.optimize import curve_fit
import matplotlib as mpl
from scipy.stats import expon, norm
import pylab as pl

# physics with python live stream @ https://www.twitch.tv/iamtontonton


# General

def set_rcParams():
	"""
		setting matplotlib style
	"""
	# extra function
	mpl.rcParams['patch.facecolor'] = (0.04, 0.065, 0.03)
	mpl.rcParams['axes.facecolor'] = 'black'
	mpl.rcParams['figure.facecolor'] = (0.04, 0.065, 0.03)
	mpl.rcParams['xtick.color'] = 'white'
	mpl.rcParams['ytick.color'] = 'white'
	# mpl.rcParams['axes.grid'] = True  # should we?
	mpl.rcParams['figure.autolayout'] = True  # 'tight_layout'

fig_counter = np.array([0])
text_path = 'assets/applied_statistics/text/'
set_rcParams()

st.set_page_config(page_title="help wanted", 
    page_icon="🧊", 
    layout="wide", 
    initial_sidebar_state="collapsed", 
    menu_items=None)

def getText_prep(filename = text_path+'week1.md', split_level = 2):
	"""
		get text from markdown and puts in dict
	"""
	# extra function
	with open(filename,'r', encoding='utf8') as f:
		file = f.read()
	level_topics = file.split('\n'+"#"*split_level+' ')
	text_dict = {i.split("\n")[0].replace('### ','') : 
                "\n".join(i.split("\n")[1:]) for i in level_topics}
    
	return text_dict  

def makeFunc_dict(filename='utils/utils_appstat.py'):
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

def figNum():
    # extra function
    fig_counter[0] += 1
    return f"Figure {fig_counter[0]}: "

def caption_figure(text, st=st):
    # extra function
    st.caption('<div align="center">' +figNum()+ text+'</div>' , unsafe_allow_html=True)


## from teachers
def format_value(value, decimals):
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

def values_to_string(values, decimals):
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

def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    # extra function
    return len(max(s, key=len))

def nice_string_output(d, extra_spacing=5, decimals=3):
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

def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    # extra function
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None



###########################################
# week 1

#
def means(arr, truncate=1):
	"""Determines mean as obtained by different methods.

            The different means are:
                * Geometric mean, Arithmetic mean, Median , Mode, Harmonic, Truncated
            returns: dictionary with means"""

	n = len(arr)
	arr = np.sort(arr) # for median
	def geometric(arr): return np.prod(arr)**(1/n)
	def arithmetic(arr): return np.sum(arr) / n
	def median(arr): return arr[n//2] if n%2==0 else arr[n//2+1]
	def harmonic(arr): return (np.sum( arr**(-1) ) / n)**(-1)
	def truncated(arr): return arithmetic(arr[truncate:-truncate])
	def mode(arr):
		unique, counts = np.unique(arr, return_counts=True)
		which = np.argwhere(counts==np.max(counts))
		return arithmetic(unique[which])

	return {
			'Geometric mean' : geometric(arr),
			'Arithmetic mean' :  arithmetic(arr), 
			'Median': median(arr),  
			'Mode': mode(arr),
			'Harmonic': harmonic(arr), 
			'Truncated' : truncated(arr)
			}

def showMeans(arr, truncate=1):
	"""
		on a hist, shows means
	"""
    # extra function
	fig, ax = plt.subplots(figsize=(6,2))
	arr = np.sort(arr.copy())
	d = means(arr, truncate=truncate)
	arr_truncated = arr[truncate:-truncate]
	counts, bins = np.histogram(arr_truncated)
	
	ax.hist(arr_truncated, bins=max([15,len(arr)//40]), color='lightblue', alpha=.7)
	colors = sns.color_palette("RdBu", 8)
	for i, c in zip(d, colors):
		ax.axvline(d[i], label=i, c=c)
	#ax.legend(facecolor='beige')
	
	#ax.set()##xscale='log')#, yscale='log')
	_ = [plt.text(d[key], (idx+1)*max(counts)//9-5 , s = key, color='purple') for idx, key in enumerate(d)]
	plt.close()

	return fig

def demo_comparing_means():
    # extra function
    cols = st.columns(5)
    n_normal = cols[0].slider("n_normal", 1, 1000, 100)
    n_exp= cols[1].slider("n_exp", 1, 1000, 0)
    n_cauchy= cols[2].slider("n_cauchy", 1, 1000, 0)
    truncate = cols[3].slider("truncate", 1, 100, 1)
    seed = cols[4].slider("seed", 1, 100, 42)
    np.random.seed(seed)
    normal = np.random.randn(n_normal)
    exp = np.random.exponential(n_exp) 
    cauchy = np.random.standard_cauchy(n_cauchy)
    
    arr = np.hstack([normal, exp, cauchy]).flatten()

    fig = showMeans(arr, truncate)   # Perhaps change this to 2D
    st.pyplot(fig)
    caption_figure('Different mean-metrics shown for a distribution.')

def std_calculations(n=400):
    # extra function
	fig, ax = plt.subplots(3,1, figsize=(4,4), sharex=True, sharey=True)
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
		

		ax[idx].hist(res['N'], bins=shared_bins, label=r'$\hat{\sigma}$', fill=True, alpha=.6,	facecolor='pink')
		ax[idx].hist(res['N-1'],bins=shared_bins, label=r'$\tilde{\sigma}$', fill=True, alpha=.6,		facecolor='yellow')
		ax[idx].set_title(f'N={N}', color="white")
        
		ax[idx].set(xticks=[], yticks=[])
	ax[0].legend(facecolor='black', edgecolor=None)
	for i in ax[:1]:
        
		l = i.legend(fontsize=12)
		frame = l.get_frame()
		for text in l.get_texts():
			text.set_color("white")
			frame.set_edgecolor('black')
	ax[idx].set_xlabel(r'standard devitation, $\sigma$', color='beige')
	#fig.suptitle(f'Distributions of standard deviations, n={n}', color='beige')
	plt.tight_layout()
	plt.close()
	return fig


def roll_a_die(num=100):
    # extra function
    x = np.random.randint(1,7,num)
    means = [np.mean(x[:n]) for n in range(2, num)]

    fig, ax = plt.subplots(1, 2, figsize=(6,2))
    
    ax[0].hist(x, bins=np.arange(1,7, 0.5), color='pink', alpha=.5)
    ax[0].set_xticks(np.arange(1,7)+.25)
    ax[0].set_xticklabels(np.arange(1,7))
    ax[0].set_xlabel('roll', color='white')
    ax[0].set_ylabel('freq', color='white')

    ax[1].plot(range(2, num), means, c='r', alpha=.7)
    ax[1].set_xlabel('number of rolls in mean', color='white')
    ax[1].set_ylabel('mean', color='white')
    
    plt.close()
    return fig

def roll_dice(rolls=200):
    # extra function
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

def makeDistibutions(n_experi = 1000, N_uni= 1000,  N_exp= 1000,  N_cauchy = 1000):
    # extra function
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
    # extra function
	fig, ax = plt.subplots(1,4,figsize = (8,4))
	for i, (s, name) in enumerate(zip(sums, ['uni', 'exp', 'chauchy','combined'])):
		ax[i].hist( s, bins=N_bins , histtype='step', 
                        color='pink', alpha=.4)
		ax[i].set_title(name, color='beige')
		ax[i].set(yticks=[], ylim=(0,ax[i].get_ylim()[1]*1.2 ))
		
		text = {'mean':s.mean(),
			'std' : s.std(ddof=1)}
		text = nice_string_output(text, extra_spacing=2, decimals=3)
		add_text_to_ax(0.1, 0.97, text, ax[i], fontsize=12, color='white')
		
	N_scale = (max(s)-min(s)) / N_bins            # The scale factor between histogram and the fit. Takes e.g. bin width into account.
	x_gauss = np.linspace(min(s), max(s), 1000)   # Create the x-axis for the plot of the fitted function
	y_gauss = N_scale*gauss_extended(x_gauss, len(s), 0, 1)                   # Unit Gaussian
	ax[3].plot(x_gauss, y_gauss, '-', color='cyan', label='Unit Gauss (no fit)') 

	plt.tight_layout()
	plt.close()
	return fig



def demoArea():
    # extra function
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


# Define your PDF / model 
def gauss_pdf(x, mu, sigma):
	"""Normalized Gaussian"""
	return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

def gauss_pdf_scaled(x, mu, sigma, a):
	"""Normalized Gaussian"""
	return a * gauss_pdf(x, mu, sigma)

def gauss_extended(x, N, mu, sigma):
	"""Non-normalized Gaussian"""
	return N * gauss_pdf(x, mu, sigma)

def chi2_demo(resolution=128, n_samples=10):

    def chi2_own(f, y_gt, x,
                a_lst= np.linspace(-2,5,resolution), 
                b_lst = np.linspace(2,6,resolution)): 
        expected = y_gt
        
        Z = np.zeros((a_lst.shape[0], b_lst.shape[0]))
        for i, a in enumerate(a_lst):
            for j, b in enumerate(b_lst):
                noise=  np.random.randn(len(x))
                observed = f(x, a, b) + noise
                chi2 = np.sum((observed-expected)**2/expected)

                Z[i,j] = chi2
        X, Y = np.meshgrid(a_lst, b_lst)
        return X,Y,Z
    
    def plot(X,Y,Z, # for contour 
                x, y_gt, # for scatter
                f, popt # for fit on scatter
                ):
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1)
        
        ax.scatter(x,y_gt, marker='x', s=42, c='cyan')

        text = {'RMSE':sum((y_gt - f(x, *popt))**2)**.5,
                }
        text = nice_string_output(text, extra_spacing=2, decimals=3)
        add_text_to_ax(0.1, 0.97, text, ax, fontsize=12, color='white')
		

        x_fit = np.linspace(min(x), max(x), 20)
        ax.plot(x_fit, f(x_fit, *popt),  c='r')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.contour3D(X, Y, Z, 100, cmap='gist_heat_r', alpha=.5)
        ax.set_xlabel('a', color='white')
        ax.set_ylabel('b', color='white')
        ax.set_zlabel(r'$\chi^2$', color='white')


        ax.set_title('$\chi^2$ of phasespace', color="white")

        ax.view_init(30, 50)
        
        plt.tight_layout()
        fig.set_facecolor('black')
        return fig

    def f(x,a,b):
        return a*x+b
    

    x = np.linspace(-1,1,n_samples) 
    y_gt = expected = f(x, a=2, b = 4 )+np.random.randn(n_samples)*.2

    X,Y,Z = chi2_own(f, y_gt, x,
                a_lst= np.linspace(-2,5,resolution),
                 b_lst = np.linspace(2,6,resolution))
    popt = [X[:,np.argmin(Z)//resolution][0],
            Y[np.argmin(Z)%resolution,:][0]
            ]
    
    fig = plot(X,Y,Z, x, y_gt, f, popt) # for fit on scatter
     
    return fig

# new chi2
def chi2_minimizer(f, x, y_gt, p0,h=0.01, 
                    lr = .1, tol=.05, max_fev=400):
    def chi2(y_pred, y_gt):
        #print(y_pred.shape, y_gt.shape)
        return np.sum((y_pred-y_gt)**2/y_gt)

    def chi2_of_f(x, y_gt,f, p):
        y_pred = f(x, *p)
        #print(y_pred.shape, y_gt.shape)
        return chi2(y_pred, y_gt), y_pred

    def getGrad(f, x, y_gt, p0, h=0.01):
        # determine gradient
        chi2_, y_p0 = chi2_of_f(x, y_gt,f, p0)

        chi2_grad = np.zeros(len(p0))#np.array([dC_da, dC_db])
        for idx, pi in enumerate(p0):
            p_tmp = p0.copy()
            #print(p_tmp)
            p_tmp[idx] += h
            chi2_grad[idx], _   = chi2_of_f(x, y_gt,f, p_tmp)

        chi2_grad -= chi2_
        chi2_grad /= h

        
        return chi2_grad

    def minimize(f, x, y_gt, p0 = np.array([2.5, 3.0]),h=0.01, 
                    lr = .1, tol=.5, max_fev=400):
        popt = p0.copy()
        grad = 9 ; i =0
        #print(popt)
        while np.linalg.norm(grad)>tol and (i < max_fev):
            grad =  getGrad(f, x, y_gt, popt, h=h)
            #print(grad)
            popt = popt - lr * grad
            i += 1
        
        print(i)
        return popt


    popt = minimize(f, x, y_gt, p0,h, lr, tol, max_fev)
    chi2_val, y_pred = chi2_of_f(x, y_gt,f, popt)
    return chi2_val, popt

def chi2_demo_2(f, p_true, p0,n_samples, noise_scale=0.2,
            h=0.01, lr = .1, tol=.05, max_fev=400):
    # define data
    x = np.linspace(-1,1,n_samples) # where is the function located
    noise =  np.random.normal(loc=0, scale=noise_scale, size=n_samples)
    y_gt = expected = f(x, *p_true ) + noise
    #print("y_gt.shape",y_gt.shape)
    chi2_val, popt = chi2_minimizer(f, x, y_gt, p0,h, 
                        lr, tol, max_fev)
    
    fig = plt.figure()
    plt.scatter(x, y_gt, label='data')

    x_plot = np.linspace(min(x), max(x), 100)
    plt.plot(x_plot,f(x_plot, *p0), label='initial guess')
    plt.plot(x_plot,f(x_plot, *popt), label=f'popt, $\chi^2=${round(chi2_val, 3)}')

    l = plt.legend(fontsize=12)
    frame = l.get_frame()
    for text in l.get_texts():
        text.set_color("white")
        frame.set_edgecolor('black')
    return fig



# Week 2
def PDFs(size = 1000, n_binomial=100, p_binomial=0.2):
    # extra function
	x_uniform = np.random.rand(size)
	
	x_binomial = np.random.binomial(n_binomial, p_binomial, size)
	x_poisson = np.random.randn(size)
	x_gaussian = np.random.randn(size)

	fig, ax = plt.subplots(1,4, figsize=(12,3), sharey=True)

	counts, bins = np.histogram(x_uniform)

	ax[0].hist(x_uniform, bins=40, color='orange')
	ax[0].set_title('Uniform', color='white')

	ax[1].hist(x_poisson, bins=40, color='pink')
	ax[1].set_title('Poisson', color='white')
	ax[2].hist(x_binomial, bins=40, color='orange')
	ax[2].set_title(f'Binomial n={n_binomial}, p={p_binomial}', color='white')
	ax[3].hist(x_gaussian, bins=20, color='pink')
	ax[3].set_title('Gaussian', color='white')
	plt.close()

	return fig


def golden_section_min(f,a,b,tolerance=1e-3, maxitr=1e3):
    factor = (np.sqrt(5)+1)/2
    
    # define c and d, which are points between a and b
    n_calls = 0
    while (abs(a-b)>tolerance) and (maxitr > n_calls):
        c = b - (b-a)/factor
        d = a + (b-a)/factor

        if f(c) > f(d):
            a = c
        else:
            b=d
        #st.write(f(a),f(c),f(d),f(b))
        n_calls += 1
    return (c+d)/2, n_calls

def maximum_likelihood_finder(mu, sample, 
								return_plot=False, verbose=False):
	
	xs = np.linspace(min(sample), max(sample), 100)

	# determine likelihood  (should really implement bisection)
	log_likelihood = lambda mu=0, sig=1: np.sum(np.log(norm.pdf(sample, scale=sig, loc=mu)))
	
	# linear search
	start_linear_search = time()
	mus, sigs = np.linspace(-2,2, 1000), np.linspace(.3, 4, 1000)
	estimates_mu = [log_likelihood(mu, sig=1) for mu in mus]
	
	mu_best_linear = mus[estimates_mu.index(np.max(estimates_mu))]
	estimates_sig = [log_likelihood(mu_best_linear, sig=sig) for sig in sigs]
	sig_best_linear = sigs[estimates_sig.index(np.max(estimates_sig))]
	stop_linear_search = time()


	# golden_section_min
	start_golden_search = time()
	a_mu, b_mu = -2,2
	a_sig, b_sig = 0.5,5
	
	f_mu_neg = lambda mu:-1*log_likelihood(mu, sig=1)
	#st.write(f_mu_neg(a_mu), f_mu_neg(0), f_mu_neg(b_mu))
	mu_best, ncalls_mu = golden_section_min(f_mu_neg,a_mu,b_mu,tolerance=1e-5, maxitr=1e3)
	
	f_sig_neg = lambda sig:-1*log_likelihood(mu_best, sig)
	#st.write(f_sig_neg(a_sig), f_sig_neg(0), f_sig_neg(b_sig))
	sig_best, ncalls_sig = golden_section_min(f_sig_neg,a_sig,b_sig,tolerance=1e-5, maxitr=1e3)
	
	stop_golden_search = time()
	
	if verbose:
		cols = st.columns(2)
		num_round = 4
		cols[0].write(f"""
			linear:

				mu_best = {round(mu_best_linear, num_round)}
				sig_best = {round(sig_best_linear, num_round)}
				time = {round(stop_linear_search-start_linear_search,3)}
			""")
		cols[1].write(f"""
			golden_section_min: 
				
				mu_best = {round(mu_best, num_round)}
				sig_best = {round(sig_best, num_round)}
				ncalls_mu =  {ncalls_mu}
				ncalls_sig =  {ncalls_sig}
				time = {round(stop_golden_search-start_golden_search,3)}
			""")

	def plot():
		fig, ax = plt.subplots(1,2, figsize=(8,3))
		ax[0].hist(sample, density=True, label='sample', 
					bins=int(len(sample)**.9)//3,
					color='red',
					alpha=.5)
		ys = norm.pdf(xs, scale=sig_best, loc=mu_best)
		ax[0].plot(xs, ys, label='best', color='cyan', ls='--')
		ax[0].set_xlabel('sample value', color='beige')
		ax[0].set_ylabel('occurance frequency', color='beige')

		ax[1].set_xlabel('parameter value', color='beige')
		ax[1].plot(mus, 1*np.array(estimates_mu), label=r'$\mu$')
		ax[1].plot(sigs, 1*np.array(estimates_sig), label='sigma')
		ax[1].set_ylabel('likelihood', color='beige')
		#ax[1].set_yscale('log')
		
		for i in ax:
			l = i.legend(fontsize=12)
			for text in l.get_texts():
				text.set_color("white")

		
		plt.tight_layout()
		plt.close()
		return fig
	if return_plot: return mu_best, sig_best, log_likelihood(mu_best, sig_best), plot()
	else: return mu_best, sig_best, log_likelihood(mu_best, sig_best)

def evalv_likelihood_fit(mu, sig, L,  sample_size, N_random_sample_runs=10,nbins = 20, plot=True):
        
    # evaluate likelihood of different samplings
    Ls = []
    for i in range(N_random_sample_runs):
        sample_new = np.random.normal(loc=mu,scale=sig, size=sample_size)
        _, _, L_new = maximum_likelihood_finder(mu, sample_new, return_plot=False, verbose=False)
        Ls.append(L_new)
    Ls = np.array(Ls)

    # make hist
    counts, bins = np.histogram(Ls, bins=nbins)
    x = (bins[1:]+bins[:-1])/2
    y = counts
    x=x[y!=0] ;  y=y[y!=0]  # rid of empy bins
    
    popt, pcov = curve_fit(gauss_pdf_scaled, x, y, p0=[L, 100, 2])
    #st.write(popt)
    x_plot = np.linspace(min(x)*1.05, max(x), 100)
    plot_gauss_pdf = gauss_pdf_scaled(x_plot, *popt)
    
    x_worse = np.linspace(-10000, L, 100)
    worse_gauss_pdf = gauss_pdf_scaled(x_worse, popt[0], popt[1], 1)
    prob_worse = np.sum(worse_gauss_pdf)  #\int (gauss_{-\infty}^L)
    if plot:
        fig = plt.figure(figsize=(8,3))
        
        plt.hist(Ls, label='sample dist', bins=nbins)
        plt.axvline(L, c='r', ls='--', label='original fit')
        plt.legend(facecolor='beige')
        
        plt.plot(x_plot, plot_gauss_pdf)
        plt.scatter(x,y, c='r')

        # area
        x_area = np.linspace(L-popt[1]*5, L, 100)
        plt.fill_between(x_area, gauss_pdf_scaled(x_area, *popt), alpha=.3, color='pink')

    return fig, prob_worse

# Week 6
def makeBlobs(size=100):
	'''
	Makes 2d random blobs
	'''
    # extra function
	X = np.random.rand(size, 2)

	X[0][X[0]<0.5] -= .5
	X[1][X[1]<0.5] -= .5

	noise = np.random.randn(size, 2)*.1
	return X+noise




