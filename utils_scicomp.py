import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd    
from time import time
from time import sleep


# General

text_path = 'assets/scientific_computing/text/'

st.set_page_config(page_title="Scientific Computing", 
    page_icon="🧊", 
    layout="wide", 
    initial_sidebar_state="collapsed", 
    menu_items=None)

def getText_prep(filename = 'pages/stat_mech.md', split_level = 2):
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

# Least Squares
def toy_sim_leastSquares():
        c = st.empty()
        size = 20
        x = np.random.uniform(0,1,size)
        noise = np.random.rand(size)
        a, b = 3, 1
        y = a*x+b +noise

        fig = plt.figure()
        plt.scatter(x,y, label="data")
        plt.legend()
        c.pyplot(fig)
        
        a_guess, b_guess = 1,1
        step_size = 0.1
        for i in range(30):
            fig = plt.figure()
            sleep(.07)
            plt.scatter(x,y, label="data")
            loss = lambda a_guess, b_guess: sum((y - (a_guess*x+b_guess))**2)**.5
            plt.plot(x, a_guess*x+b_guess, label='fit', ls='--', c='r')
            
            #update guess

            dloss_da = loss(a_guess+step_size, b_guess)- loss(a_guess, b_guess)
            dloss_db = (loss(a_guess, b_guess+step_size)- loss(a_guess, b_guess))
            a_guess -= dloss_da
            b_guess -= dloss_db     
            
            plt.legend()
            c.pyplot(fig)
        plt.close()


# Initial Value problems
def run_reactionDiffusion():

    st.markdown('## Reaction diffusion')
    st.latex(r'''
    \begin{align}
        p_t &= D_p\nabla p + p^2q + C-(K-1)p\\
        q_t &= D_q\nabla q - p^2q + Kp
    \end{align}
    ''')


    # options in sidebar
    def load_stability_data():
        # load or initialize data dictionary
        try:
            data = np.load('data.npz', allow_pickle=True)[np.load('data.npz', allow_pickle=True).files[0]].item()
            df = pd.concat({
                k: pd.DataFrame.from_dict(v, 'index') for k, v in data.items()
            }, 
            axis=1).T
        except:
            data = {};  np.savez('data', data)
            df = pd.DataFrame()
        
        return data, df

    data, df = load_stability_data()
    with st.sidebar:

        'Initial conditions'
        standard_settings = {
            "Prof's rec." : {
                                'Dp' : 1,
                                'Dq' : 8,
                                'C' : 4.5,
                                'K' : 9
            },
            "Mushrooms" : {
                                'Dp' : 1,
                                'Dq' : 9,
                                'C' : 2.3,
                                'K' : 11
            }
        }


        std_set = st.radio('standard settings', standard_settings.keys())
        
        col1, col2 = st.columns(2)
        Dp = col1.slider('Dp', 0, 10, standard_settings[std_set]['Dp'])
        Dq = col2.slider('Dq', 0, 10,standard_settings[std_set]['Dq'])
        C = col1.slider('C', 0., 10., standard_settings[std_set]['C'])
        K = col2.slider('K', 0, 12,standard_settings[std_set]['K'])

        'HyperParams'
        col1, col2 = st.columns(2)
        
        #nsteps = int(10**(col1.slider('Number of steps = 10^x', 0.,5., 2.5)))
        T_end = int(2**(col1.slider('T_end = 2^x, x=', 0,8, 1)))
        resolution = int(2**col2.slider('resolution = 2^x, x=', 3,13, 3))
        method = col1.selectbox('Method', ['forward-Euler', 'Runge-Kutta', 'backward-Euler'])
        dt = 2**(col2.slider('dt = 2^x, x=',-14,-3,-5))
        
        nsteps = int(T_end/dt)
        

        try:
            sidex = np.linspace(0, 40, resolution)
            dx = sidex[1]-sidex[0]
            df_dx = df[abs(df.dx-dx)<0.001].copy()
            df_dx.runtime = df_dx.runtime/df.nsteps
            mean_per_step = df_dx[abs(df_dx.dx-dx)<0.001].mean()['runtime']
            approx_runtime = mean_per_step*nsteps 
            
            f'runtime $\approx$' +  str(round(approx_runtime,2))
        except:
            r'runtime $\approx$ unknown'
        info = {"nsteps" : nsteps,
                "t_end" : T_end,
                "dt" : dt,
                "resolution" : resolution}

        info
    
    def makeGrid(resolution=10, low=0, high=40):
        sidex = np.linspace(0, 40, resolution)
        a, b = np.meshgrid(sidex, sidex)
        dx = sidex[1]-sidex[0]
        return a,b, dx

    def middleMark(a,b, low=10, high=30):
        '''
        makes array of shape = arr.shape filled with bools marking the middle 
        ''' # a little expensive, but doesnt matter
        middle_mark = ( ((a>10).astype(float) +\
                (a<30).astype(float) +\
                (b>10).astype(float) +\
                (b<30).astype(float))//4 ).astype(bool)
        return middle_mark  

    def laplacianEverywhere(X, dx=0.1):
        #for each entry difference to nieghbour
        # ignore boreder sites
        right = np.roll(X, -1) - X  # what right has compared to ego
        left = np.roll(X, 1) - X  # what left has compared to ego
        down = (np.roll(X.T, 1,)-X.T).T # mine vs. down
        up = (np.roll(X.T, -1)-X.T).T#,X # what i have compared to up
        
        nabla = (right+left + up+down)/dx**2

        # edges
        nabla[:,0] = ((right+up+down)/dx**2)[:,0]
        nabla[:,-1] = ((left+up+down)/dx**2)[:,-1]
        nabla[0,:] = ((right+left+down)/dx**2)[0,:]
        nabla[-1,:] = ((right+left+up)/dx**2)[-1,:]


        # corners
        #nabla[0,0] = ((right+down)/dx**2)[0,0]
        #nabla[0,-1] = ((left+down)/dx**2)[0,-1]
        #nabla[-1,0] = ((right+up)/dx**2)[-1,0]
        #nabla[-1,-1] = ((left+up)/dx**2)[-1,-1]

        #, nabla[:,-1], nabla[0,:], nabla[-1,:] = 0, 0, 0, 0  # should i set border equal to zero?
        return nabla
        
    @st.cache
    def run():
        # initialize
        start_time = time()
        a,b, dx = makeGrid(resolution)
        middle_mark = middleMark(a,b)

        p, q = np.zeros(a.shape), np.zeros(a.shape)
        p[middle_mark], q[middle_mark] = C+0.1, K/C+0.2

        meta_data = {"stable":True,
                    'dx':dx,
                    'dt':dt,
                    'resolution': resolution,
                    'method':method
                    }
        snapshots = {'p':{},'q':{}}

        # evolve
        if   method == 'forward-Euler':
            for step in range( nsteps):
                (nablaP, nablaQ) = (laplacianEverywhere(X=i, dx=dx) for i in [p,q])

                p_new = p + dt * (Dp*nablaP + p**2*q + C-(K+1)*p)
                q_new = q + dt * (Dq*nablaQ - p**2*q + K*p)
                p, q = p_new, q_new

                if (np.isnan(p).any()):
                    meta_data["stable"] = False      
                    break  
                if step%(nsteps//10)==0:
                    snapshots['p'][step], snapshots['q'][step] = p.copy(), q.copy()
            
            snapshots['p'][step], snapshots['q'][step] = p, q

        elif method == 'backward-Euler':

            # we need to take one step using forward euler first as to 
            # obtain enough infomation for the derivates needed for backward euler

            nablaP = laplacianEverywhere(X=p, dx=dx)
            nablaQ = laplacianEverywhere(X=q, dx=dx)

            p = p + dt * (Dp*nablaP + p**2*q + C-(K+1)*p)
            q = q + dt * (Dq*nablaQ - p**2*q + K*p)
            
            # now we have done 1 forward step

            for step in range( nsteps):

                def pfunc(p_new, q_new, p_prev):
                    return p_prev + dt * (Dp*laplacianEverywhere(X=p_new, dx=dx) + p_new**2*q_new + C-(K+1)*p_new)
                    #solve for p_new
                def qfunc(p_new, q_new, q_prev):
                    return q_prev + dt * (Dq*laplacianEverywhere(X=q_new, dx=dx) - p_new**2*q_new + K*p_new)
                    
                p_new = p.copy()
                q_new = q.copy()
                minimize_steps = 10
                error = [[0],[0]]
                for minimize_step in range(minimize_steps):
                    p_new_ = pfunc(p_new, q_new, p)
                    q_new_ = qfunc(p_new, q_new, q)
                    error[0].append(np.sum(abs(p_new-p_new_)))
                    error[1].append(np.sum(abs(q_new-q_new_)))
                    p_new = p_new_
                    q_new = q_new_
                    if (error[0][-1] < 0.01) and (error[1][-1] < 0.01):
                        
                        break
                p_prev = p
                q_prev = q
                




                if (np.isnan(p).any()):
                        stable = False      
                        break  

        elif method == 'Runge-Kutta':
                def dp_dt(p, q, dx):
                    nablaP = laplacianEverywhere(X=p, dx=dx)
                    return Dp*nablaP + p**2*q + C-(K+1)*p
                
                def dq_dt(p, q, dx):
                    nablaQ = laplacianEverywhere(X=q, dx=dx)
                    return Dq*nablaQ - p**2*q + K*p
                
                
                for step in range( nsteps):
                    
                    p_k1 = dp_dt(p, q, dx)
                    q_k1 = dq_dt(p, q, dx)
                    p_k2 = dp_dt(p+dt/2*p_k1, q+dt/2*q_k1, dx)
                    q_k2 = dq_dt(p+dt/2*p_k1, q+dt/2*q_k1, dx)
                    p_k3 = dp_dt(p+dt/2*p_k2, q+dt/2*q_k2, dx)
                    q_k3 = dq_dt(p+dt/2*p_k2, q+dt/2*q_k2, dx)
                    p_k4 = dp_dt(p+dt*p_k3, q+dt*q_k3, dx)
                    q_k4 = dq_dt(p+dt*p_k3, q+dt*q_k3, dx)

                    p_new = p + dt/6 * (p_k1+2*p_k2+2*p_k3+p_k4)
                    q_new = q + dt/6 * (q_k1+2*q_k2+2*q_k3+q_k4)
                    p = p_new
                    q = q_new
                    if (np.isnan(p).any()):
                        meta_data["stable"] = False 
                        break
                    
                    if step%(nsteps//10)==0:
                        snapshots['p'][step] = p
                        snapshots['q'][step] = q
                snapshots['p'][step] = p
                snapshots['q'][step] = q

        elif method == 'Crank-Nicolson':
            pass

        # return
        timer = time() - start_time
        meta_data['steps made'] = step
        meta_data['runtime'] = timer
        return snapshots, meta_data

    def repr(snapshots, meta_data):    
        method_map = {
                'forward-Euler': 0,
                'Runge-Kutta'  : 1,
                'backward-Euler': 2,
                }

        #data
        index = 0 if len(data.keys()) == 0 else max(list(data.keys()))+1
        data[index] = meta_data
        np.savez('data',data)
        df = pd.concat({
            k: pd.DataFrame.from_dict(v, 'index') for k, v in data.items()
        }, 
        axis=1).T
        for col in ['dx', 'dt', 'steps made', 'runtime']:
            df[col] = df[col].apply(float) 

        # plot snapshots
        n = st.select_slider('snapshot',snapshots['p'].keys())
        p,q = snapshots['p'][n],snapshots['q'][n]

        fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
        for i, arr, name in zip([0,1],[p,q], ['p','q']):
            ax[i].imshow(arr)
            ax[i].set(title=name, xticks=[], yticks = [])

        plt.tight_layout()
        #plt.savefig(f'figures/reDi0_K{K}.png', dpi=300, transparent=True)
        fig.patch.set_facecolor('darkgrey')
        st.pyplot(fig)

        st.markdown(r"## Stability")
        
        # plot stability
        meths = df.method.unique()
        fig, ax = plt.subplots(1,len(meths), facecolor="#222222", sharex=True, sharey=True)
        ax = ax if type(ax)==np.ndarray else [ax]
        
        for meth in meths:
            df_m = df[df.method==meth]
            meth_num = int(method_map[meth])
            c_map = {0:'r', 1:'g'}
            for i, stat in enumerate(['Unstable', 'Stable']):
                select = df_m[df_m['stable']==i]
                ax[meth_num].scatter(np.log2(select['dx']),np.log2(select['dt']) , 
                                label=stat, c=c_map[i]) 
            ax[meth_num].grid()
            ax[meth_num].set(title=meth, 
                        xlabel=r'$\log_2(dx)$', ylabel=r'$\log_2(dt)$')
            ax[meth_num].legend(facecolor='beige')
            ax[meth_num].set_facecolor((1., 0.85, 0.3))
        
        plt.tight_layout()
        
        #plt.savefig('figures/reDi1.png', dpi=300, transparent=True)
        fig.patch.set_facecolor('darkgrey')
        st.pyplot(fig)

        st.write(r'''
        Notice, the stability of the Runge-Kutta method is greater than that of the 
        forward-Euler method.''')

    snapshots, meta_data  = run()
    repr(snapshots, meta_data)