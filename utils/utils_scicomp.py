from utils.utils_global import *

text_path = 'assets/scientific_computing/text/'
 


# Linear Equations
# C
@function_profiler
def lu_factorize(M):
    """
    Factorize M such that M = LU

    Parameters
    ----------
    M : square 2D array
        the 1st param name `first`

    Returns
    -------
    L : A lower triangular matrix
    U : An upper triangular matrix
    """
    m, U = M.shape[0], M.copy()
    L = np.eye(M.shape[0]) # make identity 
    for j in range(m-1):
        for i in range(j+1,m):
            scalar    = U[i, j] / U[j,j]
            U[i]     -=  scalar*U[j]
            L[i, j]   = scalar
            
    return L, U

@function_profiler
def forward_substitute(L,b):
    '''
    Takes a square lower triangular matrix L 
    and a vector $b$ as input, and returns the 
    solution vector y to Ly = b.
    '''
    y = np.zeros(np.shape(b))
    for i in range(len(b)):
        y[i] = ( b[i] - L[i] @ y) / L [i,i]
    return y

@function_profiler
def backward_substitute(U,y):
    '''which takes a square upper triangular 
    matrix U and a vector y as input, and returns 
    the solution vector  x  to  Ux=y.
    '''
    x = np.zeros(np.shape(y))

    for i in range(1,1+len(x)):
        x[-i] = ( y[-i] - U[-i] @ x )/U[-i,-i]

    return x

@function_profiler
def solve_lin_eq(M,z):
    L,U = lu_factorize(M)
    y = forward_substitute(L,z)
    return backward_substitute(U,y)


# Least Squares
@function_profiler
def toy_sim_leastSquares():
    c = st.empty()
    size = 50
    x = np.random.uniform(0,1,size)
    noise = np.random.rand(size)
    a, b = 3, 1
    y = a*x+b +noise

    fig, ax = plt.subplots(figsize=(12,4))
    plt.scatter(x,y, label="data")
    plt.legend()
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    c.pyplot(fig)
    
    a_guess, b_guess = 1,1
    step_size = 0.1
    for i in range(20):
        fig, ax = plt.subplots(figsize=(12,4))
        sleep(.07)
        plt.scatter(x,y, label="data")
        loss = lambda a_guess, b_guess: sum((y - (a_guess*x+b_guess))**2)**.5
        plt.plot(x, a_guess*x+b_guess, label='fit', ls='--', c='r')
        
        #update guess

        dloss_da = loss(a_guess+step_size, b_guess)- loss(a_guess, b_guess)
        dloss_db = (loss(a_guess, b_guess+step_size)- loss(a_guess, b_guess))
        a_guess -= dloss_da
        b_guess -= dloss_db


        step_size *= .95
        plt.legend(facecolor='beige')
        fig.set_facecolor('black')
        ax.set_facecolor('black')
        c.pyplot(fig)
    plt.close()



# Initial Value problems
@function_profiler
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
        resolution = int(2**col2.slider('resolution = 2^x, x=', 3,13, 4))
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

        return nabla
        
    
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

    
    def make_figs(snapshots):
        saved_figs = {}
        # st.write('snapshots[''p''].keys()', snapshots['p'].keys())
        for n in snapshots['p'].keys():
            fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
            for i, arr, name in zip([0,1],[snapshots['p'][n],snapshots['q'][n]], ['p','q']):
                ax[i].imshow(arr)
                ax[i].set(title=name, xticks=[], yticks = [])
                # st.write('snapshot')

            plt.tight_layout()
            fig.patch.set_facecolor('darkgrey')
            plt.savefig(f'tmp/reDi{n}.png', dpi=300, transparent=True)
            saved_figs[n] = fig
        
        if 'saved_figs' not in st.session_state:
            st.session_state.saved_figs = saved_figs

        return saved_figs

    def repr(saved_figs, meta_data):    
        method_map = {
                'forward-Euler': 0,
                'Runge-Kutta'  : 1,
                'backward-Euler': 2,
                }
        # st.write(saved_figs.keys())
        st.write(st.session_state.saved_figs.keys())
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
        st.write(saved_figs.keys()) 
        st.pyplot(saved_figs[n]    )

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

    with st.sidebar:
        # run button
        run_ = st.button('Run simulation')

    if not 'saved_figs' in st.session_state or run_:
        snapshots, meta_data  = run()

        saved_figs = make_figs(snapshots)

    try:
        repr(saved_figs, meta_data)
    except:
        print('no data to repr')