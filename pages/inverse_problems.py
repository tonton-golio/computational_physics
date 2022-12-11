from utils.utils_inverse import *
st.title('Inverse Problems')

("""
        Course taught by: Klaus Mosegaard.""")

def week1():
    text_dict = getText_prep(filename = text_path+'week1.md', split_level = 1)

    # Notes
    with st.expander('Lecture notes', expanded=False):
        cols = st.columns(2)
        cols[0].markdown(text_dict['Header 1'])
        cols[1].markdown(text_dict['Header 2'])

        st.markdown(text_dict['Examples'])
        st.markdown(text_dict['Header 3'])

    

    # Excercise
    st.markdown(text_dict['Ex 1'])

    ## read data
    path_data = 'assets/inverse_problems/data/gravdata.txt'
    arr = np.loadtxt(path_data)
    xs, d_obs = arr[:,0], arr[:,1]

    ## display data
    with st.expander('data', expanded=False):
        df = pd.DataFrame(arr.copy(), columns=['x','d_obs (1e9)'])
        df['d_obs (1e9)'] *= 1e9
        df.T

    ## Discretization
    st.markdown(text_dict['Ex 2'])

    ## calc and show G
    G = G_matrix(xs=xs, zs=np.arange(100))
    cols = st.columns(2)
    cols[0].markdown(text_dict['Ex 3'])
    cols[1].pyplot(contour_of_G(G.T))

    ## calc and show ms
    eps_space = np.logspace(-13, -9, 60)
    ms = getParams(G, d_obs, eps_space)

    fig, ax = plt.subplots(figsize=(8,3))
    ax.contourf(ms,10, cmap=plt.cm.inferno)
    fig.set_facecolor('lightgray')
    tick_locs = np.arange(len(eps_space))[::len(eps_space)//10]
    tick_vals = np.round(eps_space,13)[::len(eps_space)//10]
    ax.set(yticks=tick_locs,yticklabels=tick_vals)
    ax.set_ylabel('epsilon', color='black')
    ax.set_xlabel('depth', color='black')

    plt.close()
    cols = st.columns(2)
    cols[0].pyplot(fig)
    cols[1].markdown(text_dict['Ex 4'])
    

    ## Find minimum 
    fig = find_minimum(G, ms, d_obs, eps_space,
                        data_error = [10**(-9)] * 18,
                        data_error1 = [10**(-8)] * 18)


    st.markdown(text_dict['Ex 5'])
    st.pyplot(fig)
    

    
    
def week2():
    text_dict = getText_prep(filename = text_path+'week2.md', split_level = 1)

    # Notes
    with st.expander('Lecture notes monday', expanded=False):
        #cols = st.columns(2)
        st.markdown(text_dict['Header 1'])


    # Ex
    with st.expander('Excercise: The Good, The Bad, and The Ugly ', expanded=False):
        st.markdown("""
            ## Error Propagation in Inverse Problems 
            ### The Good, The Bad, and The Ugly 

            $$
                d = Gm
            $$
            """)

        inv = np.linalg.inv

        cols = st.columns(3)

        
        G = np.array([[1.0, 0.0],[0.0, 0.7]])
        rank = np.linalg.matrix_rank(G)

        d_pure = np.array([[0.500],[0.001]])
        m_pure = inv(G)@d_pure
        
        cols[0].write('#### The good')
        cols[0].write(r"""The matrix $G$ """)
        cols[0].write(G) 
        cols[0].write(f"""has rank= {rank}"""),
        cols[0].write(r"""Given the data, $d_\text{pure}$""")
        cols[0].write(d_pure)
        cols[0].write(r"we obtain the parameter vector $m_\text{pure}$ : ")
        cols[0].write(m_pure)

        cols[0].write('##### Now lets add some noise:')
        n = np.array([[0.008],[0.011]])
        n_norm = np.linalg.norm(n)
        cols[0].write('n_norm')
        cols[0].write(n_norm)

        d_norm = np.linalg.norm(d_pure)


        cols[0].write('signal to noise ratio: ')
        cols[0].write(d_norm/n_norm)


        d = d_pure + n

        m = inv(G) @ d
        cols[0].write('m:')
        cols[0].write(m)

        propagated_noise = np.linalg.norm(m - m_pure)
        cols[0].write('propagated_noise')
        cols[0].write(propagated_noise)
        cols[0].write('ratio')
        cols[0].write(propagated_noise/np.linalg.norm(m_pure))


        # The bad
        cols[1].write('#### The bad')
        G_B = np.array([[1.0, 0.0],[0.002, 0.0]])
        rank = np.linalg.matrix_rank(G_B)

        d_pure = np.array([[0.500],[0.001]])
        m_pure = inv(G) @ d_pure
        
        cols[1].write(r"""The matrix $G_B$ """)
        cols[1].write(G_B) 
        cols[1].write(f"""has rank= {rank}"""),
        cols[1].write(r"""Given the data, $d_\text{pure}$""")
        cols[1].write(d_pure)
        cols[1].write(r"we obtain the parameter vector $m_\text{pure}$ : ")
        cols[1].write(m_pure)


        cols[2].write('#### The ugly')
        G = np.array([[1.0, 0.0],[0.002, 10e-24]])
        rank = np.linalg.matrix_rank(G)

        d_pure = np.array([[0.500],[0.001]])
        m_pure = inv(G)@d_pure
        
        cols[2].write(r"""The matrix $G$ """)
        cols[2].write(G) 
        cols[2].write(f"""has rank= {rank}"""),
        cols[2].write(r"""Given the data, $d_\text{pure}$""")
        cols[2].write(d_pure)
        cols[2].write(r"we obtain the parameter vector $m_\text{pure}$ : ")
        cols[2].write(m_pure)

        cols[2].write('##### Now lets add some noise:')
        n = np.array([[0.008],[0.011]])
        n_norm = np.linalg.norm(n)
        cols[2].write('n_norm')
        cols[2].write(n_norm)

        d_norm = np.linalg.norm(d_pure)


        cols[2].write('signal to noise ratio: ')
        cols[2].write(d_norm/n_norm)


        d = d_pure + n

        m = inv(G) @ d
        cols[2].write('m:')
        cols[2].write(m)

        propagated_noise = np.linalg.norm(m - m_pure)
        cols[2].write('propagated_noise')
        cols[2].write(propagated_noise)
        cols[2].write('ratio')
        cols[2].write(propagated_noise/np.linalg.norm(m_pure))


    # excercise: entropy
    with st.expander('Excercise: entropy', expanded=False):
        st.markdown(r"""
            define the entropy of a probability density 𝑓(𝑥) as: 
            $$
                H(f) = -\int_X f(x) \log f(x) dx
            $$    
            since its a pdf $\int_X f(𝑥) dx = 1$
            """)

    with st.expander('Lecture notes Wednesday', expanded=False):
        st.markdown(text_dict['Header 2'])

    cols = st.columns(2)
    cols[0].markdown(text_dict['assignment'])
    


    x = np.linspace(0,1, 10)
    y = np.linspace(0,1, 10)
    X, Y = np.meshgrid(x,y)
    ground = np.zeros((x.shape[0],y.shape[0]))
    ground[(.4<X) * (X<.6) * (.3<Y) * (Y<.7)] = 1

    fig, ax = plt.subplots()
    ax.imshow(ground)
    cols[1].pyplot(fig)
    plt.close()



    speed_outside_medium = 1.
    speed_inside_medium = .5

    # rays should be angled at 45 degrees, so lets just keep dx and dy the same. This will let us travel along the matrix diagonal.

    # figure out the matrix which connects this matrix of speeds with the parameters...

def ass1():
    import numpy as np
    import matplotlib.font_manager
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt
    import streamlit as st
    from matplotlib.transforms import Bbox

    dark_color = (0,.1,.15)

    def getText_prep_1(filename = text_path+'linear_tomography.md', split_level = 1):
        """
            get text from markdown and puts in dict
        """
        # extra function
        with open(filename,'r', encoding='utf8') as f:
            file = f.read()
        level_topics = file.split('\n'+"#"*split_level+' ')
        text_dict = {i.split("\n")[0].replace("#"*split_level+' ','') : 
                    "\n".join(i.split("\n")[1:]) for i in level_topics}
        
        return text_dict  

    def coverImage(n_seismographs=20, b = {'left':1,'right':4,'bot':1,'top':5}):
        def makeGround(x, y ,b = b):
            Z =  np.zeros((len(x),len(y)))
            X, Y = np.meshgrid(x,y)
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    if (b['left']< xi < b['right']) and (b['bot'] < yj < b['top']):
                        Z[i,j] = 1
            m = Z.copy().flatten()

            dx = x[1]-x[0]
            dy = y[1]-y[0]
            dist = (dx**2+dy**2)**.5 * 1000 
            Z *= dist * ((1/5000) - (1/5200))
            #Z
            return Z, X, Y, m

        def traceRays(x, y, Z, seismograph_locs):
            n= len(seismograph_locs)
            x_fine = np.linspace(min(x),max(x), n+1)
            y_fine = np.linspace(min(y),max(y), n+1)

            def linear(x, a, b):
                return a*x + b
            rays = {}
            ray_values_left = np.zeros(n)
            ray_values_right = np.zeros(n)
            G_left = np.zeros((n, n, n))
            for idx, s_loc in enumerate(seismograph_locs):
                # coming from the left
                xi_left = x_fine[x_fine<s_loc]
                yi_left = linear(xi_left, -1,s_loc)
                #then we need to check which box each is in 

                ii_left = [np.argmin(abs(x-i)) for i in xi_left]
                jj_left = [np.argmin(abs(y-j)) for j in yi_left]
                #ii_left
                #G_left[s_loc]
                #a, b  = np.argwhere(x_fine<s_loc), np.argwhere(x_fine<s_loc)
                #a
                ray_values_left[idx] = sum([Z[i,j] for i, j in zip(ii_left, jj_left)])
        
                # coming from the right
                xi_right = x_fine[x_fine>s_loc]
                yi_right = linear(xi_right, 1,-s_loc)

                ii_right = [np.argmin(abs(x-i)) for i in xi_right]
                jj_right = [np.argmin(abs(y-j)) for j in yi_right]
                ray_values_right[idx] = sum([Z[i,j] for i, j in zip(ii_right, jj_right)])
                
                rays[s_loc] = (xi_left, yi_left, xi_right, yi_right)

            #G_left
            ray_values = np.concatenate([ray_values_left, ray_values_right])
            return rays, ray_values_left, ray_values_right , ray_values

        def plot():
            def plot_rays(rays, ax=plt):
                for i in rays:
                    (xi_l, yi_l, xi_r, yi_r) = rays[i]

                    ax.plot(xi_l, yi_l, c="white", lw=0.75)
                    ax.plot(xi_r, yi_r, c="white", lw=0.75)

            fig, ax = plt.subplots(2,1)
            fig = plt.figure(constrained_layout=True)

            gs = GridSpec(3, 1, figure=fig)
            ax = [fig.add_subplot(gs[:2]), fig.add_subplot(gs[2])]
            ax[0].scatter(X, Y, c=Z.T, marker=',', cmap='winter', s=(400/n_seismographs)**2, alpha=.7)

            plot_rays(rays, ax[0])
            ax[0].set_ylim(ax[0].get_ylim()[::-1])
            
            #for axi in ax:
                #axi.set(facecolor=dark_color)

            ax[0].set_xlabel('x', color='white',
            size=14)
            ax[0].set_ylabel('depth', color='white',
            size=14)
            
            
            ax[1].set_xlabel('Siesmograph index', color='white', 
            size=14)
            ax[1].set_ylabel('$t_\gamma$', color='white', 
            size=14)
            ax[1].bar(np.arange(len(d_left))-0.35, d_left,width=.25)
            ax[1].bar(np.arange(len(d_right))+0.35, d_right,width=.25)
            #fig.set_facecolor(dark_color)
            plt.tight_layout()
            return fig

        x = np.linspace(0,14, n_seismographs)
        y = np.linspace(0,12, n_seismographs)

        Z, X, Y, m = makeGround(x,y)
        seismograph_locs = np.linspace(1,max(x)-1, n_seismographs)

        rays, d_left, d_right, d_obs = traceRays(x, y, Z, seismograph_locs)
        
        fig = plot()
        return d_obs, fig, m,

    def make_G(N=13):
        G_right = [np.eye(N,k=1+i).flatten() for i in range(N-2)]
        G_left = [np.flip(np.eye(N,k=-(1+i)), axis=0).flatten() for i in range(N-2)]
        
        z = np.zeros((1,N**2))
        G = np.concatenate([z, G_left[::-1],z, z, G_right,z])

        G *= 2**.5 * 1000
        return G

    def plot_G_summed(G,N):
        G = G.copy()/ (2**.5 * 1000)
        fig, ax = plt.subplots( figsize=(6,3))
        
        plt.imshow(G.sum(axis=0).reshape(N,N), extent=(0,N,N,0))
        plt.xticks([0,N//2,N])
        plt.colorbar()
        return fig

    def make_m(top=2, bot=8, left=3, right=7):
        '''the section of earth under investigation'''
        M = np.zeros((N,N))
        M[top:bot, left:right] = (1/5000) - (1/5200)
        return M.flatten()

    def forward(G, m, noise_scale=1/18,seed=42):
        'returns d'
        def addNoise(t_pure, seed):
            np.random.seed(seed)
            noise = np.random.randn(len(G))
            noise /= np.linalg.norm(noise)
            n = noise * noise_scale * np.linalg.norm(t_pure)
            t_obs = t_pure + n
            return t_obs, np.linalg.norm(n)
        
        d_pure = G@m
        d_obs, n_norm = addNoise(d_pure, seed)
        return d_obs, n_norm

    def genDataPlot(m_true, d_obs, st=st):
        fig, ax = plt.subplots(1, 3, figsize=(12,3))

        ax[0].imshow(G, aspect=6)
        ax[0].set_title('$G$', color='white')
        ax[1].imshow(m_true.reshape(N,N), extent=(0,N,N,0))
        ax[1].set_title('$m$', color='white')  

        ax[2].set_title(r'$d_{obs}$', color='white')
        ax[2].bar(np.arange(N)-0.35, d_obs[:N],width=.35 , label='left')
        ax[2].bar(np.arange(N)+0.35, d_obs[N:],width=.5, label='ight')
        ax[2].legend()
        ax[2].set_xticks(*[range(N)[::4]]*2)
        ax[2].set_xlabel('Detector number', color='white')
        st.pyplot(fig)

    def backward(d_obs, n_norm, G, n_eps=10):
        epss, offs, ms = np.logspace(-3, 4, n_eps), [], []

        for eps in epss:
            m = np.linalg.inv(G.T@G  + eps**2*np.eye(N**2)) @ (G.T @ d_obs) #least square
            off = np.abs( np.linalg.norm(G@m - d_obs) - n_norm) # residual
            offs.append(off);  ms.append(m)

        
        epss_zoom = np.linspace(epss[np.argmin(offs)-1],
                                    epss[np.argmin(offs)+1],
                                    n_eps*2)
        offs_zoom, ms = [], []
        for eps in epss_zoom.copy():
            m = np.linalg.inv(G.T@G  + eps**2*np.eye(N**2)) @ (G.T @ d_obs) #least square
            off = np.abs( np.linalg.norm(G@m - d_obs) - n_norm) # residual
            offs_zoom.append(off);  ms.append(m)
        m_opt = ms[np.argmin(offs_zoom)]
        return m_opt, epss, offs, epss_zoom, offs_zoom

    def plot_m_opt(m_opt):
        fig, ax = plt.subplots()
    

        
        ax.imshow(m_opt.reshape(N,N), extent=(0,N,N,0))
        
        ax.set(xticks=np.arange(2,14,2)-.5, xticklabels=range(2,14,2))
        ax.set_title('Pred. density of earth', color='white')
        ax.set_xlabel('x', color='white')

        return fig

    def plot_epss(epss, offs, epss_zoom, offs_zoom):
        fig, ax1 = plt.subplots()
        #T_E = np.arange(1,max(T)+1,1)
        # The data.
        ax1.plot(epss, offs, c='white')
        ax1.set_xlabel(r'$T\,/\mathrm{K}$')
        ax1.set_ylabel(r'$C_p\,/\mathrm{J\,K^{-1}\,mol^{-1}}$')
        

        ax1.set(yscale = 'log', xscale = 'log',facecolor=dark_color)
        
        ax1.set_xlabel('epsilon', color='white', fontsize=14)
        ax1.set_ylabel('|t_obs - G@m|', color='white', fontsize=14)

        # Inset figure of low-temperature fit, located by data coordinates in ax1.
        bb_data_ax2 = Bbox.from_bounds(min(epss)*1e1, min(offs)*10, max(epss)*1e-3, max(offs)*0.3)
        disp_coords = ax1.transData.transform(bb_data_ax2)
        fig_coords_ax2 = fig.transFigure.inverted().transform(disp_coords)
        bb_ax2 = Bbox(fig_coords_ax2)
        ax2 = fig.add_axes(bb_ax2)
        ax2.set_facecolor('beige')
        ax2.plot(epss_zoom, offs_zoom, c='black')

        return fig



    # main render
    text_dict = getText_prep_1()

    "# Linear Tomography"
    st.markdown(text_dict['intro'])

    def sliders_5():
        cols = st.columns(5)
        top = cols[0].slider('top', 0., 11., 3.18)
        bot = cols[1].slider('bot', 0., 11., 4.5)
        left = cols[2].slider('left', 0., 11., 9.)
        right = cols[3].slider('right', 0., 13., 13.)
        n_seismographs = cols[4].select_slider('n seis.', range(12,99,20))
        return top, bot, left, right, n_seismographs
    top, bot, left, right, n_seismographs = sliders_5()

    d, fig, m = coverImage(n_seismographs, b = {'left':min(left,right), 'right':max(left,right), 'bot':min(bot,top), 'top':max(bot,top)})
    st.pyplot(fig)



    ## G
    '### Generating data'
    cols=st.columns(2)
    cols[0].markdown(text_dict['Generating data'])

    st.code(text_dict['G code'])
    cols_3=st.columns(3)

    N = cols_3[2].slider('N', 1,50,12)
    G = make_G(N)
    fig = plot_G_summed(G, N); cols[1].pyplot(fig)

    top = cols_3[0].slider('top', 0, N, 4)
    bot = cols_3[0].slider('bot', 0, N, 5)
    left = cols_3[1].slider('left', 0, N, 3)
    right = cols_3[1].slider('right', 0, N, 5)
    m_true = make_m(min(top, bot), max(top, bot), min(left, right), max(left, right))

    noise_scale = 1 / cols_3[2].slider('1/Noise_scale', 1,50,18)

    d_obs, n_norm = forward(G, m_true, noise_scale, 111)
    genDataPlot(m_true, d_obs, st=st)


    ## predicting m
    '### Predicting $m$'
    cols = st.columns(2)
    cols[0].markdown(text_dict['Predicting m'])

    n_eps = cols[1].slider('n_eps', 2, 99, 10)
    '**This yields the prediction of the internal structure of the earth shown below.**'
    m_opt, epss, offs, epss_zoom, offs_zoom = backward(d_obs, n_norm, G, n_eps)
    # maybe add an insert
    fig = plot_m_opt( m_opt,)

    fig2 = plot_epss( epss, offs, epss_zoom, offs_zoom)
    cols[1].pyplot(fig2)
    st.pyplot(fig)

    st.markdown(text_dict['A delta function'])

# Navigator
topic_dict = {
    'Linear Tomography' : ass1,
    'week 1': week1,
    'Week 2': week2, 
  }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()



