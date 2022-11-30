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


# Navigator
topic_dict = {
    'week 1': week1,
    'Week 2': week2,
  }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()



