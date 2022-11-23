from utils.utils_inverse import *
st.title('Inverse Problems')

from scipy.constants import *


def week1():
    text_dict = getText_prep(filename = text_path+'week1.md', split_level = 1)

    with st.expander('Week 1 notes', expanded=False):
        st.markdown(text_dict['Header 1'])

    
        st.markdown(text_dict['Examples'])
        st.markdown(text_dict['Header 2'])

    st.header('Week 1 excercise')

    # read data
    path_data = 'assets/inverse_problems/data/gravdata.txt'
    with open(path_data) as f:
        file = f.read().replace('E', ' ').split('\n')
    arr = np.array([line.split() for line in file]).astype(float)
    arr[:, 1] = arr[:, 1] * 10** (arr[:, 2])
    arr = arr[:, :2]
    with st.expander('data', expanded=False):
        st.table(arr)


    st.markdown(r"""
        ### discretize the integral:
        $$$
        d_j = \frac{∂g}{∂x} (x_j) =∫^∞_0 \frac{2Gz}{x_j^2 + z^2} ∆ρ(z) dz
        $$$

        #### Initial attempt
        So we wanna replace the integration with a sum
        $$
            d_j = \sum_i^n \frac{2Gz_i}{x_j^2 + z_i^2} ∆ρ(z_i)
        $$
        """)
    
    
   

    st.markdown(r"""
        #### Method 2
        $$
        \begin{align*}
            d_j^i = G\log
                \left(
                    \frac{z^{i2}_\text{base} + x_j^2}{z^{i2}_\text{top} + x_j^2}
                \right)
                \delta\rho_i
            \\
            d_j = \sum_i G\log
                \left(
                    \frac{z^{i2}_\text{base} + x_j^2}{z^{i2}_\text{top} + x_j^2}
                \right)
                \delta\rho_i
            \\
            G_{j,i} = G\log
                \left(
                    \frac{z^{i2}_\text{base} + x_j^2}{z^{i2}_\text{top} + x_j^2}
                \right)
        \end{align*}
        $$
        """)

    def G_ij(zi, xj): 
        return gravitational_constant * np.log( ((zi+1)**2 + xj**2) / ( zi**2 + xj**2 ) )

    G = np.array([[G_ij(zi, xj) for zi in np.arange(100)] for xj in arr[:, 0]])
    print(G)
    fig, ax = plt.subplots(figsize=(12,4))
    ax.contourf(G)
    plt.gca().invert_yaxis()
    plt.close()
    st.pyplot(fig)

    st.markdown(r"""
        #### now we have G, we can move on to the next step
        $$
            m = [G^TG + \epsilon^2I]^{-1} G^T d_\text{obs}
        $$
        this comes from minimizing: some regularized loss function, see image on phone
        """)

    m = []
    eps_space = np.logspace(-12, -10, 200)
    for epsilon in eps_space:
        m_e = np.linalg.inv(G.T@G + epsilon**2 * np.eye(100) ) @  (G.T @arr[:,1])

        m.append(m_e)

    m = np.array(m)

    fig, ax = plt.subplots(figsize=(12,3))
    ax.contourf(m)
    #ax.set_yticks(range(len(eps_space)), np.round(eps_space,17))
    #ax.set(yscale='log')
    m.shape
    plt.close()
    st.pyplot(fig)


    st.markdown('#### Its the final countdown')


    res = [abs(np.linalg.norm(G @ m_e - arr[:,1]) - np.linalg.norm( [10**(-9)] * 18)) for m_e in m]
    fig, ax = plt.subplots(figsize=(12,3))
    plt.plot(eps_space, res)
    ax.set(xlabel= r'$\epsilon$', xscale='log', yscale='log', ylabel=r'$||\bar{\Omega}||$', #xlim=(1e-15, 1e-11)
        )

    plt.close()
    st.pyplot(fig)
    
    






# Navigator
topic_dict = {
    'week 1': week1,
  }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()



