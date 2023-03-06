from utils.utils_global import *
import matplotlib.patches as patches
import matplotlib.colors as colors
import plotly.graph_objects as go
from scipy.stats import multivariate_normal
import qutip as qt

text_path = 'assets/quantum_optics/text/'


###########################################
"""
TODO

marginal distribution of wigner 2d
"""
# topic 4
@function_profiler
def plot_coherent_on_phase_space(plotly=False):
    xrange = [-10.0, 10.0]
    yrange = [-10.0, 10.0]
    cols = st.columns(2)
    real_alpha = cols[0].slider('Real part of complex eigenvalue', xrange[0], xrange[1], 0.0)
    imag_alpha = cols[1].slider('Imaginary part of complex eigenvalue', yrange[0], yrange[1], 0.0)
    st.write("$\\alpha={}+i({})$".format(real_alpha, imag_alpha))
    alpha_square = real_alpha**2+imag_alpha**2
    st.write("$|\\alpha|^2=n={}$".format(alpha_square))
    phi = np.arctan2(imag_alpha, real_alpha)
    phi_deg = np.rad2deg(phi)
    st.write("$\\phi={}={}\degree$".format(phi, phi_deg))
    delta_phi = np.arctan2(1/2, real_alpha**2+imag_alpha**2)
    delta_phi_deg = np.rad2deg(delta_phi)
    st.write("$\\Delta\\phi={}={}\degree$".format(delta_phi, delta_phi_deg))

    mean = [real_alpha, imag_alpha]
    radius = 1/2
    cov = [[radius, 0], [0, radius]]
    r=1.25
    x, y= np.mgrid[xrange[0]*r:xrange[1]*r:100j, yrange[0]*r:yrange[1]*r:100j]
    pos = np.dstack((x, y))
    z = multivariate_normal.pdf(pos, mean=mean, cov=cov)

    if plotly:
        fig = go.Figure(data=
                        go.Heatmap(
                            z=z, 
                            x=x, 
                            y=y,
                            zsmooth='best',
                        ),
        )
        fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
        )
        fig.update_layout(
            xaxis = dict(
                    tickmode = 'linear',
                    dtick = 20
                    ),
            yaxis = dict(
                    tickmode = 'linear',
                    dtick = 20
                    )
        )

        st.plotly_chart(fig)
    else:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('none')
        ax.contourf(x, y, z, levels=100)

        x = np.linspace(0, real_alpha)
        y = np.linspace(0, imag_alpha)
        ax.plot(x, y, c='w', ls=':')

        c = patches.Circle((real_alpha, imag_alpha), radius=radius, edgecolor='w', facecolor='none')
        ax.add_artist(c)

        ax.set_aspect('equal')
        ax.set_xlabel('$\\hat{X}_1$')
        ax.set_ylabel('$\\hat{X}_2$')

        st.pyplot(fig)


@function_profiler
def plot_number_on_phase_space():
    xrange = [-10.0, 10.0]
    yrange = [-10.0, 10.0]
    n = st.slider('Number of photon', 0, 100, 0)
    st.write("$n={}$".format(n))
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('none')

    mean = [0, 0]
    radius = np.sqrt(0.5*(0.5+n))
    cov = [[radius, 0], [0, radius]]
    r=1.25
    x, y= np.mgrid[xrange[0]*r:xrange[1]*r:100j, yrange[0]*r:yrange[1]*r:100j]
    pos = np.dstack((x, y))
    z = multivariate_normal.pdf(pos, mean=mean, cov=cov)
    ax.contourf(x, y, z, levels=100)

    c = patches.Circle((0, 0), radius=radius, edgecolor='w', facecolor='none')
    ax.add_artist(c)

    ax.set_aspect('equal')
    ax.set_xlabel('$\\hat{X}_1$')
    ax.set_ylabel('$\\hat{X}_2$')

    st.pyplot(fig)


@function_profiler
def plot_wigner_coherent(Ndimension=100, three_dimensional=False):
    xrange = [-5.0, 5.0]
    yrange = [-5.0, 5.0]
    cols = st.columns(2)
    real_alpha = cols[0].slider('Real part of complex eigenvalue', xrange[0], xrange[1], 0.0)
    imag_alpha = cols[1].slider('Imaginary part of complex eigenvalue', yrange[0], yrange[1], 0.0)
    alpha = complex(real_alpha, imag_alpha)

    plot_range = xrange[1]*2
    xvec = np.linspace(-plot_range, plot_range, 100)
    pvec = xvec.copy()

    psi = qt.coherent(Ndimension, alpha)
    W = qt.wigner(psi, xvec, pvec) 

    if not three_dimensional:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('none')

        wcmap = qt.wigner_cmap(W)
        cbar = ax.contourf(xvec, pvec, W, levels=100, cmap=wcmap)
        fig.colorbar(cbar, label='Quasi probability')

        ax.set_aspect('equal')
        ax.set_xlabel('$\\hat{X}_1$')
        ax.set_ylabel('$\\hat{X}_2$')

        st.pyplot(fig)

    else:
        fig = go.Figure(data=[go.Surface(z=W, x=xvec, y=pvec, cmid=0)])
        fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
                )
        #fig.update_traces(contours_z=
        #                  dict(show=True, usecolormap=True,
        #                       highlightcolor="limegreen", project_z=True
        #                       )
        #                  )
        fig.update_layout(
                xaxis_title="q",
                yaxis_title="p",
                #zaxis_title="$W(q, p)$",
                )
        st.plotly_chart(fig)


@function_profiler
def plot_wigner_number(Ndimension=100, three_dimensional=False):
    plot_range = 7.5
    xvec = np.linspace(-plot_range, plot_range, 100)
    pvec = xvec.copy()
    n = st.slider('Number of photon', 0, 20, 0)

    psi = qt.basis(Ndimension, n)
    W = qt.wigner(psi, xvec, pvec) 

    if not three_dimensional:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('none')

        wcmap = qt.wigner_cmap(W)
        cbar = ax.contourf(xvec, pvec, W, levels=100, cmap=wcmap)
        fig.colorbar(cbar, label='Quasi probability')

        ax.set_aspect('equal')
        ax.set_xlabel('$\\hat{X}_1$')
        ax.set_ylabel('$\\hat{X}_2$')

        st.pyplot(fig)

    else:
        fig = go.Figure(data=[go.Surface(z=W, x=xvec, y=pvec, cmid=0)])
        #fig.update_traces(contours_z=
        #                  dict(show=True, usecolormap=True,
        #                       highlightcolor="limegreen", project_z=True
        #                       )
        #                  )
        st.plotly_chart(fig)


def plot_wigner_cat(Ndimension=100, three_dimensional=False):
    xrange = [0.0, 3.0]
    yrange = [0.0, 3.0]
    cols = st.columns(2)
    real_alpha = cols[0].slider('Real part of complex eigenvalue (cat)', xrange[0], xrange[1], 1.0)
    imag_alpha = cols[1].slider('Imaginary part of complex eigenvalue (cat)', yrange[0], yrange[1], 0.0)
    alpha = complex(real_alpha, imag_alpha)

    plot_range = xrange[1]*2
    xvec = np.linspace(-plot_range, plot_range, 100)
    pvec = xvec.copy()

    psi = qt.coherent(Ndimension, alpha)
    psi_minus = qt.coherent(Ndimension, -alpha)
    cat = psi + psi_minus
    W = qt.wigner(cat, xvec, pvec) 
    
    if not three_dimensional:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('none')
        
        cbar = ax.contourf(xvec, pvec, W, levels=100, cmap=plt.cm.RdBu_r, norm=colors.CenteredNorm())
        fig.colorbar(cbar, label='Quasi probability')

        ax.set_aspect('equal')
        ax.set_xlabel('$\\hat{X}_1$')
        ax.set_ylabel('$\\hat{X}_2$')

        st.pyplot(fig)

    else:
        fig = go.Figure(data=[go.Surface(z=W, x=xvec, y=pvec, cmid=0)])
        #fig.update_traces(contours_z=
        #                  dict(show=True, usecolormap=True,
        #                       highlightcolor="limegreen", project_z=True
        #                       )
        #                  )
        st.plotly_chart(fig)
