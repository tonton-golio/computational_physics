from utils.utils_global import *

from scipy.stats import multivariate_normal
import qutip as qt

import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

import plotly.graph_objects as go
from plotly.subplots import make_subplots

text_path = 'assets/quantum_optics/text/'


###########################################
"""
TODO

[-] marginal distribution of wigner 2d
[-] Wigner 3d axis label 
[-] Wigner 3d color bar label 
[] make code interface and get user-defined state
[] time evolution animation with plotly
[] make all plot with plotly
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

    #mean = [real_alpha, imag_alpha]
    radius = 1/2
    #cov = [[radius, 0], [0, radius]]
    #r=1.25
    #x, y= np.mgrid[xrange[0]*r:xrange[1]*r:100j, yrange[0]*r:yrange[1]*r:100j]
    #pos = np.dstack((x, y))
    #z = multivariate_normal.pdf(pos, mean=mean, cov=cov)

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
        #ax.contourf(x, y, z, levels=100)

        x = np.linspace(0, real_alpha)
        y = np.linspace(0, imag_alpha)
        ax.plot(x, y, c='k', ls='-', lw=1)

        c = patches.Circle((real_alpha, imag_alpha), radius=radius, edgecolor='k', facecolor='none')
        ax.add_artist(c)

        ax.set_xlim(xrange)
        ax.set_ylim(yrange)

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
    #ax.contourf(x, y, z, levels=100)

    c = patches.Circle((0, 0), radius=radius, edgecolor='k', facecolor='none')
    ax.add_artist(c)

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    ax.set_aspect('equal')
    ax.set_xlabel('$\\hat{X}_1$')
    ax.set_ylabel('$\\hat{X}_2$')

    st.pyplot(fig)


####################
## Wigner
####################

@function_profiler
def _plot_wigner(W, xvec, pvec, three_dimensional, theme):
    W_original = W.copy()
    bin_width_x = xvec[1]-xvec[0]
    bin_width_p = pvec[1]-pvec[0]
    W = W*bin_width_x*bin_width_p
    W = np.round(W, 8)

    if theme=="Light":
        cmap="RdBu_r"
        colorscale="RdBu_r"
    else:
        cmap="coolwarm"
        colorscale="balance"

    if not three_dimensional:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('none')

        if cmap=='wigner':
            cmap = qt.wigner_cmap(W)
    
        cbar = ax.contourf(xvec, pvec, W, 
                           levels=100, cmap=cmap, norm=colors.CenteredNorm())
        fig.colorbar(cbar, label='Quasi probability')

        ax.set_aspect('equal')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$p$')
        #ax.xaxis.set_tick_params(rotation=-90)

        divider = make_axes_locatable(ax)
        ax_marginalx = divider.append_axes("top", 0.8, pad=0.25, sharex=ax)
        ax_marginalp = divider.append_axes("right", 0.8, pad=0.25, sharey=ax)

        ax_marginalx.xaxis.set_tick_params(labelbottom=False)
        ax_marginalp.yaxis.set_tick_params(labelleft=False)
        #ax_marginalp.xaxis.set_tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
        ax_marginalp.xaxis.set_tick_params(labelrotation=-90)

        ax_marginalx.plot(xvec, W.sum(axis=0))
        ax_marginalp.plot(W.sum(axis=1), pvec)

        ax_marginalx.set_ylabel("$\\left| \\psi (x) \\right|^2$")
        ax_marginalp.set_xlabel("$\\left| \\psi (p) \\right|^2$", )

        st.pyplot(fig)

    else:
        fig = go.Figure()
        # main surface plot
        fig.add_trace(
                go.Surface(z=W, x=xvec, y=pvec, 
                           cmid=0, 
                           #colorscale=colorscale,
                           colorbar=dict(
                               title="Quasi probability",
                               titleside="right"))
                )
        fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
                )
        # additional contour plot on xy-plane
        #fig.update_traces(contours_z=
        #                  dict(show=True, usecolormap=True,
        #                       highlightcolor="limegreen", project_z=True,
        #                       )
        #                  )
        fig.update_layout(scene=dict(
                xaxis_title="x",
                yaxis_title="p",
                zaxis_title="W(x, p)",
                ))
        st.plotly_chart(fig)


@function_profiler
def plot_wigner_number(theme, Ndims=100, three_dimensional=False):
    st.write("$\\Ket{\\psi}=\\Ket{n}$")
    plot_range = 10.0
    xvec = np.linspace(-plot_range, plot_range, 200)
    pvec = xvec.copy()
    n = st.slider('Number of photon', 0, 25, 0)

    psi = qt.basis(Ndims, n)
    W = qt.wigner(psi, xvec, pvec) 

    _plot_wigner(W, xvec, pvec, three_dimensional, theme)


@function_profiler
def plot_wigner_coherent(theme, Ndims=100, three_dimensional=False):
    st.write("$\\Ket{\\psi} = \\Ket{\\alpha} = \\hat{D}(\\alpha) \\Ket{0}$")
    st.write("$\\hat{D}(\\alpha) = \\exp \\left( \\alpha \\hat{a}^\dag - \\alpha^* \\hat{a} \\right)$")
    xrange = [-5.0, 5.0]
    yrange = [-5.0, 5.0]
    cols = st.columns(2)
    real_alpha = cols[0].slider('Real part of complex eigenvalue', xrange[0], xrange[1], 0.0)
    imag_alpha = cols[1].slider('Imaginary part of complex eigenvalue', yrange[0], yrange[1], 0.0)
    alpha = complex(real_alpha, imag_alpha)

    plot_range = xrange[1]*2
    xvec = np.linspace(-plot_range, plot_range, 100)
    pvec = xvec.copy()

    psi = qt.coherent(Ndims, alpha)
    W = qt.wigner(psi, xvec, pvec) 
    
    _plot_wigner(W, xvec, pvec, three_dimensional, theme)


@function_profiler
def plot_wigner_squeezed(theme, Ndims=100, three_dimensional=False):
    st.write("$\\Ket{\\psi}=\\Ket{\\alpha, \\xi} = \\hat{D}(\\alpha) \\hat{S} (\\xi)\\Ket{0}$")
    st.write("$\\hat{D}(\\alpha) = \\exp \\left( \\alpha \\hat{a}^\dag - \\alpha^* \\hat{a} \\right)$")
    st.write("$\\hat{S}(\\xi) = \\exp \\left[ \\frac{1}{2} \\left( \\xi^* \\hat{a}^2 - \\xi {\\hat{a}^\dag}^2 \\right) \\right]$")
    st.write("$\\xi = r e^{i\\theta}$")

    cols = st.columns(2)

    xrange = [0.0, 5.0]
    yrange = [0.0, 5.0]

    real_alpha = cols[0].slider('Real part of complex eigenvalue of displacement', xrange[0], xrange[1], 0.0)
    imag_alpha = cols[1].slider('Imaginary part of complex eigenvalue of displacement', yrange[0], yrange[1], 0.0)
    alpha = complex(real_alpha, imag_alpha)

    r = cols[0].slider('Squeeze parameter r', 0.0, 1.5, 0.0)
    theta = cols[1].slider('Squeeze parameter theta (degree)', 0.0, 2.0*np.pi, 0.0)
    xi = r*complex(np.cos(theta), np.sin(theta))

    plot_range = xrange[1]*2
    xvec = np.linspace(-plot_range, plot_range, 200)
    pvec = xvec.copy()

    psi = qt.displace(Ndims, alpha) * qt.squeeze(Ndims, xi) * qt.basis(Ndims, 0)
    W = qt.wigner(psi, xvec, pvec) 

    _plot_wigner(W, xvec, pvec, three_dimensional, theme)


@function_profiler
def plot_wigner_cat(theme, Ndims=100, three_dimensional=False):
    st.write("$\\Ket{\\psi}=\\frac{1}{\\sqrt{2+2 e^{-2 \\left|\\alpha\\right|^2} \\cos \\Phi}}\\left(\\Ket{\\alpha} + e^{i\\Phi} \\Ket{-\\alpha}\\right)$")
    xrange = [0.0, 5.0]
    yrange = [0.0, 5.0]
    cols = st.columns(2)
    real_alpha = cols[0].slider('Real part of complex eigenvalue (cat)', xrange[0], xrange[1], 2.5)
    imag_alpha = cols[1].slider('Imaginary part of complex eigenvalue (cat)', yrange[0], yrange[1], 0.0)
    alpha = complex(real_alpha, imag_alpha)
    phi = st.slider('phi', 0.0, 2*np.pi, 0.0)
    imag_phi = complex(0.0, phi)


    plot_range = xrange[1]*2
    xvec = np.linspace(-plot_range, plot_range, 250)
    pvec = xvec.copy()

    ket_alpha = qt.coherent(Ndims, alpha)
    ket_alpha_minus = qt.coherent(Ndims, -alpha)
    psi = ket_alpha + np.exp(imag_phi)*ket_alpha_minus
    psi = psi.unit()
    W = qt.wigner(psi, xvec, pvec) 

    _plot_wigner(W, xvec, pvec, three_dimensional, theme)


