import streamlit as st
st.set_page_config(page_title="Scientific Computing", 
	page_icon="🧊", 
	layout="wide", 
	initial_sidebar_state="collapsed", 
	menu_items=None)
st.title("Scientific Computing")

with st.expander('Bounding Errors', expanded=False):
	st.markdown('''
	### Bounding Errors: 

	https://www.youtube.com/watch?v=GFhhRdF54eI
	
	**Sources of approximation** include modelling, 
	empirical measurements, previous computations, truncation/discretization, rounding.

	**Absolute error** and **relative error** are different in the obvious manner:
	''')
	st.latex(r'''
		\begin{align*}
		\text{abs. error } &= \text{approx. value }-\text{true value}\\
		\text{rel. error } &= \frac{\text{abs. error}}{\text{true value}}.
		\end{align*}
	''')

	st.markdown('''
	**Data error and computational errror**, the hats indicate an approximation;
	''')

	st.latex(r'''
	\begin{align*}
	\text{total error} &= \hat{f(\hat{x})-f(x)}\\
	&= \hat{f(\hat{x})-f(\hat{x})} &+(f(\hat{x})-f(x)\\
	&= \text{computational error} & \text{prpagated data error}\\
	\end{align*}''')

	st.markdown('''
	**Truncaiton error and rounding error** are the two parts of computational error. 
	Truncation error stems from truncating infinite series, or replacing derivatives 
	with finite differences. Rounding error is like the error from like floating point accuracy.

	**Forward vs. backward error**

	**Sensitivity and conditioning**

	**Stability and accuracy**

	**Floating point numbers**, a video from like 8 years ago by numberphile: 
	https://www.youtube.com/watch?v=PZRI1IfStY0.
	*floating point is scientic notation in base 2*. 

	Another video: https://www.youtube.com/watch?v=f4ekifyijIg. 
	* Fixed points have each bit correspond to a specific scale.
	* floating point (32 bit) has: 1 sign bit (0=postive, 1=negative), 8 exponent bits, 
	and 23 mantissa bits. 

	* another video on fp addition: https://www.youtube.com/watch?v=782QWNOD_Z0

	overflow and underflow; refers to the largest and smallest numbers that can be 
	contained in a floating point.

	**Complex arithmatic**

	''')

with st.expander('Linear Equations', expanded=False):
	st.markdown('Linear Equations')
with st.expander('Linear Least Squares', expanded=False):
	st.markdown('Linear Equations')
with st.expander('Eigensystems', expanded=False):
	st.markdown('Linear Equations')
with st.expander('Nonlinear Equations	Optimization', expanded=False):
	st.markdown('Linear Equations')
with st.expander('Initial Value Problems for Ordinary Differential Equations', expanded=False):
	st.markdown('Linear Equations')
with st.expander('Partial Differential Equations	FFT and Spectral Methods', expanded=False):
	st.markdown('Linear Equations')




						
