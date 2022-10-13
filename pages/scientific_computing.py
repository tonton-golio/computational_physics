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
	\text{total error} &= \hat{f}(\hat{x})-f(x)\\
	&= \left(\hat{f}(\hat{x})-f(\hat{x})\right) &+&\left(f(\hat{x})-f(x)\right)\\
	&= \text{computational error} &+& \text{propagated data error}\\
	E_\text{tot} &= E_\text{comp} &+& E_\text{data}
	\end{align*}''')

	st.markdown(r'''
	**Truncaiton error and rounding error** are the two parts of computational error. 
	Truncation error stems from truncating infinite series, or replacing derivatives 
	with finite differences. Rounding error is like the error from like floating point accuracy.

	Truncation error, $E_\text{trunc}$ can stem from; 
	* simplification of physical model
	* finite basis sets
	* truncations of infinite series
	* ...

	*Example:* computational error of $1^\text{st}$ order finite difference
	''')
	st.latex(r'''
	\begin{align*}
	f'(x) \approx \frac{f(x+h)-f(x)}{h}\equiv\hat{f'(x)}\\
	f(x+h) = f(x)+hf'(x)+ \frac{h^2}{2}f''(\theta), |\theta-x|\leq h\\
	\frac{f(x+h)-f(x)}{h} = f'(x) + \frac{h}{2}f''(\theta)\\
	\hat{f'(x)} - f'(x) = \frac{h}{2}f''(\theta), \text{let} \equiv \text{Sup}_{|\theta-x|\leq h} (f''(\theta))\\
	E_\text{trunc} = \hat{f'(x)}-f'(x)\leq \frac{M}{2}h\sim O(h)
	\end{align*}''')

	st.markdown(r'''But what about the rounding error? 
		(assume R.E. for $f$ is $\epsilon \Rightarrow E_\text{ronud} \leq \frac{2\epsilon}{h}\sim 0(\frac{1}{h})$
		''')

	st.latex(r'''
	\begin{align*}
		E_\text{comp} = \frac{M}{2}h + \frac{2\epsilon}{h}\\
		0 = \frac{d}{dh}E_\text{comp} = \frac{M}{2}-\frac{2\epsilon}{h^2}\\
		\frac{M}{2} = \frac{2\epsilon}{h^2} 
		\Leftrightarrow h^2 = \frac{4\epsilon}{M}\Leftrightarrow h_\text{optimal} = 2\sqrt{\frac{\epsilon}{M}}
	\end{align*}''')

	try:
		st.image('https://farside.ph.utexas.edu/teaching/329/lectures/img320.png')
	except:
		st.image('assets/images/errors.png')


	st.markdown(r'''
	**Forward vs. backward error**

	foward error is the error in the output, backward error is the error in the input.

	**Sensitivity and conditioning**

	Condition number: $COND(f) \equiv \frac{|\frac{\Delta y}{y}|}{|\frac{\Delta x}{x}|} = \frac{|x\Delta y|}{|y\Delta x|}$

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




						