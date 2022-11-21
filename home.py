import streamlit as st
import pandas as pd
st.set_page_config(page_title="Computational Physics", 
	page_icon=":taxi:", 
	layout="wide", 
	initial_sidebar_state="collapsed", 
	menu_items=None)
st.markdown(r"""
	# Computational Physics
	*by*: Anton Gersvang Golles, Riz Fernando Noronha & Yoshiaki Horiike.

	Please contribute on [github](https://www.github.com/tonton-golio/computational_physics)
	""")
st.image('assets/images/mandel_front.png', width=420)


with st.expander('TODO',expanded=False):
	st.markdown(r"""
	- [] dark mode by default (or style sheet for each theme)
	- [] add simulations to SciComp
		- [] solvers
		- [] minimization
		- [x] eigenstates
	""")
