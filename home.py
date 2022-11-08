import streamlit as st
import pandas as pd
st.set_page_config(page_title="Computational Physics", 
	page_icon=":taxi:", 
	layout="wide", 
	initial_sidebar_state="collapsed", 
	menu_items=None)
st.markdown(r"""
	# Computational Physics
	#### Notes and simulations for courses""")
st.image('assets/images/mandel_front.png', width=420)

st.markdown("""
### [github](https://www.github.com/tonton-golio/computational_physics)
###### Contributers
* Anton
* Riz
* Yoshiaki""")


with st.expander('TODO',expanded=False):
	st.markdown(r"""
	- [] dark mode by default (or style sheet for each theme)
	- [] optimize bethelattice percolation search to make nice plots and confirm p_c
	- [] find critical point for percolation on bethe lattice of different degrees
	- [] big runs for susceptibility in 2d ising, add dashed line to plot

	- [x] move functions from complex into complex_utils	
	- [x] make graphic for front page
	- [x] close plots
	- [x] get networks to work in browser
	""")
