import streamlit as st
import pandas as pd
st.set_page_config(page_title='Computational Physics',#"plz contribute", 
	page_icon=":microscope:", 
	layout="wide", 
	initial_sidebar_state="collapsed", 
	menu_items=None)
st.markdown(r"""
	# Computational Physics
	*by*: Anton Gersvang Golles, Riz Fernando Noronha & Yoshiaki Horiike.

	Please contribute on [github](https://www.github.com/tonton-golio/computational_physics)
	""")
st.image('assets/images/mandel_front.png', width=420)

import cProfile
import re
cProfile.run('re.compile("foo|bar")')