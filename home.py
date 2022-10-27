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
try:
	st.image('streamlit_app1/assets/images/cover.jpg')
except:
	st.image('assets/images/cover.jpg')

st.markdown("""
### [github](https://www.github.com/tonton-golio/computational_physics)
###### Contributers
* Anton
* Riz
* Yoshiaki""")


with st.expander('TODO',expanded=False):
	st.markdown(r"""
	- [] clean up functions 😅
	- [] add feedback box on each page?
	- [] get networks to work in browser
	- [] make graphic for front page
	""")
