import streamlit as st
import pandas as pd
st.set_page_config(page_title="Computational Physics", 
	page_icon=":taxi:", 
	layout="wide", 
	initial_sidebar_state="collapsed", 
	menu_items=None)
st.title("Computational Physics")
try:
	st.image('streamlit_app1/assets/images/cover.jpg')
except:
	st.image('assets/images/cover.jpg')

st.markdown("""
[github](https://www.github.com/tonton-golio/computational_physics)
""")

st.markdown(r"""
## Contributters
#### [Anton](https://www.github.com/tonton-golio/)
#### [Riz](https://www.github.com/rizfn/)
#### [Yoshiaki](https://www.github.com/yoshiysoh/)
	""")
