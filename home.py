import requests, json
import streamlit as st
import pandas as pd
st.set_page_config(page_title='Computational Physics',#"plz contribute", 
	page_icon=":microscope:", 
	layout="centered", 
	initial_sidebar_state="expanded", 
	menu_items=None)
st.markdown(r"""
	# Computational Physics
	Notes and simluations for computational physics.
			
	If you notice errors; feel welcome to submit a pull request or mention it in the discord server.
	""")

cols = st.columns(2)
with cols[0]:
	# contribute on github
	# st.markdown(r"""
	# 	[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tonton-golio/computational_physics)
	# 	""")
	
	# make the above element centered
	st.write("""
	<div style="text-align: center">
		<a href="https://github.com/tonton-golio/computational_physics">
			<img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />
		</a>
	</div>
	""", unsafe_allow_html=True)
with cols[1]:
	# join the discord!
	st.markdown(r"""
		[![Discord](https://img.shields.io/discord/865661082025134336?color=7289da&logo=discord&logoColor=white)](https://discord.gg/JzZRhUNV5c)
		""")


st.image('assets/images/mandel_front.png', use_column_width=True)


# contributors
def get_contributors():
	contributors = {}
	logins = json.loads(requests.get("https://api.github.com/repos/tonton-golio/computational_physics/contributors?per_page=100").text)
	
	for login in logins:
		try: name = login['login']
		except: continue

		try: avatar = login['avatar_url']
		except: avatar = None


		try: contributions = login['contributions']
		except: contributions = 1

		try: url = login['html_url']
		except: url = None

		contributors[name] = {'avatar': avatar, 'contributions': contributions, 'url': url}


	contributors_sorted = {k: v for k, v in sorted(contributors.items(), key=lambda item: item[1]['contributions'], reverse=True)}
	# contributors_sorted
	return contributors_sorted, logins




def make_contributors_table(contributors):
	# now we want a nice table of contributors
	cols = st.columns(min([4, len(contributors)]))
	for i, col in enumerate(cols):
		for j in range(0, len(contributors), 4):
			try:
				contributor = list(contributors.keys())[i+j]
				col.image(contributors[contributor]['avatar'], width=100)
				col.markdown(f"**[{contributor}]({contributors[contributor]['url']})**")
				col.markdown(f"Commits: {contributors[contributor]['contributions']}")
			except IndexError:
				pass
 
 
st.markdown(r"""### Contributors (so far)""")
contributors, logins = get_contributors()
if len(contributors) > 0:
	make_contributors_table(contributors)
else:
	st.write("Contributors not found.")
	st.write(logins)
