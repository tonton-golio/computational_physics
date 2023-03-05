import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import os
import re
import pandas as pd


def get_tree(os_ = "windows", entensions = ['md','txt']):

    textdict = dict()
    counter = 0

    if os_ == "windows":
        dir_sep = "\\"
    elif os_ == "linux":
        dir_sep = "/"

    dir_sep = {
        "windows" : "\\",
        "linux" : "/"
    }[os_]
    for root, dirs, files in os.walk('.'):
        if '.'+dir_sep+'assets' in root or '.'+dir_sep+'pages' in root:
            for file in files:
                
                extension = file.split('.')[1]
                if extension in entensions:
                    with open(root + dir_sep + file,'rt', encoding='utf-8') as f:

                        textdict[counter] = {
                            'root' : root,
                            'filename' : root + dir_sep+file,
                            'content' : f.read(),
                            }
                        counter += 1

    return textdict
    
def search(searchterm, tree):
    assert type(searchterm) == str
    assert len(searchterm) > 2
 
    p = re.compile(searchterm, re.IGNORECASE)
    matches = dict()

    for i in tree:
        find_all = p.findall(tree[i]['content'])
        if find_all == []:  pass
        else:  matches[tree[i]['filename']] = len(find_all)

    
    matches = [(key,val) for key, val in matches.items()]
    df = pd.DataFrame(matches)
    df.sort_values(by=[1], ascending=False, inplace = True)#.head(5)
    
    topresults = df.head(5)[0].values
    return topresults


with st.sidebar:
    tree_md = get_tree(os_="linux") 

    searchterm = st.text_input(label = 'Search')
    #st.write("searchterm = ", searchterm)
    try:
        topresults = search(searchterm, tree_md)
        #st.write(topresults)
        # formatting topresults
        #subjects = [i.split('/')[-3] for i in topresults]
        #subjects

        topresults = {i.split('/')[-1].split('.')[0] : i for i in topresults}
        filepath = topresults[st.selectbox(label = 'Select pages', options = topresults.keys())]
        # get page



        tree_py = get_tree(os_="linux", entensions = ['py']) 
        searchterm = filepath.split('/')[-1]
        #st.write("searchterm = ", searchterm)
        topresults = search(searchterm, tree_py)
        page = topresults[0].split('/')[-1].split('.')[0]
        # navigate to file
        link = 'https://physics.streamlit.app/'
        st.markdown('[link to: {}]({})'.format(page, link+page))
            

    except:
        st.write('Please enter a searchterm (at least 3 characters long)')

