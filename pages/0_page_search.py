import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import os
import re
import pandas as pd
from graphviz import Digraph


def get_tree(os_ = "windows", entensions = ['md','txt']):

    textdict = dict()
    counter = 0

    dir_sep = { # directory separator
        "windows" : "\\",
        "linux" : "/", # also mac
        "darwin": "/",
    }[os_]


    for root, dirs, files in os.walk('.'):
        if '.'+dir_sep+'assets' in root or '.'+dir_sep+'pages' in root:
            for file in files:
                try:
                    extension = file.split('.')[1]
                    if extension in entensions:
                        with open(root + dir_sep + file,'rt', encoding='utf-8') as f:

                            textdict[counter] = {
                                'root' : root,
                                'filename' : root + dir_sep+file,
                                'content' : f.read(),
                                }
                            counter += 1
                except:
                    pass

    return textdict
    
def draw_tree(tree, level1_names_input = None, level2_names_input = None, st=st):
    dic = {}
    for i in tree:
        level0 = tree[i]['root'].split('/')[1]
        level1 = tree[i]['root'].split('/')[2]
        name = tree[i]['filename'].split('/')[-1]
        if level0 not in dic:
            dic[level0] = {}
        if level1 not in dic[level0]:
            dic[level0][level1] = []
        dic[level0][level1].append(name)
    #st.write(dic)



    # polar tree
    g = Digraph()
    # set background color
    g.attr(bgcolor='transparent')
    # reduce margins
    g.attr(margin='0') # this didn't work there is still room above and below the graph
    # set graph style
    g.attr('graph', rankdir='LR', size='16,6', ratio='fill', splines='ortho', nodesep='0.5', ranksep='0.5')
    # set node style
    g.attr('node', shape='circle', style='filled', color='lightblue', fontsize='6', fixedsize='true', width='1.5')
    for i, (level0_name, level1_names) in enumerate(dic.items()):
        # adding the root node
        g.node(level0_name, 'home', pos='{},{}!'.format(0, 0))
        #i, (level0_name, level1_names)

        # adding the children nodes
        for j, (level1_name, level2_names) in enumerate(level1_names.items()):
            if level1_names_input is not None:
                if level1_name in level1_names_input: 
                    g.node(level1_name, level1_name,  pos='{},{}!'.format(30*j, -1))
                    g.edge(level0_name, level1_name, style='dashed', color='red')
                    if level2_names_input is not None:
                        for name in level2_names:
                            if name in level2_names_input:
                                g.node(level1_name+name, name[:-3], pos='{},{}!'.format(30*j, -2))
                                g.edge(level1_name, level1_name+name, style='box', color='red')


    #g.view()
    st.graphviz_chart(g, use_container_width=True)


        

def search(searchterm, tree, return_all = False):
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
    df.sort_values(by=[1], ascending=False, inplace = True)
    
    if return_all:
        results = df[0].values
        return results
    else:
        topresults = df.head(5)[0].values
        return topresults
        

cols = st.columns((1,4))
tree_md = get_tree(os_="linux") 

searchterm = cols[0].text_input(label = 'Search')



#st.write("searchterm = ", searchterm)
try:

    topresults = search(searchterm, tree_md, return_all = True)
    level1_names_input = [i.split('/')[2] for i in topresults]
    level2_names_input = [i.split('/')[-1] for i in topresults]
    
    draw_tree(tree_md, level1_names_input, level2_names_input, st=cols[1])
    #level1_names_input

    #st.write(topresults)
    # formatting topresults
    #subjects = [i.split('/')[-3] for i in topresults]
    #subjects

    #topresults = {i.split('/')[-1].split('.')[0] : i for i in topresults}
    #filepath = topresults[cols[0].selectbox(label = 'Select page', options = topresults.keys())]
    # get page


    #tree_py = get_tree(os_="linux", entensions = ['py']) 
    #searchterm = filepath.split('/')[-1]
    #st.write("searchterm = ", searchterm)
    #topresults = search(searchterm, tree_py)
    #page = topresults[0].split('/')[-1].split('.')[0]
    # navigate to file
    #link = 'https://physics.streamlit.app/'
    #cols[0].markdown('[link to: {}]({})'.format(page, link+page))
        

except:
    cols[0].write('Please enter a searchterm (at least 3 characters long)')

