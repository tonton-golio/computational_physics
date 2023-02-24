import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import os
import re
import pandas as pd


textdict = dict()
counter = 0

for root, dirs, files in os.walk('.'):
    if '.\\assets' in root or '.\\pages' in root:
        for file in files:
            extension = file.split('.')[1]
            if extension in ['md','txt']:
                with open(root + '\\' + file,'rt', encoding='utf-8') as f:
                    textdict[counter] = {
                        'root' : root,
                        'filename' : root + '\\'+file,
                        'content' : f.read(),
                        }
                    counter += 1
                    
#st.write(textdict)
searchterm = st.text_input(label = 'Search')
st.write(searchterm)
if searchterm == '':
    pass
elif len(searchterm) < 3:
    pass
else: 
    p = re.compile(searchterm, re.IGNORECASE)
    matches = dict()

    for i in textdict:
        test = p.findall(textdict[i]['content'])
        if test == []:
            pass
        else:
            matches[textdict[i]['filename']] = len(test)
    matches = [(key,val) for key, val in matches.items()]
    df = pd.DataFrame(matches)
    df.sort_values(by=[1], ascending=False, inplace = True)#.head(5)
    
    topresults = df.head(5)[0]
    filepath = topresults[0]
    st.write(topresults)
"""
for root, dirs, files in os.walk('.\\pages'):
    for file in files:
        file
        extension = file.split('.')[1]
        if extension in ['py']:
            with open(root + '\\' + file,'rt', encoding='utf-8') as f:
                st.write(f.read())
                if filepath[2:] in f.read():
                    filepath, file
"""