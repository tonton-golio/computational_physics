import streamlit as st
import importlib

import matplotlib. pyplot as plt
from numpy import mean, vstack, array, argmin
from numpy.random import uniform, normal
from seaborn import load_dataset


# Further functionality
# * check answers
# * save user answers


def getText_prep(filename = 'assets/learn_to_code/initial.md', split_level = 1):
	"""
		get text from markdown and puts in dict
	"""
	# extra function
	with open(filename,'r', encoding='utf8') as f:
		file = f.read()
	level_topics = file.split('\n'+"#"*split_level+' ')
	text_dict = {i.split("\n")[0].replace('### ','') : 
                "\n".join(i.split("\n")[1:]) for i in level_topics}
    
	return text_dict  

def code_interpreter(level, cols):
    def take_user_input(st=st):
        user_code = st.text_area('Python input here:'+' '*level)
        st.caption('Ctrl+Enter to render')
        st.code(user_code)
        return user_code

    def firewall(user_code, st=st):
        banned = ['import', 'install']
        for b in banned:
            if b in user_code: 
                st.write('cant deal with imports')
                return 'block'
        return 'pass'

    def format_user_code(user_code):
        if len(user_code) == 0: user_code = 'pass'
        user_code = user_code.replace('print(', 'st.write(')
        lines = user_code.split('\n')
        string = 'import streamlit as st \n' + \
            f"""def fun(st=st, x=None,y=None,z=None):\n""" + \
            ''.join(['\t' + line + "\n" for line in lines])

        return string

    def str2py(string): # save to python file
        with open("tmp"+'.py', 'w') as the_file:
                the_file.write(string)
    # User input --> .py file
    user_code = take_user_input(cols[0])
    if firewall(user_code, st=cols[1]) == 'block': 
        return 0
    string = format_user_code(user_code)
    str2py(string)

def run():
    try:
        import tmp  # import python file
        importlib.reload(tmp)
    except SyntaxError:
        return 'Error: SyntaxError'
    except NameError:
        return 'Error: NameError'
    except KeyError:
        return 'Error: KeyError'
    except OverflowError:
        return """Error: OverflowError\n you seem to have made a number greater than $2^{127}$. For more info see *scientific computing week 1*."""
    return tmp.fun
    
    
# Levels
def helloWorld():
    key = 'Hello world!'
    level = 0

    st.markdown(f'### lvl {level} - {key}')
    st.markdown(f'{text_dict[key]}')

    cols = st.columns(2)
    
    code_interpreter(level, cols)

    cols[1].caption('output:')
    if cols[1].button('run'): run()(st=cols[1])

def variables():
    key = 'Variables'
    level = 1

    st.markdown(f'### lvl {level} - {key}')
    st.markdown(f'{text_dict[key]}')

    cols = st.columns(2)
    
    code_interpreter(level, cols)

    cols[1].caption('output:')
    if cols[1].button('run'): run()(st=cols[1])
    
def operators():
    key = 'Operators'
    level = 2

    st.markdown(f'### lvl {level} - {key}')
    st.markdown(f'{text_dict[key]}')

    cols = st.columns(2)
    
    code_interpreter(level, cols)

    cols[1].caption('output:')
    if cols[1].button('run'): run()(st=cols[1])

def forLoops():
    key = 'For-loops'
    level = 3

    st.markdown(f'### lvl {level} - {key}')
    st.markdown(f'{text_dict[key]}')

    cols = st.columns(2)
    
    code_interpreter(level, cols)

    cols[1].caption('output:')
    if cols[1].button('run'): run()(st=cols[1])
    
def functions():
    key = 'Functions'
    level = 4

    st.markdown(f'### lvl {level} - {key}')
    st.markdown(f'{text_dict[key]}')

    cols = st.columns(2)
    
    code_interpreter(level, cols)

    cols[1].caption('output:')
    if cols[1].button('run'): run()(st=cols[1])

def bubbleSort():
    key = 'Bubble-sort'
    level = 5

    st.markdown(f'### lvl {level} - {key}')
    st.markdown(f'{text_dict[key]}')

    cols = st.columns(2)
    
    code_interpreter(level, cols)

    cols[1].caption('output:')
    if cols[1].button('run'): run()(st=cols[1])
     
def kMeans():
    key = 'kMeans'
    level = 6

    st.markdown(f'### lvl {level} - {key}')
    st.markdown(f'{text_dict[key]}')

    cols = st.columns(2)
    
    code_interpreter(level, cols)

    cols[1].caption('output:')

    
    ## init
    x = uniform(0,1,(100,2))
    x = load_dataset('iris').values[:,:2]
    cs = vstack([normal(loc=mean(x, axis=0)[0], scale = .6, size=3),
                    normal(loc=mean(x, axis=0)[1], scale = .1, size=3)]).T
    
    if cols[1].button('run'): run()(st=cols[1], x=x)
    
    if st.button('Show solution'):
        st.markdown(text_dict['kMeans solution'])

        # solution
        fig, ax = plt.subplots(1,5, figsize=(16,4))
        for i in range(5):
            d2c = array([(x-c)**2 for c in cs]).sum(axis=2).T  # ditance to centroids
            aff = argmin(d2c, axis=1)
            
            # update cs
            cs = array([mean(x[aff == a], axis=0) for a in set(aff)])

            for a in set(aff):
                xa = x[aff==a]
                ax[i].scatter(xa[:,0], xa[:,1])
            ax[i].set(yticks=[], xticks=[])
        st.pyplot(fig)

        shapes = {
            'x' : x.shape,
            'd2c' : d2c.shape,
            'aff' : aff.shape,
            'cs' : cs.shape,
        } ; # st.write(shapes)
    
    
# main render script
text_dict = getText_prep()
cols = st.columns(2)
# render header
cols[0].markdown("""
    # Learn to code
    * *tab doesn't work, use spaces*
    * *press c to clear cache if its buggy*
    """)
st.markdown("---")


# render level
level_dict = {
    0 : helloWorld,
    1 : variables,
    2 : operators,
    3 : forLoops,
    4 : functions,
    5 : bubbleSort,
    6 : kMeans,
}

level = cols[1].select_slider('select level', level_dict.keys())

f = level_dict[level]; f()