import streamlit as st
import importlib
from numpy.random import uniform


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

def code_interpreter(level = 1, args = [None, None, None]):
    def take_user_input(st=st):
        user_code = st.text_area('Python input here:'+' '*level)
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

    def run():
        try:
            import tmp  # import python file
            importlib.reload(tmp)
            tmp.fun(st=cols[1], x=uniform(0,1,100))   # run and display
        except SyntaxError:
            cols[1].write('Error: SyntaxError')
        except NameError:
            cols[1].write('Error: NameError')
        except KeyError:
            cols[1].write('Error: KeyError')
        except OverflowError:
            cols[1].write("""Error: OverflowError\n you seem to have made a number greater than $2^{127}$. For more info see *scientific computing week 1*.""") 

    ####
    cols = st. columns(2)

    # User input --> .py file
    user_code = take_user_input(cols[0])
    if firewall(user_code, st=cols[1]) == 'block': 
        return 0
    string = format_user_code(user_code)
    str2py(string)
    
    
    cols[1].caption('output:')
    c = cols[1].empty()
    if c.button('run'): run()
    

def level0():
    st.markdown(text_dict['Hello world!'])
    code_interpreter(level)

def level1():
    st.markdown(text_dict['Variables'])
    code_interpreter(level)
    
def level2():
    st.markdown(text_dict['Operators'])
    code_interpreter(level)

def level3():
    st.markdown(text_dict['For loops!'])
    code_interpreter(level)
    
def level4():
    st.markdown(text_dict['Functions'])
    
    code_interpreter(level)

def level5():
    st.markdown(text_dict['Bubble-sort'])
    
    code_interpreter(level,)


text_dict = getText_prep()

# main render script
c = st.columns(2)
# render header
c[0].markdown("""
    # Learn to code
    *tab doesn't work, use spaces*
    """)
st.markdown("---")

# render level
level_dict = {
    0 : level0,
    1 : level1,
    2 : level2,
    3 : level3,
    4 : level4,
    5 : level5,
}
level = c[1].select_slider('select level', level_dict.keys())
f = level_dict[level]; f()
