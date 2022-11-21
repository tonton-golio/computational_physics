from utils_appstat import *
st.title('Applied statistics')


def week1():
    text_dict = getText_prep(filename = text_path+'week1.md', split_level = 1)
     
    #st.header('Week 1')
    with st.expander('Week 1 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

    # compare N-1 and N...
    if st.button('run std_calculations'):
        st.pyplot(std_calculations())

    st.markdown(text_dict['Header 2'])

    st.markdown(text_dict['Header 3'])

    # roll a die
    if st.button('run roll_a_die'):
        st.pyplot(roll_a_die())


    # roll dice
    if st.button('run roll_dice'):
        st.pyplot(roll_dice())
        
    st.markdown(text_dict['Header 4'])
    
def week2():
    text_dict = getText_prep(filename = text_path+'week2.md', split_level = 1)
     
    #st.header('Week 2')
    with st.expander('Week 2 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

    st.pyplot(PDFs(1000))

    st.markdown(text_dict['Header 2'])

    st.pyplot(fitSimple(size =  100, nsteps = 100))

def week3():
    text_dict = getText_prep(filename = text_path+'week3.md', split_level = 1)
     
    #st.header('Week 3')
    with st.expander('Week 3 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

def week4():
    #st.header('Week 4')
    text_dict = getText_prep(filename = text_path+'week4.md', split_level = 1)
     
    with st.expander('Week 4 description', expanded=False):
        st.markdown(text_dict['description'])
    st.markdown(text_dict['Header 1'])

    def random(size, dist = 'normal', mu=0, sigma=1):
        return [i for i in range(size)]

def week5():
    #st.header('Week 5')
     
    text_dict = getText_prep(filename = text_path+'week5.md', split_level = 1)
    with st.expander('Week 5 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

def week6():
    #st.header('Week 6')

    text_dict = getText_prep(filename = text_path+'week6.md', split_level = 1)
    with st.expander('Week 6 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])

    X = makeBlobs(100)
    st.pyplot(scatter(X[:,0], X[:,1]))
     

def week7():
    #st.header('Week 7')
    
    text_dict = getText_prep(filename = text_path+'week7.md', split_level = 1)
    
    with st.expander('Week 7 description', expanded=False):
        st.markdown(text_dict['description'])

    st.markdown(text_dict['Header 1'])


# Navigator
topic_dict = {
    'week 1': week1,
    'week 2': week2,
    'week 3': week3,
    'week 4': week4,
    'week 5': week5,
    'week 6': week6,
    'week 7': week7,    }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()


