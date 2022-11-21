from utils_inverse import *
st.title('Inverse Problems')


def week1():
    text_dict = getText_prep(filename = text_path+'week1.md', split_level = 1)

    #st.header('Week 1 notes')
    st.markdown(text_dict['Header 1'])

    with st.expander('Examples', expanded=False):
        st.markdown(text_dict['Examples'])


    st.markdown(text_dict['Header 2'])


# Navigator
topic_dict = {
    'week 1': week1,
  }

topic = st.sidebar.selectbox("topic" , list(topic_dict.keys()))

run_topic = topic_dict[topic] ; run_topic()



