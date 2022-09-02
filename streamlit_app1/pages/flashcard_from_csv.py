import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

@st.cache(allow_output_mutation=True)
def get_data():
    return []


df = pd.read_csv('streamlit_app1/assets/scientific_computing/review_questions_ch1.csv',
	index_col=0)
question_numbers = list(df.index)


try:
    df_answers = pd.read_csv('streamlit_app1/assets/scientific_computing/review_questions_ch1_answers.csv',
        index_col=0)
except:
    df_answers = pd.DataFrame()


st.title('Flashcards')


if 'count' not in st.session_state:
    st.session_state.count = int(question_numbers[np.random.randint(0,len(question_numbers))])


rand_q_num = st.session_state.count
st.write(rand_q_num)
st.subheader(df['text'][rand_q_num])



user_id = st.text_input("User ID")
answer = st.text_input('Answer')
certainty = st.slider("certainty", 0, 100)

if st.button("Submit answer"):
    get_data().append({"Question number":rand_q_num,
    	"UserID": user_id, 
    	"answer": answer, 
    	"certainty": certainty})
    df_answers = pd.concat([df_answers,pd.DataFrame(get_data())])
    df_answers.to_csv('../assets/scientific_computing/review_questions_ch1_answers.csv')
    df_answers.reset_index(inplace=True)
    df_answers.drop(['index'], axis=1,inplace=True)
    st.write(df_answers[df_answers["Question number"]==rand_q_num])


# Button to obtain new question by refreshing:
if st.button('New Question'):
   #print("") #Auto refresh page
   st.session_state.count = int(question_numbers[np.random.randint(0,len(question_numbers))])


if st.button('Manual question number input'):
   #print("") #Auto refresh page
   num = st.number_input('pick',1,len(df))
   if st.button('go'):
       st.session_state.count = num


# save scores some how?