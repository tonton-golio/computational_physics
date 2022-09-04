# streamlit_app.py

import streamlit as st
from gsheetsdb import connect

# Create a connection object.
conn = connect()

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

@st.cache(allow_output_mutation=True)
def get_data():
    return []

course = 'scientific computing'
chapter = 'Bounding Errors'

try: # Github path
    questions = pd.read_csv('streamlit_app1/assets/scientific_computing/review_questions_ch1.csv',
        index_col=0)
except: # local path
    questions = pd.read_csv('assets/scientific_computing/review_questions_ch1.csv',
        index_col=0)

question_numbers = list(questions.index)


st.title('Flashcards')


if 'count' not in st.session_state:
    st.session_state.count = int(question_numbers[np.random.randint(0,len(question_numbers))])

random_question_number = st.session_state.count

st.subheader(questions['text'][random_question_number])


user_id = st.text_input("User ID")
answer = st.text_input('Answer')
certainty = st.slider("certainty", 0, 5)

if st.button("Submit answer"):
    get_data().append({
        "Course" : course,
        "Chapter" : chapter,
        "QuestionNumber":st.session_state.count,
        "UserID": user_id, 
        "Answer": answer, 
        "Certainty": certainty,
        'DateTime' : datetime.now(), # Getting the current date and time
        })
    #df_answers = pd.concat([df_answers,pd.DataFrame(get_data())])
    #df_answers.to_csv('streamlit_app1/assets/scientific_computing/review_questions_ch1_answers.csv')
    #df_answers.reset_index(inplace=True)
    #df_answers.drop(['index'], axis=1,inplace=True)
    #st.write(df_answers[df_answers["Question number"]==rand_q_num])
df0 = pd.DataFrame(get_data())
df0

# Button to obtain new question by refreshing:
if st.button('New Question'):
   st.session_state.count = int(question_numbers[np.random.randint(0,len(question_numbers))])


# Perform SQL query on the Google Sheet.
# Uses st.cache to only rerun when the query changes or after 10 secs.
@st.cache(ttl=10)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows

sheet_url = st.secrets["public_gsheets_url"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')

# Print results.
#for row in rows:
#    st.write(f"{row.name} has a :{row.pet}:")




df = pd.DataFrame(rows)
df 