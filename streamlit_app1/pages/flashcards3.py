# flashcards3.py

import numpy as np
import matplotlib.pyplot as plt
import time
import streamlit as st
import pandas as pd
import gspread
import re

credentials = {
    "type": "service_account",
    "project_id": "compphysics",
    "private_key_id": st.secrets['private_key_id'],
    "private_key": st.secrets['private_key'],
    "client_email": st.secrets['client_email'],
  	"client_id": st.secrets['client_id'],
  	"auth_uri": "https://accounts.google.com/o/oauth2/auth",
  	"token_uri": "https://oauth2.googleapis.com/token",
  	"auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  	"client_x509_cert_url": st.secrets['client_x509_cert_url']
}

def format_math_text(question):

    regex = r"\$.{1,100}\$"

    matches = re.finditer(regex, question, re.MULTILINE)

    parts = []
    prev_end, match_found = 0, False
    for matchNum, match in enumerate(matches, start=1):
        parts.append(question[prev_end:match.start()])
        parts.append(question[match.start():match.end()])
        prev_end = match.end()
        match_found = True
    if match_found:
        parts.append(question[match.end():])
    else:
        parts.append(question)

    for i in parts:
        if r"$" in i:
            st.latex(i.replace('$',''))
        else:
            st.write(i)


sa = gspread.service_account_from_dict(credentials)
sh = sa.open('reviewQuestions')
wks = sh.worksheet('Sheet1')
records = wks.get_all_records()
answers = pd.DataFrame.from_dict(records)


@st.cache(allow_output_mutation=True)
user_id = st.sidebar.text_input("User ID")

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


question = questions['text'][random_question_number]
format_math_text(question)




answer = st.text_input('Answer')
certainty = st.slider("certainty", 0, 5)

if st.button("Submit answer"):
    new_answers = {
        "Course" : course,
        "Chapter" : chapter,
        "QuestionNumber":st.session_state.count,
        "UserID": user_id, 
        "Certainty": certainty,
        "Answer": answer, 
        'DateTime' : time.time(), # Getting the current date and time
        }
    new_answers = list(new_answers.values())
    loc = f"A{len(answers)+2}:G{len(answers)+2}"
    #st.write(loc, new_answers)
    wks.update(loc,[new_answers])

    records = wks.get_all_records()
    answers = pd.DataFrame.from_dict(records)
    answers[answers["Course"]==course][answers["Chapter"]==chapter][answers["QuestionNumber"]==random_question_number]




# Button to obtain new question by refreshing:
if st.button('New Question'):
   st.session_state.count = int(question_numbers[np.random.randint(0,len(question_numbers))])

