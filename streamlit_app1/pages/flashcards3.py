import streamlit as st
import pandas as pd
import gspread


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


sa = gspread.service_account_from_dict(credentials)
sh = sa.open('reviewQuestions')

wks = sh.worksheet('Sheet1')


#print("Rows: ", wks.row_count)
#print("Cols: ", wks.col_count)


records = wks.get_all_records()

df = pd.DataFrame.from_dict(records)
df
#records
