import streamlit as st
import pandas as pd
import gspread

sa = gspread.service_account()
sh = sa.open('reviewQuestions')

wks = sh.worksheet('Sheet1')


#print("Rows: ", wks.row_count)
#print("Cols: ", wks.col_count)


records = wks.get_all_records()

df = pd.DataFrame.from_dict(records)
df
#records


wks.update('A5', 'Anton')