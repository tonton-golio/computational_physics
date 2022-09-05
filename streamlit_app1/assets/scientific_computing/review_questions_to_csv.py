import re
import pandas as pd

review_questions_filename = 'review_questions_ch1.txt' 
with open(review_questions_filename) as f:
	text = f.read().replace('\n','').replace('- ','')


regex = r"[0-2][0-9]?[.][0-9][0-9]? "
matches = re.finditer(regex, text, re.MULTILINE)

questions = {}
for matchNum, match in enumerate(matches, start=1):
    questions[matchNum] = {"start" : match.start(), "end" : match.end()}
    
for matchNum in questions.keys():
	my_end = questions[matchNum]['end']
	try:
		next_beginning = questions[matchNum+1]['start'] 
	except KeyError:
		next_beginning = 99999999
	questions[matchNum]['text'] = text[my_end:next_beginning].replace('(a)','\n(a)').replace('(b)','\n(b)').replace('(c)','\n(c)').replace('(d)','\n(d)').replace('(e)','\n(e)').replace('(f)','\n(f)')

#print(questions[32])
df = pd.DataFrame.from_dict(questions).T
df.to_csv(review_questions_filename.replace('.txt','.csv'))




# So we still wanna figure out what type of question we have:
#	True/False
def true_false(question):
	if 'True or false' in question:
		return True
	else:
		return False

#	Number answer
#	Text field answer


# if we have a multipart question, how should we handle this?
# note: they are ussually formatted as (a) asdasdasdas, (b) asdsad
## for each letter, display on a new line:
def is_multipart_question(question = 'A question about (a) cats, (b) dogs, etc.'):
	if '(a)' in question:
		return True
	else:
		return False

def number_multiparts(question):
	if is_multipart_question(question):
		count = 1
		for i in 'a b c d e f g h i j k l m n o p'.split():
			if f'({i})' in question:
				count += 1
			else:
				break
	else:
		return 0






