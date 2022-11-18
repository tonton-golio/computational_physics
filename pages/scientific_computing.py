from scicomp_utils import *

# Pages
def home():
	st.markdown(r"""
	# Scientific Computing

	Welcome to scientific computing!

	In this course, we;
	* learned about the errors arrising in computation.
	* wrote equation solvers for linear and non-linear systems of equations.
	* implemented methods for matrix diagonalization.
	* time evolved initial value problems.
	* looked at partial differential equations
	
	Use the sidebar to navigate!
	""")

def boundingErrors():
	st.title("Bounding Errors")

	# Links
	cols = st.columns(4)
	cols[0].markdown(r"[youtube: errors](https://www.youtube.com/watch?v=GFhhRdF54eI)")
	cols[1].markdown(r"[youtube: Floating point numbers](https://www.youtube.com/watch?v=PZRI1IfStY0)")
	cols[2].markdown(r"[youtube: fp ]( https://www.youtube.com/watch?v=f4ekifyijIg)")
	cols[3].markdown(r"[youtube: fp addition](https://www.youtube.com/watch?v=782QWNOD_Z0) ")

	# Main text
	text_dict = getText_prep(filename = text_path+'bounding_errors.md', split_level = 2)

	st.markdown(text_dict["Header 1"])
	st.markdown(text_dict["Header 2"])
	st.markdown(text_dict["Header 3"])
	with st.expander('Go deeper', expanded=False):
		st.markdown(text_dict["Example"])

def linearEquations():
	st.title('Linear Equations')

	# Main text
	text_dict = getText_prep(filename = text_path+'linear_equations.md', split_level = 2)

	st.markdown(text_dict["Header 1"])
	st.markdown(text_dict["Header 2"])
	
def linearLeastSquares():
	st.title('Linear Least Squares')

	# Main text
	text_dict = getText_prep(filename = text_path+'linearLeastSquares.md', split_level = 2)

	st.markdown(text_dict["Header 1"])

	run_least_squares = st.button('run',)
	if run_least_squares: toy_sim_leastSquares()


	st.markdown(text_dict["Header 2"])
	st.code(text_dict["Code"])
	st.markdown(text_dict["Header 3"])

def eigenSystems():
	st.title('EigenSystems')

	# Main text
	text_dict = getText_prep(filename = text_path+'eigenSystems.md', split_level = 2)

	st.markdown(text_dict["Header 1"])

def nonlinearEquationsOptimization():
	st.title('Non-linear Equations Optimization')

	# Main text
	text_dict = getText_prep(filename = text_path+'nonlinearEquationsOptimization.md', split_level = 1)
	st.markdown(text_dict["Header 1"])
	st.markdown(text_dict['Header 2'])
	st.markdown(text_dict['Header 3'])

def initialValueProblems():
	st.title('Initial-value Problems')

	# Main text
	text_dict = getText_prep(filename = text_path+'initialValueProblems.md', split_level = 2)

	st.markdown(text_dict["Header 1"])

	run_reactionDiffusion()

def partialDifferentialEquations():
	st.title('Partial, differential Equations (PDEs)')
	# Main text
	text_dict = getText_prep(filename = text_path+'initialValueProblems.md')
	st.markdown(text_dict["Header 1"])

# Navigator
func_dict = {
	'Home' : home,
	'Bounding Errors' : boundingErrors,
	'Linear Equations': linearEquations,
	'Linear Least Squares' : linearLeastSquares,
	"Eigensystems" : eigenSystems,
	"Nonlinear Equations Optimization" : nonlinearEquationsOptimization,
	"Initial Value Problems" : initialValueProblems,
	"Partial Differential Equations":partialDifferentialEquations,
}

topic = st.sidebar.selectbox("topic" , list(func_dict.keys()))
func = func_dict[topic] ; func()