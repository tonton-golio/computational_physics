from utils_scicomp import *

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

	with st.expander(r'Going from abstract linear systems to matrices', expanded=False):
		st.markdown(text_dict['Header 2'])

	with st.expander(r'Can we solve it?', expanded=False):
		st.markdown(text_dict['Can we solve'])

	with st.expander(r'Sensitivity of a Linear System of Equations', expanded=False):
		st.markdown(text_dict['Sensitivity of a Linear System of Equations'])


	with st.expander(r'How to build the algorithms from scratch', expanded=False):
		st.markdown(text_dict['How to build the algorithms from scratch'])


	st.markdown(r'### LU-factorization')

	matrix = np.array([[1,4,12],
						[5,4,2],
						[9,5,67]])
	cols = st.columns(2)
	cols[0].markdown('If we start with a matrix like this one:')
	cols[0].table(matrix)
	cols[0].markdown('we may factorize into a pair of upper and lower triangular matricies')
	L, U = lu_factorize(matrix.astype(float))
	cols[0].table(np.round(U,2))
	cols[0].table(np.round(L,2))

	cols[1].code(text_dict['lu_factorize'])

	'Multiplying $L$ and $U$ should yield the input matrix:'
	matrix_out = (L@U).astype(float)
	#matrix_out

	
	st.markdown('Now that we have a factorization algorithm, we almost can solve a linear equation, we just need a forward and backward substitution')
	cols = st.columns(2)
	cols[0].code(text_dict['forward_substitute'])
	cols[1].code(text_dict['backward_substitute'])
	
	"As it usually goes with programming, things are simple when the pieces are in place"
	st.code(text_dict['solve_lin_eq'])
	"For QR-factorization, see least squares page"
	with st.expander(r'Errors', expanded=False):
		st.markdown(text_dict["Header 3"])
	
def linearLeastSquares():
	st.title('Linear Least Squares')

	# Main text
	text_dict = getText_prep(filename = text_path+'linearLeastSquares.md', split_level = 2)

	st.markdown(text_dict["Header 1"])

	run_least_squares = st.button('Run toy simulation',)
	if run_least_squares: toy_sim_leastSquares()

	with st.expander('Least Square Solution', expanded=False):
		st.markdown(text_dict["Least Square Solution"])

	with st.expander('Building a least squares algorithm from scratch', expanded=False):
		st.markdown(text_dict["Building a least squares algorithm from scratch"])

	
	st.code(text_dict["Code"])
	st.markdown(text_dict["Header 3"])
	st.code(text_dict['least_squares'])

def eigenSystems():
	st.title('EigenSystems')

	# Main text
	text_dict = getText_prep(filename = text_path+'eigenSystems.md', split_level = 2)

	st.markdown(text_dict["Header 1"])
	with st.expander('Deets', expanded=False):
		st.markdown(text_dict["Header 2"])

	with st.expander('The power method', expanded=False):
		st.markdown(text_dict['The power method'])

	with st.expander('Gershgorin', expanded=False):
		st.markdown(text_dict['Gershgorin centers'])
		st.code(text_dict['gershgorin'])

	with st.expander('Rayleigh iterate', expanded=True):
		st.markdown(text_dict['rayleigh iterate'])

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
	text_dict = getText_prep(filename = text_path+'partialDifferentialEquations.md')
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
	#"Partial Differential Equations":partialDifferentialEquations,
}

topic = st.sidebar.selectbox("topic" , list(func_dict.keys()))
func = func_dict[topic] ; func()