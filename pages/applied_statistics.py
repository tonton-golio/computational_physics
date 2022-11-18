import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 



st.title('Applied statistics')

st.header('Week 1')
with st.expander('Week 1 description', expanded=False):
     st.markdown("""
     (Introduction, General Concepts, ChiSquare Method):
     Nov 21: 8:15-10:00: Introduction to course and overview of curriculum.
          Mean and Standard Deviation. Correlations. Significant digits. Central limit theorem. (12-13 Measuring in Aud. A!)
     Nov 22: Error propagation (which is a science!). Estimate g measurement uncertainties.
     Nov 25: ChiSquare method, evaluation, and test. Formation of Project groups.
     """)


st.markdown(r"""
     ### mean and standard deviation
     mean (*average*) is the geometric midpoint of a data-set. 

     ``np.mean(x)``

     $$
          \left< x \right> = \frac{1}{N}\sum_i^N x_i
     $$

     Standard deviation is a measure of how much datapoint deviates from the dataset mean.

     ``np.std(x)``


     $$
          \chi = \frac{1}{N}\sum_i^N ( \left< x \right> -x)^2
     $$

     ### Correlations
     speaks to whether a feature varies in concordance with another.

     iris = sns.load_dataset('iris')
     corr = pd.corr(iris)

     $$
          math
     $$

     ### Central limit theorem
     Something something: the reason a small system shows a soft transition has something to do with the central limit theorem.


     ### Error propagation
     hmm, how do we propagate these errors?



     ### Estimating uncertainties





     ###  ChiSquare method, evaluation, and test



     """)

st.header('Week 2')
with st.expander('Week 2 description', expanded=False):
     st.markdown("""
     Week 2 (PDFs, Likelihood, Systematic Errors):
     Nov 28: Probability Density Functions (PDF) especially Binomial, Poisson and Gaussian.
     Nov 29: Principle of maximum likelihood and fitting (which is an art!).
     Dec 2: 8:15 - Group A: Project (for Wednesday the 14th of December) doing experiments in First Lab.
                 9:15 - Group B: Systematic Uncertainties and analysis of "Table Measurement data" Discussion of real data analysis (usual rooms).
     """)

st.markdown(r"""
     
     ### Probability density functions (PDFs)
     *a function of a continuous random variable, whose integral across an interval gives the probability that the value of the variable lies within the same interval.*

     ### Binomial distribution
     $$
          math
     $$
     ### Poisson distribution
     $$
          math
     $$
     ### Gaussian distribution
     $$
          math
     $$
     """)

def PDFs(size = 1000):
     x_uniform = np.random.rand(size)
     x_normal = np.random.randn(size)
     x_binomial = np.random.binomial(10, .3, size)
     x_poisson = np.random.randn(size)
     x_gaussian = np.random.randn(size)

     fig, ax = plt.subplots(1,5, figsize=(12,3))

     counts, bins = np.histogram(x_uniform)

     ax[0].hist(x_uniform)
     ax[0].set_title('Uniform', color='white')
     ax[1].hist(x_normal)
     ax[1].set_title('Normal', color='white')
     ax[2].hist(x_poisson)
     ax[2].set_title('Poisson', color='white')
     ax[3].hist(x_binomial)
     ax[3].set_title('Binomial', color='white')
     ax[4].hist(x_gaussian)
     ax[4].set_title('Gaussian', color='white')
     plt.close()
     st.pyplot(fig)

PDFs(1000)


st.markdown(r"""
     ### Principle of maximum likelihood and fitting
     $$
          likelihood = math
     $$

     we want to maximize this term ^^

     """)

def fitSimple(size =  100, nsteps = 100):

     x = np.random.rand(size)
     noise = lambda: np.random.randn()
     f = lambda x, a, b=0: a*x**2 + b*noise()
     y = f(x, 4, 0)



     # fitting


     loss = lambda a: sum((y - f(x,a))**2)


     delta = 0.1
     a0 = 1

     current_loss = loss(a0)
     for i in range(nsteps):
          pos_loss = loss(a0+delta)
          neg_loss = loss(a0-delta)
          if pos_loss < current_loss:
               current_loss = pos_loss
               a0 += delta
          elif neg_loss < current_loss:
               current_loss = neg_loss
               a0 -= delta
          else:
               delta *= 0.9

     fig, ax = plt.subplots(figsize=(8,3))
     ax.scatter(x,y, label='data')
     x_plot = np.linspace(min(x), max(x), max([100,size]))
     ax.plot(x_plot, f(x_plot, a0), ls='--', c='r', label='fit')
     plt.legend(facecolor='beige')
     plt.close()
     st.pyplot(fig)


fitSimple(size =  100, nsteps = 100)


st.header('Week 3')
with st.expander('Week 3 description', expanded=False):
     st.markdown("""
     Week 3 (Using Simulation and More Fitting):
     Dec 5: 8:15 - Group B: Project (for Wednesday the 14th of December) doing experiments in First Lab.
                 9:15 - Group A: Systematic Uncertainties and analysis of "Table Measurement data". Discussion of real data analysis (usual rooms).
     Dec 6: Producing random numbers and their use in simulations.
     Dec 9: Summary of curriculum so far. Fitting tips and strategies.
     """)

st.markdown(r"""
     ### Systematic Uncertainties and analysis of "Table Measurement data"


     ### Producing random numbers and their use in simulations.

     We may produce random number using ``np.random.rand()`` (these are uniform). These can be used in simulations, as:
     * we may add random error (noise) to our data
     * We may let outcomes be decided probablitically (if a $>$ random value: do action.)
     """)


st.header('Week 4')
with st.expander('Week 4 description', expanded=False):
     st.markdown("""
     Week 4 (Hypothesis Testing and limits):
     Dec 12: Hypothesis testing. Simple, Chi-Square, Kolmogorov, and runs tests.
     Dec 13: More hypothesis testing, limits, and confidence intervals. Testing your random (?) numbers.
          Project should been submitted by Wednesday the 14th of December at 22:00!
     Dec 16: Table Measurement solution discussion. Simpson's paradox.
     """)

st.markdown(r"""
     ### Hypothesis testing. 

     #### Simple
     #### Chi-Square
     #### Kolmogorov
     #### runs tests.

     ### limits
     ### confidence intervals

     ### Testing your random (?) numbers.
     ### Simpson's paradox
     """)

def random(size, dist = 'normal', mu=0, sigma=1):
     return [i for i in range(size)]





st.header('Week 5')
with st.expander('Week 5 description', expanded=False):
     st.markdown("""
     Week 5 (Bayesian statistics and Multivariate Analysis):
     Dec 19: Bayes theorem and Baysian statistics (Mathias).
     Dec 20: Multi-Variate Analysis (MVA). Fits in 2D. The linear Fisher discriminant.
     """)

st.markdown(r"""

     """)



st.header('Week 6')
with st.expander('Week 6 description', expanded=False):
     st.markdown("""
     Week 6 (Real data classification/analysis and introduction to Machine Learning):
     Jan 2: Machine Learning (ML). Neural Networks, Decision Trees and other MLs.
          Problem set should be submitted by Tuesday the 3rd of January at 22:00!
     Jan 3: Analysis of real and complex data on separating/classifying events. Analysis of testbeam data (part I).
     Jan 6: Time series analysis (Mathias). Analysis of testbeam data (part II).
     """)

st.markdown(r"""
     Machine learning algorithms can be split into two types, supervised (known labels) and unsupervised (unknown labels). We may solve three tasks: clustering, classification & regression.
     ### Clustering
     """)

def makeBlobs(size=100):
     '''
          Makes 2d random blobs
     '''
     X = np.random.rand(size, 2)

     X[0][X[0]<0.5] -= .5
     X[1][X[1]<0.5] -= .5

     noise = np.random.randn(size, 2)*.1
     return X+noise


X = makeBlobs(100)
fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1])
plt.close()
st.pyplot(fig)




st.header('Week 7')
with st.expander('Week 7 description', expanded=False):
     st.markdown("""
     Week 7 (Advanced fitting, Calibration, and Problem Set deliberation):
     Jan 9: Advanced fitting with both functions and models.
     Jan 10: Calibration and use of control channels.
     Jan 13: Discussion of Problem Set solution.
     Week 8 (Fitting and exam):
     Jan 16: Discussion of selected parts of course curriculum. Exercise on fitting.
     Jan 17: Deliberation on previous (2016) exam. Exam questions. Catch up on exercises.
     Jan 19: Exam given (posted on course webpage 8:00 in the morning).
     Jan 20: 20:00 Exam to be handed in (on www.eksamen.ku.dk).
     """)

st.markdown(r"""

     """)





