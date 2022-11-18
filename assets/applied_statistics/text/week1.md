

# description
(Introduction, General Concepts, ChiSquare Method):
Nov 21: 8:15-10:00: Introduction to course and overview of curriculum.
   Mean and Standard Deviation. Correlations. Significant digits. Central limit theorem. (12-13 Measuring in Aud. A!)
Nov 22: Error propagation (which is a science!). Estimate g measurement uncertainties.
Nov 25: ChiSquare method, evaluation, and test. Formation of Project groups.



# Header 1
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


