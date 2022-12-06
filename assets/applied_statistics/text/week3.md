

# description

Week 3 (Using Simulation and More Fitting):
Dec 5: 8:15 - Group B: Project (for Wednesday the 14th of December) doing experiments in First Lab.
         9:15 - Group A: Systematic Uncertainties and analysis of "Table Measurement data". Discussion of real data analysis (usual rooms).
Dec 6: Producing random numbers and their use in simulations.
Dec 9: Summary of curriculum so far. Fitting tips and strategies.

# Header 1
### Systematic Uncertainties and analysis of "Table Measurement data"
...

# Header 2
### Producing random numbers and their use in simulations.

For producing random number, we have to main approacheds: the *transformation method* and the *accept-reject method*.

# Header 3
#### Transformation method

**steps**:
1. ensure that the PDF is normalized
1. integrate
1. invert

But which integral:

$$
    F(x) = \int_{-\infty}^x f(x') dx'
$$

# Header 4
**Example**

consider the exponential dist.
$$
    f(x) = \lambda \exp (-\lambda x), x\in [0,\infty]
$$
normalized? yes

integrate:
$$
    F(x) = 1 - \exp(-\lambda x)
$$

invert:

$$
    F^{-1}(p) = \frac{\log (1-p)}{\lambda}
$$

# Header 5
### Accept-reject

#### the accept reject method (aka von Neumann method)

generate random x and random y, we have f(x). If y corresponding to an x is greater than f(x), then reject f(x).


The accept-reject method is a common technique for generating random samples from a distribution. The idea is to generate random samples from a known distribution (called the "proposal distribution") and then accept or reject them according to a set of acceptance criteria. Here is an example of how the accept-reject method can be used to generate random samples from a uniform distribution:

In the code on the right, sample_uniform generates a random sample from the uniform distribution with lower bound a and upper bound b. accept_reject accepts or rejects the sample according to the acceptance criteria, and sample generates a specified number of samples from the uniform distribution using the accept-reject method.
# blank
## Header 6
```python
# Sample from the uniform distribution
# with lower bound a and upper bound b
def sample_uniform(a, b):
  return random.uniform(a, b)

# Accept or reject the sample according to the
# acceptance criteria
def accept_reject(x, pdf, cdf):
  # Generate a random number from the uniform distribution
  u = random.uniform(0, 1)

  # Accept the sample if u <= pdf(x) / cdf(x)
  # where pdf is the probability density function
  # of the proposal distribution and cdf is the
  # cumulative density function of the target distribution
  if u <= pdf(x) / cdf(x):
    return x
  else:
    return None

# Generate samples from the uniform distribution
def sample(a, b, num_samples):
  samples = []

  # Generate samples until we have num_samples
  while len(samples) < num_samples:
    # Sample from the uniform distribution
    x = sample_uniform(a, b)

    # Accept or reject the sample
    y = accept_reject(x, pdf, cdf)
    if y is not None:
      samples.append(y)

  return samples
```
## Blank



# Header 7

### How fast are these different methods?

monte carlo fast AF: $\frac{1}{\sqrt{N}}$

numerical (eg Trapezoidal) is slower...: $\frac{1}{N^{2/d}}$, in which d is the dimensionality.


*side note, also be aware of binning when dealing with discrete data, as sometimes we get multiple different values in a bin...*