

KEY: description

Week 3 (Using Simulation and More Fitting):
* Dec 5: 8:15 - Group B: Project (for Wednesday the 14th of December) doing experiments in First Lab.
         9:15 - Group A: Systematic Uncertainties and analysis of "Table Measurement data". Discussion of real data analysis (usual rooms).
* Dec 6: Producing random numbers and their use in simulations.
* Dec 9: Summary of curriculum so far. Fitting tips and strategies.


KEY: Intro
### Producing random numbers

For producing random number, we have to main approacheds: the *transformation method* and the *accept-reject method*.

KEY: Transformation method
##### Transformation method

The transformation method is executed via the following **steps**:
$$
  \text{assert: PDF is normalized}
  \Rightarrow
\text{integrate}
  \Rightarrow
\text{invert}
$$

$$
    F(x) = \int_{-\infty}^x f(x') dx'
$$

KEY: Header 4

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

KEY: Accept-reject
#### Accept-reject (aka von Neumann method)

The idea is to generate random samples from a known distribution (called the "proposal distribution") and then accept or reject them according to a set of acceptance criteria. 

Here is an example of how the accept-reject method can be used to generate random samples from a uniform distribution:

KEY: Accept-reject code

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


KEY: Header 7

### How fast are these different methods?

Monte carlo fast: $\frac{1}{\sqrt{N}}$, 
numerical (eg Trapezoidal) is slower...: $\frac{1}{N^{2/d}}$, in which d is the dimensionality.
