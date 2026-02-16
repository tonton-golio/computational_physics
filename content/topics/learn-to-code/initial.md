# Getting Started with Programming



# Hello world!
> Go ahead and **print** the *string*; "Hello world!"

# Variables
Varaibles let us remember values. Variable assignment is done with the **=** sign.
> Assign numbers to two variables and print their sum.

# Operators
Python can also math, try your standard operators :
$$
\begin{align*}
    + && - && * && / && ** && // && \%
\end{align*}
$$
> Define two numbers to variables, and use each operator on them.


# For-loops
Python can do things many times with **loops**!
> Assign you name to a variable & loop over the letters, printing each.


# Functions
Functions are lists of instructions written as:
```python
def funcName(x):
    print(x)
```
and called like; `funcName(x)`

> Write a function which prints the difference between the squareroot and the square of a single positive number.


# Bubble-sort
bubble sort is a an algorithm for sorting an array, which only makes nearest-neighbour comparisons. $x$ is an array of random numbers.
> sort $x$ using for-loops and comparisons


# kMeans
$\mathbf{X}$ is a two dimensional array, with 100 samples. Within the span of $\mathbf{X}$, are three centroids. 

> Write a script to update the location of the centroids to be the center of those samples which are nearest this centroid.

# kMeans solution
```python
for i in range(5):
    # ditance to centroids
    d2c = np.array([(x-c)**2 for c in cs]).sum(axis=2).T
    
    # affiliations
    aff = np.argmin(d2c, axis=1)
    
    # update centroids
    cs = np.array([np.mean(x[aff == a], axis=0) for a in set(aff)])
```

