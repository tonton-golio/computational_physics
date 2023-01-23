
# title
Quantifying noise of gene expression

# Week 1 description
- Nov 21: How to characterize noise in gene expression

# Cellular identity
In multicellular organism, all cells have same DNA. 
Then why they are so diverse?
The answer is gene expression.
Depending on cell type, expressed genes are different.

# What is gene expression?
When we say genes are expressed, it means there are protein from that gene.
The idea of gene expression is formulated as the Central Dogma.
## The Central Dogma
The central dogma is the idea of genetic information flow. 
The information of DNA are transcribed to RNA and then the information of RNA will
be translated to protein.
## What do we need to express the gene?
Gene expression is controlled by enzyme.
- RNA polymerase: transcribe DNA to RNA 
- Robisome: translate RNA to protein

# Transcriptional regulation
There are two type of promotor which is part of DNA. 
Promotor attract protein which regulate gene expression such as RNA polymerase, 
repressor, activator and so on.
- Strong promotor: it attarcts RNA polymerase strongly, so without repressor the gene is always on.
- Weak promotor: it attracts RNA polymerase weakly, so without activator the gene is always off.
- Exmaple of promotor - *lac* promotor.

# How to measure gene expression?
To measure gene expression, we can fuse reporter protein such as GFP to target DNA 
region. 
We can visualize the gene expression by fluorescent protein!

# Two way of measuring gene expression - single cell and bulk
Easiest way to measure gene expression is measuring light intensity of sum of 
10.000.000.000 cells in test tube.
Recent single-cell measurement technology revealed that gene expression is NOISY.

# Why is the gene expression noisy?
Low copy numbers such as transcription factors and genes cause the noisy gene 
expression.
Noise in gene expression leads to different cell-fates in a genetically homogeneous
population.
Examples are below.
- Bistability in *comK* expression
- bacterial persister cells

# Molecules in the cell move via diffusion
From FRAP and single-molecule measurement, it was revealed that molecule inside of
the cell has Brownian motion.
Thus the chemical reactions inside of the cell have stochasticity.
Each cell's gene expression is depending on ramdom event of molecule's reactions, 
but by focusing on the distribution of the cell, we can quantify the noise.

# Definition of the total noise
We define the total nosie as 
$$
\eta(t) 
= 
\frac{\text{standard deviation}}{\text{average}}
= 
\frac{\sqrt{\text{variace}}}{\text{average}}
$$
Here average and variace of number of protein $N(t)$ is 
$$
\left< N(t) \right>
=
\frac{1}{n} \sum_{j=1}^n N_j(t)
$$
$$
\left< (N(t) - \left<N(t)\right> )^2 \right>
=
\frac{1}{n} \sum_{j=1}^n (N_j(t) - \left<N(t)\right> )^2 
$$
$j$ is index of cell.
From this the total noise is mathematically
$$
\eta(t) 
=
\frac
{\sqrt{\left< (N(t) - \left<N(t)\right> )^2 \right>}}
{\left< N(t) \right>}
$$
Mathematically this total nosie can be decomposed into two type of noise i.e.
extrinsic noise and intrinsic noise.

# Extrinsic noise
If the two noise are correlated e.g. two gene expressions in the same cell
are correlated, covariance become nonzero.
By defining $N_j^{(i)}$ of number of protein $i$ in the cell $j$,  covariance is
$$
\left< 
\left( N^{(1)}(t) - \left<N^{(1)}(t)\right> \right)
\left( N^{(2)}(t) - \left<N^{(2)}(t)\right> \right)
\right>
\\=
\left< N^{(1)}(t)N^{(2)}(t) \right>
-
\left< N^{(1)}(t) \right>
\left< N^{(2)}(t) \right>
$$
Thus, extrinsic noise is defined as
$$
\eta_\mathrm{E} (t)
=
\frac
{
    \sqrt{
        \left< N^{(1)}(t)N^{(2)}(t) \right>
        -
        \left< N^{(1)}(t) \right>
        \left< N^{(2)}(t) \right>
    }
}
{
    \sqrt{
        \left< N^{(1)}(t) \right>
        \left< N^{(2)}(t) \right>
    }

}
$$
Here denominator means devided by average.

# Intrinsic noise
We can measure non-correlated noise by quantifying the gap between two gene 
expression.
$$
 N^{(1)}(t) - N^{(2)}(t) 
$$
Intrinsic noise would be
$$
\eta_\mathrm{I} (t)
=
\frac
{
    \sqrt{
        \left< \left(N^{(1)}(t)-N^{(2)}(t) \right)^2 \right>
    }
}
{
    \sqrt{
        2
        \left< N^{(1)}(t) \right>
        \left< N^{(2)}(t) \right>
    }

}
$$
Again denominator is division by average and extra 2 will make sense later.

# Decomposing total noise into intrinsic noise and extrinsic noise
By summing up square of extrinsic noise and intrinsic noise, we obtain total noise.
$$
\begin{align*}

&\eta_\mathrm{E}(t)^2  + \eta_\mathrm{I}(t)^2 
\\&= 
\frac
{
    \left< N^{(1)}(t)N^{(2)}(t) \right>
    -
    \left< N^{(1)}(t) \right>
    \left< N^{(2)}(t) \right>
    +
    \frac{1}{2}
    \left< \left(N^{(1)}(t)-N^{(2)}(t) \right)^2 \right>
}
{
    \left< N^{(1)}(t) \right>
    \left< N^{(2)}(t) \right>

}
\\&=
\frac
{
    \left< N^{(1)}(t)N^{(2)}(t) \right>
    -
    \left< N^{(1)}(t) \right>
    \left< N^{(2)}(t) \right>
    +
    \frac{1}{2}
    \left(
    \left< N^{(1)}(t)^2 \right>
    +
    \left< N^{(2)}(t)^2 \right>
    \right)
    -
    \left< N^{(1)}(t)N^{(2)}(t) \right>
}
{
    \left< N^{(1)}(t) \right>
    \left< N^{(2)}(t) \right>

}
\\&=
\frac
{
    \frac{1}{2}
    \left(
    \left< N^{(1)}(t)^2 \right>
    +
    \left< N^{(2)}(t)^2 \right>
    \right)
    -
    \left< N^{(1)}(t) \right>
    \left< N^{(2)}(t) \right>
}
{
    \left< N^{(1)}(t) \right>
    \left< N^{(2)}(t) \right>

}

\end{align*}

$$
If $N^{(1)}(t) = N^{(2)}(t) = N(t)$, this is same as total noise.




