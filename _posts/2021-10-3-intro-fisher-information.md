---
layout: post
title: An Introduction to Fisher Information
katex: True
orbit: True
---

A common question among statisticians and data analysts is how accurately we
can estimate the parameters of a distribution given some samples from it.
*Fisher information* can help answer this question by quantifying the amount of
information that the samples contain about the unknown parameters of the
distribution.

---

### Orbit and Spaced Repetition

This tutorial uses [Orbit](https://withorbit.com/), a learning tool for
periodic review. At the end of some of the sections there will be an Orbit
review area. The review will ask questions related to the material, and you
should attempt to answer them. If you create an account with Orbit, then you
will be periodically prompted (over email) to answer a few of the review
questions. Periodically answering the review questions will help you develop a
long-lasting memory of the material.

---

Suppose we have samples from a distribution where the form is known but the
parameters are not. For example, we might know the distribution is a Gaussian,
but we don't know the value of the mean or variance. We want to know
how difficult the parameters are to estimate given the samples. One way to
answer this question is to estimate the amount of information that the samples
contain about the parameters. The more information the samples contain about
the parameters, the easier they are to estimate. Conversely, the less
information the samples contain about the parameters, the harder they are to
estimate.

Fisher information is one way to measure how much information the samples
contain about the parameters. There are alternatives, but Fisher information is
the most well known. Before we get to the formal definition, which takes some
time to get familiar with, let's motivate Fisher information with an example.

Let's say we have a sample from a Gaussian distribution with a mean
$\mu$ and variance $\sigma^2$. The variance $\sigma^2$ is known, and the goal
is to estimate the mean, $\mu$. To understand how difficult this is, we would
like to know how much information we can expect the sample to contain about
$\mu$.  [Figure 1](#fig:gaussians) shows three Gaussian distributions with
three different variances. We expect that the mean is easiest to estimate for
the Gaussian distribution with the smallest variance. A random sample is more
likely to be close to the mean when the variance is small than when the
variance is large. This implies that the information content should grow as
$\sigma$ shrinks.

<div class="figure" id="fig:gaussians">
<img src="{{ site.base_url }}/images/intro_fisher_information/gaussians.svg" style="width:600px"/>
<div class="caption" markdown="span">
**Figure 1:** Three Gaussian distributions with increasing standard deviations.
As the standard deviation increases the samples carry less information about
the mean of the distribution.
</div>
</div>

Let's use $\mathcal{I}\_x(\mu)$ to represent the information content of a
sample $x$ at the mean $\mu$.  In the case of the Gaussian, we might expect the
information content of the sample to be inversely proportional to the variance,
$\mathcal{I}\_x(\mu) \propto 1 / \sigma^2$. In general the information content
will be a function of $\mu$, the parameter we want to estimate. Different
values of the parameter could be easier or harder to estimate. However, for the
Gaussian, $\mu$ only shifts the mode of the distribution, so the information
content only depends on $\sigma$ and not on $\mu$.

Another important point is that $x$ is a random sample. We don't want to
specify a value for $x$. Instead, we'd like the information content of $x$ to
consider all possible values for $x$ and their corresponding probabilities. A
value for $x$ which might tell us a lot about the parameter but is exceedingly
unlikely shouldn't contribute much to the expected information content of the
sample. Taking an expectation over $x$ is a natural way to account for this.

The Fisher information can be expressed in multiple ways, none of which are
easy to interpret at a glance. Let's start with one of these definitions and
follow it with an explanation.  Assume $x$ is a random variable sampled from a
distribution $p(x \mid \theta)$ where $\theta$ is an unknown scalar parameter.

**Definition** (Fisher information). The Fisher information content of $x$ at
$\theta$ is:
<div id="eq:fisher_information"/>
\begin{equation}
    \mathcal{I}_x(\theta) = \mathbb{E} \left[ \left(\frac{d}{d \theta} \log p(x \mid \theta)\right)^2 \right],
\end{equation}
where the expectation is taken over $x$.

The expectation is an integral if $x$ is continuous:
\\[
    \mathcal{I}_x (\theta) = \int_x p(x \mid \theta) \left(\frac{d}{d \theta} \log p(x \mid \theta)\right)^2 \, dx,
\\]
and a sum if $x$ is discrete. The distribution $p(x \mid
\theta)$ when viewed as a function of $\theta$ is the *likelihood function*, and $\log
p(x \mid \theta)$ is the *log-likelihood*. To simplify notation, let's use
$\ell(\theta \mid x)$ to represent the log-likelihood of the parameter $\theta$
given the sample $x$. The derivative of the log-likelihood with respect to the
parameter, $\ell^\prime(\theta \mid x)$, is called the *score function*. Using
this terminology, the Fisher information is the expected value of the square of
the score function:
\\[
    \mathcal{I}_x(\theta) = \mathbb{E} \left[ \ell^\prime(\theta \mid x) ^2 \right].
\\]

The Fisher information attempts to quantify the sensitivity of the random
variable $x$ to the value of the parameter $\theta$. If small changes in
$\theta$ result in large changes in the likely values of $x$, then the samples
we observe tell us a lot about $\theta$. In this case the Fisher information
should be high. This idea agrees with our interpretation of the Gaussian
distributions in [figure 1](#fig:gaussians). A small variance means we will see
large changes in the observed $x$ with small changes in the mean. In this case
the Fisher information of $x$ about the mean $\mu$ is large.

<div class="figure" id="fig:log_prob">
<img src="{{ site.base_url }}/images/intro_fisher_information/log_prob.svg" style="width:500px"/>
<div class="caption" markdown="span">
**Figure 2:** An example log-likelihood function $\ell(\theta \mid x)$. The
section of the log-likelihood surrounded by the dashed line changes rapidly as
a function of $\theta$. This would likely correspond to a region of high Fisher
information. The region surrounded by the dotted line changes slowly as a
function of $\theta$. This would likely correspond to a region of low Fisher
information.
</div>
</div>

[Figure 2](#fig:log_prob) plots an example of a log-likelihood function,
$\ell(\theta \mid x)$, for a single value of $x$. The curve highlighted by the
dashed line is a region where the log-likelihood changes rapidly as $\theta$
changes. This likely corresponds to a region of high Fisher information. The
part of the curve highlighted by the dotted line barely changes as a function
of $\theta$. This likely corresponds to a region of low Fisher information. I
say "likely" because the Fisher information is an expectation over all values
of $x$ and [figure 2](#fig:log_prob) only shows the log-likelihood for a single
value of $x$.

<orbit-reviewarea color="cyan">
  <orbit-prompt
    question="Is the mean of a Gaussian with a small variance easier or harder to estimate from a sample than the mean of a Gaussian with a large variance?"
    answer="Easier"
  ></orbit-prompt>
  <orbit-prompt
    question="State the definition of the Fisher information of a sample $x$ at the parameter $\theta$."
    answer="$\mathcal{I}_x(\theta) = \mathbb{E}\left[\ell^\prime(\theta \mid x)^2 \right]$"
  ></orbit-prompt>
  <orbit-prompt
    question="In the definition of Fisher information $$\mathcal{I}_x(\theta) = \mathbb{E}\left[\ell^\prime(\theta \mid x)^2 \right]$$ which variable is the expectation taken over?"
    answer="$x$"
  ></orbit-prompt>
  <orbit-prompt
    question="What is the name for $p(x \mid \theta)$ when viewed as a function of $\theta$?"
    answer="The likelihood function"
  ></orbit-prompt>
  <orbit-prompt
    question="What is the score function?"
    answer="The derivative of the log-likelihood function: $$\ell^\prime(\theta \mid x) = \frac{d}{d\theta} \log p(x \mid \theta)$$"
  ></orbit-prompt>
  <orbit-prompt
    question="What is the function $\ell(\theta \mid x)$ called?"
    answer="The log-likelihood"
  ></orbit-prompt>
  <orbit-prompt
    question="What is the function $\ell^\prime(\theta \mid x)$ called?"
    answer="The score function"
  ></orbit-prompt>
  <orbit-prompt
    question="Which sample has more information content about the mean of the Gaussian distribution: a sample from a larger variance distribution or a sample from a smaller variance distribution?"
    answer="The sample from a smaller variance distribution"
  ></orbit-prompt>

</orbit-reviewarea>


### Gaussian Distribution

As mentioned earlier, the log-likelihood in [figure 2](#fig:log_prob) is for a
single value of $x$. To compute the Fisher information, we need to consider the
log-likelihood for many values of $x$. Let's illustrate this with the example
of a Gaussian distribution.

<div class="figure" id="fig:fisher_gaussian_steps">
<img src="{{ site.base_url }}/images/intro_fisher_information/fisher_gaussian_steps.svg" style="width:700px"/>
<div class="caption" markdown="span">
**Figure 3:** The Gaussian distribution is on the left with the sequence of
transformations to arrive at the squared derivative of the log-likelihood
needed to compute the Fisher information in [equation
1](#eq:fisher_information) at the parameter $\mu$. The terms are plotted as a
function of $x$ for a mean $\mu = 0$ and variance $\sigma = 1$.
</div>
</div>

As before, let's say we have a random sample $x$ from a Gaussian distribution
for which we would like to compute the Fisher information at the unknown mean,
$\mu$. [Figure 3](#fig:fisher_gaussian_steps) shows from left to right the
construction of the term inside the expectation in [equation
1](#eq:fisher_information) used to compute the Fisher information. The
probability distribution is a zero-mean, unit-variance Gaussian distribution
($\mu = 0$ and $\sigma = 1$). [Figure 3c](#fig:fisher_gaussian_steps) shows the
derivative with respect to $\mu$ of the log-likelihood but as a function of
$x$.  [Figure 3d](#fig:fisher_gaussian_steps) shows the square of this
derivative as a function of $x$. The Fisher information is computed by taking
the expectation over $x$ of the curve in [figure
3d](#fig:fisher_gaussian_steps). In other words, we multiply the curve in
[figure 3a](#fig:fisher_gaussian_steps) with the curve in [figure
3d](#fig:fisher_gaussian_steps) and integrate the result.

The log of the Gaussian distribution is:
\\[
    \ell(\mu \mid x, \sigma) = \log p(x \mid \mu, \sigma)
        = -\left(\log (\sqrt{2\pi} \sigma) + \frac{1}{2\sigma^2}(x - \mu)^2\right),
\\]
and its derivative with respect to $\mu$, the score function, is:
\\[
    \ell^\prime(\mu \mid x, \sigma) =
        \frac{d}{d\mu} \log p(x \mid \mu, \sigma) = \frac{1}{\sigma^2}(x - \mu).
\\]
This derivative is shown in [figure 3c](#fig:fisher_gaussian_steps) but as a
function of $x$. To visualize this derivative, we can plot the log-likelihood.
The likelihood function of a Gaussian is also a Gaussian since the function is
symmetric in $x$ and $\mu$.  [Figure 4a](#fig:fisher_gaussian_derivs) shows the
log-likelihood for three values of $x$. For each value
of $x$, a tangent to the slope of the curve at $\mu=0$ (*i.e.* the derivative
plotted in [figure 3c](#fig:fisher_gaussian_steps)) is shown. The derivative of
the log-likelihood with respect to $\mu$ but as a function of $x$ is reproduced
in [figure 4b](#fig:fisher_gaussian_derivs). The slopes of the tangents in
[figure 4a](#fig:fisher_gaussian_derivs) correspond to the values of the points
in [figure 4b](#fig:fisher_gaussian_derivs).

<div class="figure" id="fig:fisher_gaussian_derivs">
<img src="{{ site.base_url }}/images/intro_fisher_information/fisher_gaussian_derivs.svg" style="width:700px"/>
<div class="caption" markdown="span">
**Figure 4:** In (a) the log of the Gaussian is plotted as a function of $\mu$
for three values of $x$ ($-1$, $0$ and $2$). For each curve, the tangent at
$\mu=0$ is plotted. This is used to compute the derivative of the
log-likelihood with respect to $\mu$ at $0$ which is shown in (b) as a function
of $x$.
</div>
</div>

The Fisher information of the Gaussian at $\mu$ is the expected value of
the squared score function:
\\[
    \mathcal{I}_x(\mu) = \mathbb{E}\left[\left(\frac{1}{\sigma^2}(x - \mu)\right)^2\right]
               = \frac{1}{\sigma^4} \mathbb{E}\left[(x - \mu)^2\right]
               = \frac{\sigma^2}{\sigma^4}
               = \frac{1}{\sigma^2}.
\\]
As expected, the Fisher information is inversely proportional to the variance.

### Bernoulli Distribution

Let's use the Bernoulli distribution as another example. The Bernoulli
distribution is that of a biased coin which has probability $\theta$ of turning
up heads (or $1$) and probability $1-\theta$ of turning up tails (or $0$). We
should expect that the more biased the coin, the easier it is to identify the
bias from an observation of the coin toss. As an extreme example, if $\theta$
is $1$ or $0$, then a single coin toss will tell us the value of $\theta$. The
Fisher information of the sample $x$ (the result of the coin toss) will be
higher the closer $\theta$ is to either $1$ or $0$.

<div class="figure" id="fig:fisher_bernoulli">
<img src="{{ site.base_url }}/images/intro_fisher_information/fisher_bernoulli.svg" style="width:700px"/>
<div class="caption" markdown="span">
**Figure 5:** The sequence of terms needed to compute the Fisher information
for a Bernoulli distribution at the parameter $\theta$. All of the terms are
plotted as a function of the parameter $\theta$ for both values of $x$, namely
$0$ and $1$.
</div>
</div>

The Bernoulli distribution $p(x \mid \theta)$ is plotted as a function of the
parameter $\theta$ in [figure 5a](#fig:fisher_bernoulli). The two values of
the distribution as a function of $\theta$ are $p(x=1 \mid \theta) = \theta$
and $p(x=0 \mid \theta) = 1 -\theta$. The log-likelihood for each value of $x$
is plotted as a function of $\theta$ in [figure 5b](#fig:fisher_bernoulli),
and the score function is plotted in
[figure 5c](#fig:fisher_bernoulli). The derivatives are:
\\[
    \frac{d}{d \theta} \log p(x=1 \mid \theta) = \frac{1}{\theta}
    \quad \textrm{and} \quad
    \frac{d}{d \theta} \log p(x=0 \mid \theta) = \frac{1}{\theta - 1}.
\\]
To get the Fisher information, shown in
[figure 5d](#fig:fisher_bernoulli), we take the expectation over $x$ of the
squared derivatives:
\\[
    \mathcal{I}_x(\theta) = \theta \frac{1}{\theta^2} + (1-\theta) \frac{1}{(\theta-1)^2}
     = \frac{1}{\theta} + \frac{1}{1 - \theta}
     = \frac{1}{\theta (1 - \theta)}.
\\]
The Fisher information in [figure 5d](#fig:fisher_bernoulli) has the shape we
expect. As $\theta$ approaches $0$ or $1$, the Fisher information grows
rapidly. Just as in the Gaussian distribution, the Fisher information is
inversely proportional to the variance of the Bernoulli distribution which is
$\textrm{Var}(x) = \theta (1-\theta)$. The smaller the variance, the more we
expect the sample of $x$ to tell us about the parameter $\theta$ and hence the
higher the Fisher information.

### Properties of Fisher Information

The Fisher information has several properties which make it easier to work
with. I'll mention two of the more salient ones here &mdash; the chain rule and the
post-processing inequality.

**Chain rule.** Analogous to the chain rule of probability, the Fisher
information obeys an additive chain rule:
\\[
\mathcal{I}\_{x, y}(\theta) =  \mathcal{I}\_{x \mid y}(\theta) + \mathcal{I}\_y(\theta).
\\]
The conditional Fisher information is defined as:
\\[
    \mathcal{I}\_{x \mid y}(\theta) = \int\_{y^\prime} p(y^\prime) \, \mathcal{I}\_{x \mid y=y^\prime}(\theta) \, d y^\prime
    = \int\_{y^\prime} p(y^\prime) \, \mathbb{E}\left[\left(\frac{d}{d \theta} \log p(x \mid y=y^\prime, \theta)\right)^2 \right] \, d y^\prime.
\\]
When the samples $x$ and $y$ are independent, the chain rule simplifies to:
\\[
\mathcal{I}\_{x, y}(\theta) =  \mathcal{I}\_{x}(\theta) + \mathcal{I}\_y(\theta).
\\]
So if we have $x\_1, \ldots, x\_n$ independent and identically distributed
samples, the Fisher information for all $n$ samples simplifies to $n$ times the
Fisher information of a single sample:
\\[
\mathcal{I}\_{x\_1, \ldots, x\_n}(\theta) =  \sum\_{i=1}^n \mathcal{I}\_{x\_i}(\theta) = n \, \mathcal{I}\_{x\_1}(\theta).
\\]

**Post-processing.** The Fisher information obeys a data processing
inequality. The Fisher information of $x$ at $\theta$ cannot be increased by
applying any function to $x$. If $f(x)$ is an arbitrary function of $x$, then:
\\[
\mathcal{I}\_{f(x)}(\theta) \le \mathcal{I}\_{x}(\theta).
\\]
The inequality holds with equality when $f(x)$ is a *sufficient statistic* for
$\theta$. A statistic is sufficient for $\theta$ if $\theta$ does not change
the conditional probability of $x$ given the statistic:
\\[ p(x~\mid~f(x),~\theta)~ =~p(x~\mid~f(x)). \\]

### Alternate Definitions

Fisher information can be expressed in two other equivalent forms. The first
form is:
\\[
    \mathcal{I}_x(\theta) = -\mathbb{E} \left[\ell^{\prime\prime}(\theta \mid x)  \right],
\\]
and the second form is:
\\[
    \mathcal{I}_x(\theta) = \textrm{Var}\left(\ell^\prime (\theta \mid x) \right).
\\]
A reason to know about these alternate definitions is that in some cases they
can be easier to compute than the version in [equation
1](#eq:fisher_information). To show these are equivalent to the definition in
[equation 1](#eq:fisher_information), we need a couple of observations.

<div id="obs:log_derivative_trick" />
**Observation 1.** This observation is sometimes called the log-derivative trick
in reinforcement learning algorithms. Using the chain rule of differentiation:
\\[
    \ell^\prime(\theta \mid x) = \frac{d}{d\theta} \log p(x \mid \theta)
        = \frac{1}{p(x \mid \theta)} \frac{d}{d \theta} p(x \mid \theta).
\\]

<div id="obs:expected_score" />
**Observation 2.** The expected value of the score function is zero. That is
$\mathbb{E}\left[\ell^\prime(\theta \mid x)\right] = 0$. We can show this using
the log-derivative trick from [observation 1](#obs:log_derivative_trick):
\\[
\begin{equation\*}
\begin{split}
\mathbb{E}\left[\ell^\prime(\theta \mid x)\right] &=
    \mathbb{E}\left[\frac{d}{d\theta} \log p(x \mid \theta) \right] \cr
    &= \mathbb{E}\left[\frac{1}{p(x\mid \theta)}\frac{d}{d\theta}  p(x \mid \theta) \right] \cr
    &= \int_x p(x \mid \theta) \frac{1}{p(x\mid \theta)}\frac{d}{d\theta}  p(x \mid \theta) \, dx \cr
    &= \int_x \frac{d}{d\theta}  p(x \mid \theta) \, dx \cr
    &= \frac{d}{d\theta} \int_x  p(x \mid \theta) \, dx = \frac{d}{d\theta} 1 = 0.
\end{split}
\end{equation\*}
\\]
In the last step above (and in the rest of this tutorial) we assume the derivative
and integral can be exchanged.

To show $\mathcal{I}\_x(\theta) = -\mathbb{E} \left[\ell^{\prime\prime}(\theta
\mid x)\right]$, we start by expanding the $\ell^{\prime\prime}(\theta \mid x)$
using [observation 1](#obs:log_derivative_trick) and then apply the product
rule of differentiation:
\\[
\begin{align\*}
    \ell^{\prime \prime}(\theta \mid x) &=
    \frac{d}{d\theta} \frac{d}{d\theta} \log p(x \mid \theta) \cr
    &= \frac{d}{d \theta} \left( \frac{1}{p(x \mid \theta) } \frac{d}{d \theta} p(x \mid \theta) \right) \cr
    &= -\frac{1}{p(x \mid \theta)^2 } \left(\frac{d}{d \theta} p(x \mid \theta)\right)^2  +
        \frac{1}{p(x \mid \theta) } \frac{d^2}{d \theta^2} p(x \mid \theta) \cr
    &= -\ell^\prime(\theta \mid x)^2  +
        \frac{1}{p(x \mid \theta) } \frac{d^2}{d \theta^2} p(x \mid \theta),
\end{align\*}
\\]
where we use [observation 1](#obs:log_derivative_trick) again in the last step.
The next step is to take the expectation over $x$:
\\[
    \mathbb{E} \left[\ell^{\prime \prime}(\theta \mid x) \right] =
    -\mathbb{E} \left[\ell^\prime(\theta \mid x)^2 \right]  +
        \mathbb{E} \left[ \frac{1}{p(x \mid \theta) } \frac{d^2}{d \theta^2} p(x \mid \theta) \right].
\\]
The first term is on the right is the negative of the Fisher information. The second term on the right is zero:
\\[
\begin{align\*}
\mathbb{E} \left[ \frac{1}{p(x \mid \theta) } \frac{d^2}{d \theta^2} p(x \mid \theta) \right]
    &= \int_x p(x \mid \theta) \frac{1}{p(x \mid \theta) } \frac{d^2}{d \theta^2} p(x \mid \theta)\, dx  \cr
    &= \int_x  \frac{d^2}{d \theta^2} p(x \mid \theta) \, dx \cr
    &= \frac{d^2}{d \theta^2} \int_x p(x \mid \theta) \, dx = \frac{d^2}{d \theta^2} 1 = 0.
\end{align\*}
\\]
Thus we have the result that $\mathcal{I}\_x(\theta) = -\mathbb{E}
\left[\ell^{\prime \prime}(\theta \mid x) \right]$.

The statement $\mathcal{I}\_x(\theta) = \textrm{Var}\left(\ell^\prime (\theta
\mid x) \right)$, follows directly from the fact that the expected value of the
score function is $0$ ([observation 2](#obs:expected_score)):
\\[
    \textrm{Var}\left(\ell^\prime(\theta \mid x)\right) =
        \mathbb{E}\left[\ell^\prime(\theta \mid x)^2 \right] -
        \mathbb{E}\left[ \ell^\prime(\theta \mid x) \right]^2
    = \mathbb{E}\left[ \ell^\prime(\theta \mid x)^2 \right]
    = \mathcal{I}_x(\theta).
\\]

### Multivariate Fisher Information

The definition of Fisher information can be extended to include multiple
parameters. In this case, the parameters of the distribution are now a
$d$-dimensional vector, $\theta \in \mathbb{R}^d$. The multivariate first-order
generalization of the Fisher information is:
\\[
    \mathcal{I}\_x(\theta) = \mathbb{E}\left[\nabla_\theta \ell(\theta \mid x)
        \nabla\_\theta \ell(\theta \mid x)^\top\right],
\\]
where $\nabla\_\theta$ is the gradient operator which produces the vector of
first derivatives of $\ell(\theta \mid x)$ with respect to $\theta$. Note in
this case the Fisher information is a symmetric matrix in $\mathbb{R}^{d \times
d}$. The diagonal entries of the matrix $\mathcal{I}\_x(\theta)\_{ii}$ have the
usual interpretation. The higher these entries are, the more information $x$
contains about the $i$-th parameter, $\theta\_i$. The off-diagonal entries are
somewhat more subtle to interpret.  Roughly speaking, if
$\mathcal{I}\_x(\theta)\_{ij}$ is high then $x$ contains more information about
the relationship between $\theta\_i$ and $\theta\_j$.

The multivariate second-order expression for the Fisher information is also a
natural extension of the scalar version:
\\[
    \mathcal{I}\_x(\theta) = -\mathbb{E}\left[ \nabla^2\_\theta \ell(\theta \mid x) \right],
\\]
where $\nabla^2_\theta$ is the operator which produces the matrix of second
derivatives of the log-likelihood with respect to $\theta$.

### Cramér-Rao Bound

An estimator for a parameter of a distribution is a function which takes as
input the sample and returns an estimate for the parameter. Let's use
$\hat{\theta}(x)$ to represent an estimator for the parameter $\theta$. The
estimator $\hat{\theta}(x)$ is unbiased if its expected value is equal to the
parameter $\theta$:
\\[
    \mathbb{E}\left[\hat{\theta}(x) \right\] = \theta.
\\]
The Cramér-Rao bound is an inequality which relates the variance of an
estimator of a parameter $\theta$ to the Fisher information of a sample $x$ at
$\theta$. If $x$ contains less information about $\theta$, then we expect
$\theta$ to be harder to estimate given $x$. The Cramér-Rao bound makes this
statement precise.  In the simplest case, if $\hat{\theta}(x)$ is an unbiased
estimator of $\theta$ given $x$, the Cramér-Rao bound states:
\\[
    \textrm{Var}\left(\hat{\theta}(x)\right) \ge \frac{1}{\mathcal{I}_x(\theta)}.
\\]
One way to think of the Cramér-Rao bound is as a two-player game say between
Alice and Bob:
1. Alice chooses the parameter $\theta$.
2. Alice samples $x \sim p(x \mid \theta)$ and sends $x$ to Bob.
3. Bob estimates $\hat{\theta}(x)$.

The Cramér-Rao bound says that on average the squared difference between Bob's
estimate and the true value of the parameter will be greater than $1 /
\mathcal{I}\_x(\theta)$.

The proof of the Cramér-Rao bound is only a few lines. First, since the
expected value of the score function is zero ([observation
2](#obs:expected_score)), we have:
\\[
    \textrm{Cov}\left(\hat{\theta}(x), \ell^\prime(\theta \mid x) \right)
    = \mathbb{E} \left[\hat{\theta}(x)\ell^\prime(\theta \mid x) \right] -
    \mathbb{E} \left[\hat{\theta}(x)\right]\mathbb{E}\left[\ell^\prime(\theta \mid x) \right]
    = \mathbb{E} \left[\hat{\theta}(x) \ell^\prime(\theta \mid x) \right].
\\]
Using the log-derivative trick again ([observation 1](obs:log_derivative)):
\\[
\begin{align\*}
    \mathbb{E} \left[\hat{\theta}(x) \ell^\prime(\theta \mid x) \right]
    &= \int_x p(x\mid \theta) \hat{\theta}(x) \frac{\partial}{\partial \theta} \log p(x \mid \theta) d\,x \cr
    &= \int_x p(x\mid \theta) \hat{\theta}(x) \frac{1}{p(x\mid \theta)} \frac{\partial}{\partial \theta} p(x \mid \theta) d\,x \cr
    &= \frac{\partial}{\partial \theta} \int_x \hat{\theta}(x) p(x \mid \theta) d\,x \cr
    &= \frac{\partial}{\partial \theta} \mathbb{E} \left[\hat{\theta}(x)\right] \cr
    &= \frac{\partial}{\partial \theta} \theta = 1,
\end{align\*}
\\]
where in the second-to-last step we use the fact that the estimator is
unbiased. This tells us that the covariance is one:
\\[
    \textrm{Cov}\left(\hat{\theta}(x), \ell^\prime(\theta \mid x) \right) = 1.
\\]
Combining this with the Cauchy-Schwarz inequality we have:
\\[
    \textrm{Var}\left(\hat{\theta}(x)\right) \textrm{Var}\left(\ell^\prime(\theta \mid x) \right)
    \ge \textrm{Cov}\left(\hat{\theta}(x), \ell^\prime(\theta \mid x)\right)^2 = 1,
\\]
hence:
\\[
    \textrm{Var}\left(\hat{\theta}(x)\right) \ge \frac{1}{\textrm{Var}\left(\ell^\prime(\theta \mid x) \right)}
    = \frac{1}{\mathcal{I}_x(\theta)}.
\\]

<orbit-reviewarea color="cyan">
  <orbit-prompt
    question="State the chain rule property of Fisher information?"
    answer="$\mathcal{I}_{x, y}(\theta) = \mathcal{I}_{x \mid y}(\theta) + \mathcal{I}_{y}(\theta)$"
  ></orbit-prompt>
  <orbit-prompt
    question="If $f(x)$ is an arbitrary function of $x$, what is the relationship of $\mathcal{I}_{f(x)}(\theta)$ to $\mathcal{I}_x(\theta)$?"
    answer="$\mathcal{I}_{f(x)}(\theta) \le \mathcal{I}_x(\theta)$"
  ></orbit-prompt>
  <orbit-prompt
    question="State the conditional Fisher information $\mathcal{I}_{x \mid y}(\theta)$ in terms of $p(y^\prime)$ and $\mathcal{I}_{x \mid y=y^\prime}(\theta)$."
    answer="$\mathcal{I}_{x \mid y}(\theta) = \int_{y^\prime} p(y^\prime) \, \mathcal{I}_{x \mid y=y^\prime}(\theta) \, d y^\prime$"
  ></orbit-prompt>
  <orbit-prompt
    question="In words, explain the post-processing inequality for Fisher information."
    answer="No amount of post-processing of the sample $x$ can increase the Fisher information content of $x$ at $\theta$."
  ></orbit-prompt>
  <orbit-prompt
    question="If we have $n$ independent and identically distributed samples each with Fisher information content $\mathcal{I}(\theta)$, what is the Fisher information content of all of the samples?"
    answer="$n\, \mathcal{I}(\theta)$"
  ></orbit-prompt>
   <orbit-prompt
    question="State the second-order definition of the Fisher information of $x$ at $\theta$."
    answer="$\mathcal{I}_x(\theta) = -\mathbb{E}\left[\ell^{\prime \prime}(\theta \mid x) \right]$"
  ></orbit-prompt>
   <orbit-prompt
    question="What is the log-derivative trick?"
    answer="$\frac{d}{d \theta} \log p(x \mid \theta) = \frac{1}{p(x \mid \theta)}\frac{d}{d\theta} p(x \mid \theta)$"
  ></orbit-prompt>
   <orbit-prompt
    question="What is the variance of the score function, $\textrm{Var}\left(\ell^\prime(\theta \mid x)\right)$, equivalent to?"
    answer="The Fisher information"
  ></orbit-prompt>
   <orbit-prompt
    question="What is the expected value of the score function $\mathbb{E}\left[ \ell^\prime(\theta \mid x)\right]$?"
    answer="$0$"
  ></orbit-prompt>
   <orbit-prompt
    question="Is the Fisher information a vector or a matrix when $\theta \in \mathbb{R}^d$? What is its size?"
    answer="The Fisher information is a matrix with size $d \times d$"
  ></orbit-prompt>
   <orbit-prompt
    question="Define the Fisher information when $\theta \in \mathbb{R}^d$ in terms of $\nabla_\theta \ell(\theta~\mid~x)$."
    answer="$\mathcal{I}_x(\theta) = \mathbb{E}\left[\nabla_\theta \ell(\theta \mid x) \nabla_\theta \ell(\theta \mid x)^\top \right]$"
  ></orbit-prompt>
   <orbit-prompt
    question="What does it mean for an estimator $\hat{\theta}(x)$ of a parameter $\theta$ to be unbiased?"
    answer="$\mathbb{E}\left[\hat{\theta}(x) \right] = \theta$"
  ></orbit-prompt>
   <orbit-prompt
    question="What is the name for the inequality $\textrm{Var}\left(\hat{\theta}(x)\right) \ge 1 / \mathcal{I}_x(\theta)$?"
    answer="The Cramér-Rao bound"
  ></orbit-prompt>
   <orbit-prompt
    question="State the Cramér-Rao bound for an unbiased estimator $\hat{\theta}(x)$ of a parameter $\theta$."
    answer="$\textrm{Var}\left(\hat{\theta}(x)\right) \ge 1 / \mathcal{I}_x(\theta)$"
  ></orbit-prompt>
</orbit-reviewarea>

### More Uses of Fisher Information

The Fisher information has applications beyond quantifying the difficulty in
estimating parameters of a distribution given samples from it. I'll briefly
discuss two such applications: natural gradient descent and data privacy.

**Natural gradient descent.** Fisher information is used to compute the natural
gradient used in numerical optimization. Natural gradient
descent[^natural_gradient] is not commonly used directly in large
machine-learning problems due to computational difficulties, but it motivates
some more commonly used optimization methods.

One way to view standard gradient descent is that it searches for the best
update within a small region around the current parameters. This region is
defined by the standard Euclidean distance to the existing parameters. Natural
gradient descent is the same idea, but instead of defining the region with
Euclidean distance it defines the region using the Kullback-Liebler (KL)
divergence. In this case the KL divergence is used to measure the distance
between the likelihood function at the current parameters and the likelihood
function at the updated parameters.

Natural gradient descent looks similar to Newton's method. It replaces the
inverse Hessian with the inverse of the expected Hessian, which is the same as
the inverse of the Fisher information matrix. The update is:
\\[
\mathcal{I}\_(\theta)^{-1} \nabla_\theta \mathcal{L}(\theta),
\\]
where $\mathcal{L}$ is the likelihood function.

**Data privacy.** One relatively recent use for Fisher information (which is an
area I am working on[^privacy]) is using it as a tool for data privacy. We can
invert the role of the samples and the parameters and measure the Fisher
information of the parameters at the samples. The parameters in this case could
be a machine-learning model, and the samples are data from different individuals
on which the model was trained. The Fisher information of the model about the
data then quantifies the privacy loss for different individuals when revealing
the model.

### Footnotes

[^natural_gradient]:
    See for example Shun-ichi Amari, *Natural Gradient Works Efficiently in Learning*, Neural Computation, 1998. [(link)](https://direct.mit.edu/neco/article/10/2/251/6143/Natural-Gradient-Works-Efficiently-in-Learning)

[^privacy]:
    Our recent research on this is detailed in Hannun, *et al.*, *Measuring Data Leakage in Machine-Learning Models with Fisher Information*, Uncertainty in Artificial Intelligence, 2021.  [(link)](https://arxiv.org/abs/2102.11673)
