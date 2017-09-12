---
layout: post
title: An In-depth Guide to Connectionist Temporal Classification
mathjax: true
---

In this tutorial you will learn about the *Connectionist Temporal
Classification* (CTC) algorithm -- a popular model for neural sequence
transduction.

**Contents**
 
---
1. TOC
{:toc}

## The Problem 

We've seen an explosion in the use of neural networks for mapping sequences to
sequences. One work-horse algorithm for neural sequence transduction is CTC.
Certain assumptions made by the CTC algorithm make it especially well suited
for problems such as automatic speech recognition and hand-writing recognition,
where it's used in many state-of-the-art models.

Let's start with the problem which will allow us to setup some notation for
later. Let $$X = [x_1, x_2, \ldots, x_T]$$ be an input sequence of length
$$T$$. Let $$Y = [y_1, y_2, \ldots, y_U]$$ denote the length $$U$$ output
sequence. Let $$\mathcal{X}$$ and $$\mathcal{Y}$$ be the set of all sequences
$$X$$ and $$Y$$ respectively. The problem is to learn a "good" mapping of
elements of $$\mathcal{X}$$ to elements of $$\mathcal{Y}$$. For example, in
speech recognition a "good" mapping is one which takes an audio input to an
accurate transcription.

The sequence transduction problem has three properties which make it difficult
to use traditional supervised learning algorithms.
- Not all $$Y \in \mathcal{Y}$$ have the same length. The same is usually true
  for $$X \in \mathcal{X}$$. In other words, the lengths $$T$$ and $$U$$ can
  vary.
- The ratio of the lengths $$T$$ and $$U$$ can vary.
- We don't have an accurate alignment (a correspondence of the elements) of
  $$X$$ and $$Y$$.

In these circumstances, CTC can learn a function to map elements of
$$\mathcal{X}$$ to a distribution over members of $$\mathcal{Y}$$. We can then
infer a likely $$Y$$ from this distribution. For such an algorithm to be useful
it needs to solve two problems: the objective function should be computable and
inference should be tractable.

**Objective Function:** We should be able to efficiently compute a score for
how likely any $$Y$$ is given an $$X$$. In our case the score will be a
conditional probability $$p(Y \mid X)$$, though this isn't a strict requirement.
The function $$p(Y \mid X)$$ should be differentiable. This makes optimizing
the function parameters easier. We need to compute the score $$p(Y \mid X)$$
for $$X$$'s and $$Y$$'s of variable and differing lengths. We also want to
avoid algorithms which require an alignment between $$X$$ and $$Y$$.

**Inference:** Given a model for $$p(Y \mid X)$$, for any $$X$$ we
need to tractably infer a likely $$Y$$. This means solving 
\begin{align}
Y^\* = \text{argmax}\_{Y \in \mathcal{Y}} p(Y \mid X).
\end{align}
Ideally an optimal $$Y$$ can be efficiently found. With CTC we'll settle for a
close to optimal solution which is usually not too expensive to find.

## The Algorithm

The first thing we need to do is compute a score of how likely a $$Y$$ is given
an $$X$$. To do this, the CTC model allows a set of alignments between $$X$$
and $$Y$$. To get around the fact that $$X$$ and $$Y$$ have variable lengths
and the alignment is unknown, CTC *marginalizes* over all possible allowed
alignments between the two sequences. In this section we'll cover what these
allowed alignments are, how to compute the CTC loss function and how to perform
inference with a learned model.

### Alignment
The CTC algorithm assumes that alignments between the input $$X$$ and the
output $$Y$$ have a specific form. To motivate the CTC alignments, first
consider a naive approach.

Let's use an example. Let the input length $$T = 6$$ and $$Y =$$ [c, a, t]. In
this case, one way to align $$Y$$ and $$X$$ would be to let the elements of
$$Y$$ align to multiple elements of $$X$$. So the alignment could be [c, c, a,
a, a, t]. This approach has two problems.
- Some elements of $$X$$ may not correspond to any element of $$Y$$. In this
  case, we may not want to force every element of $$X$$ to match with an
  element of $$Y$$.
- The sequence $$Y$$ can have consecutive repeat characters. This means the
  alignments for a given $$Y$$ are not unique. We have no way to determine if
  the alignment [c, c, a, a, a, t] refers to [c, a, t] or [c, a, a, t] among
  others. Later we'll need to distinguish between these cases to perform
  inference with the model.

To get around these problems, CTC introduces a new token to the set of allowed
output characters. This new token is sometimes called the "blank" token. We'll
refer to it here as $$\epsilon$$. When an input element aligns to $$\epsilon$$
then no output element corresponds to it. 

The alignments allowed by CTC are of length $$T$$, the length of the input. Let
$$\mathcal{A}$$ be the set of alignments. We can compute $$\mathcal{A}$$ from
$$Y$$ in two steps. 
1. Optionally insert an $$\epsilon$$ at the beginning, end, and between every
   character of $$Y$$.  
2. Arbitrarily repeat any character present in the $$\epsilon$$-expanded $$Y$$
   (including $$\epsilon$$) until the new sequence has length $$T$$.

This produces one possible alignment. The set $$\mathcal{A}$$ contains all possible
alignments we can generate this way. If there are consecutive repeat characters
in $$Y$$ then the $$\epsilon$$ between them is required. This allows us to
differentiate between alignments of [c, a, t] and [c, a, a, t]. 

Let's go back to the example. A few allowed alignments of $$Y$$ are
[$$\epsilon$$, c, c, $$\epsilon$$, a, t], [c, c, a, a, t, t] and [c,
$$\epsilon$$, $$\epsilon$$, $$\epsilon$$, a, t]. A few disallowed alignments are
[c, $$\epsilon$$, c, $$\epsilon$$, a, t], [c, c, a, t, t] and [c, $$\epsilon$$,
$$\epsilon$$, $$\epsilon$$, t, t]. These are not allowed for the following
reasons: the first corresponds to $$Y =$$ [c, c, a, t], the second has length
$$5$$, and the third is missing the 'a'.

A convenient way to visualize the alignment between $$X$$ and $$Y$$ is with a
2D alignment matrix. Here's one for the [$$\epsilon$$, c, c,
$$\epsilon$$, a, t] alignment. Sometimes it's useful to include the
$$\epsilon$$'s in the alignment matrix. Both are shown below. Note that some of
the $$\epsilon$$ tokens are left unaligned. They are optional since they don't
fall between consecutive repeat characters in $$Y$$.

<div class="figure">
<img src="{{ site.base_url }}/images/ctc/alignment.svg" />
<div class="caption" markdown="span">
Alignment matrices of the output $$Y =$$ [c, a, t] to an input with length $$T
= 6$$.  The alignment is given by $$A =$$ [$$\epsilon$$, c, c, $$\epsilon$$, a,
t]. **Left:** The alignment without the $$\epsilon$$ token. **Right:** The
alignment including the $$\epsilon$$ token.
</div>
</div>

We can observe several properties about CTC right away. First, the allowed
alignments between $$Y$$ and $$X$$ are strictly monotonic. If the 'c' in our
example aligns to the first input element then the 'a' must align to input
element two or greater. This implies a second property: the length of $$Y$$ can
be no longer than the length of $$X$$. A third property is that the alignment
of $$X$$ to $$Y$$ is many-to-one. Many input elements can align to a single
output element but not vice-versa.

### Loss Function
With the alignment setup, we can write down the CTC objective function for a
single $$(X, Y)$$ pair
\begin{align}
p(Y \mid X) = \sum\_{A \in \mathcal{A}} \prod\_{t=1}^T p_t(a\_t \mid X).
\end{align}
Recall $$\mathcal{A}$$ is the set of alignments of $$Y$$ to $$X$$ allowed by
CTC. The element $$A = [a_1, \ldots, a_T]$$ is a member of $$\mathcal{A}$$. The
function $$p_t(a_t \mid X)$$ can be any function which produces a distribution
for each input time-step $$t$$ over the output alphabet given the full
input $$X$$.

The inner product of the loss function computes the probability of a given
alignment $$A$$ frame-by-frame. The outer sum marginalizes over all possible
alignments to give the conditional output probability $$p(Y \mid X)$$.

The hard part of the CTC loss is not understanding what it is, but computing it
efficiently. The set $$\mathcal{A}$$ can be *very* large.[^0] For most
practical problems, we can't delineate the elements of $$\mathcal{A}$$ to
compute the sum above.

Instead, we can compute the CTC loss both exactly and efficiently with a
dynamic programming algorithm. Like any dynamic programming algorithm, the key
is recognizing the subproblems.

To simplify the problem let the sequence $$Z = [z_{1}, \ldots, z_{2U+1}]$$
consist of the elements of $$Y$$ with an $$\epsilon$$ at the beginning, end,
and between every character. Let $$\alpha_{i,j}$$ be the CTC score for the
subsequences $$X_{1:i}$$ and $$Z_{1:j}$$. We can compute $$\alpha_{i, j}$$ if
we know the values $$\alpha_{\lt i, \lt j}$$. There are two cases.

**Case 1:** Either $$z_{j} = \epsilon$$ or $$z_{j} =
z_{j-2}$$ (a consecutive repeat character in $$Y$$). In this case
\begin{align}
\alpha\_{i, j} = (\alpha\_{i-1, j-1} + \alpha\_{i-1, j}) p_i(z_{j} \mid X).
\end{align}
In this case, only alignments which have matched $$z_{j-1}$$ to some previous
input element are allowed. The first condition is because we can't leave any
characters in $$Y$$ unaligned. The second condition is because we *must* have
an $$\epsilon$$ between repeat characters.

**Case 2:** If we aren't in the first case, then we're in the second case and
\begin{align}
\alpha\_{i, j} = (\alpha\_{i-1, j-2} + \alpha\_{i-1, j-1} + \alpha\_{i-1, j}) p_i(z_j \mid X).
\end{align}
We have this case because $$\epsilon$$ between unique characters is optional.
Alignments which align $$z_{j-2}$$ but not $$z_{j-1}$$ to a previous input are
allowed. Below is an animation of the steps taken by the dynamic programming
algorithm.

<div class="figure">
<img src="{{ site.base_url }}/images/ctc/ctc_cost.svg" />
<div class="caption" markdown="span">
An animation of the dynamic programming algorithm used to compute the CTC
score. The input $$X$$ with $$T=6$$ is on the vertical axis. The
$$\epsilon$$-expanded output $$Y =$$ [c, a, t] is on the horizontal axis. Node
$$(i, j)$$ in the diagram represents $$\alpha_{i,j}$$ -- the CTC score between
the subsequences $$X_{1:i}$$ and $$Z_{1:j}$$. The two states used to
compute the final score are marked with concentric circles.
</div>
</div>

The final score is the sum of the two final states, marked in the example
figure with concentric circles,
\begin{align}
p(Y \mid X) = \alpha\_{T,S} + \alpha\_{T, S-1}
\end{align}
where $$S = 2U + 1$$.

As long as the individual output model $$p_i(z \mid X)$$ is differentiable then
the entire loss function is differentiable. This is true since computing $$p(Y
\mid X)$$ consists only of sums and products of the $$p_i(z \mid X)$$.

The time complexity of this dynamic programming algorithm is $$O(TU)$$ with a
fairly small constant. Conveniently, the time to compute the loss function does
not depend at all on the size of the output alphabet.

For a training set $$\mathcal{D}$$, the parameters of a model are tuned to
minimize the negative log-likelihood
\begin{align}
\sum\_{(X, Y) \in \mathcal{D}} -\log p(Y \mid X)
\end{align}
as opposed to maximizing the likelihood directly.
 
### Inference

To infer the most likely output sequence for a given input we solve
\begin{align}
Y^\* = \text{argmax}\_{Y \in \mathcal{Y}} p(Y \mid X).
\end{align}

The simplest method to compute a likely $$Y$$ is to take the most likely output
at each time-step. Since there are no conditional dependencies in the output,
this computes exactly
\begin{align}
A^\* = \text{argmax}\_{A \in \mathcal{A}} p_t(a_t \mid X).
\end{align}
We can then collapse repeat characters and remove $$\epsilon$$ tokens to
produce $$Y$$.

In most applications this simple algorithm works well because CTC tends to
allocate most of the probability to a single alignment $$A$$. However, this
method is not guaranteed to find the most likely $$Y$$. This is because
multiple alignments can map to the same $$Y$$. The sequences [a, a,
$$\epsilon$$] and [a, a, a] could individually have lower probability than [b,
b, b], though the sum of their probabilities could be larger. In this case, the
beam search algorithm would propose [b, b, b] as the most likely hypothesis,
corresponding to $$Y =$$ [b]. To account for this, the algorithm should
consider the fact that [a, a, a] and [a, a, $$\epsilon$$] map to the same
output, namely $$Y =$$ [a].

We can solve this problem with a modified beam search. The modified beam search
is also not guaranteed to find the most likely $$Y$$, but it has the nice
property that we can trade-off more computation (namely a larger beam-size) for
an asymptotically better solution.

A regular beam search would compute a new set of hypotheses at each input
time-step. The new set of hypotheses is generated from the previous set by
extending each hypothesis with all possible output characters.

<div class="figure">
<img src="{{ site.base_url }}/images/ctc/beam_search.svg" />
<div class="caption" markdown="span">
Steps two and three of a standard beam search algorithm with an output
alphabet $$\{\epsilon, a, b\}$$ and a beam size of three. The current
hypotheses in the beam are given by the blue nodes. The proposed extensions are
grey and the selected extensions are red.
</div>
</div>

We can modify the vanilla beam search to handle multiple alignments mapping to
the same output. In this case instead of keeping a list of alignments in the
beam, we store the output prefixes after collapsing repeats and removing
$$\epsilon$$ characters. At each step of the search we accumulate scores for a
given prefix based on all the alignments which map to it. The image below
displays steps two, three and four of the algorithm. The dashed lines indicate
the output prefix that the proposed extension maps to.

<div class="figure">
<img src="{{ site.base_url }}/images/ctc/prefix_beam_search.svg" />
<div class="caption" markdown="span">
Steps two, three and part of step four of a CTC beam search with an output
alphabet $$\{\epsilon, a, b\}$$ and a beam size of three. The $$\lambda$$ token
indicates the empty string. The blue nodes indicate the current hypothesis in
the beam and the red and grey nodes indicate possible extensions. The red nodes
are the outputs included in the selected hypotheses for the next time-step. The
dashed lines indicate the hypotheses they contribute to.
</div>
</div>

A proposed extension can map to two output prefixes if the character is a
repeat. This is shown at $$T=3$$ in the figure above where 'a' is proposed as
an extension to the prefix [a]. Both [a] and [a, a] are valid outputs with this
proposed extension. For the [a,a] case we should only include the part of the
score of the previous prefix for alignments which end in $$\epsilon$$. This is
because $$\epsilon$$ must be between consecutive repeat characters. For the [a]
case, where we don't extend the prefix, we should only consider the part of the
score of the previous prefix for alignments which don't end in $$\epsilon$$. 

Given this, it's necessary to keep track of two probabilities for each prefix
in the beam. The probability of all alignments which end in $$\epsilon$$ and
the probability of all alignments which don't end in $$\epsilon$$. Note that
when we rank the hypotheses at each step before pruning the beam, we should
rank by the combined score.

The implementation of this algorithm does not require much code. The code is,
however, dense and tricky to get right. Refer to this [gist][decoder-gist] for
an example implementation in Python.

In some problems, such as speech recognition, incorporating a language model
over the outputs significantly improves accuracy. In this case we can
repose the inference problem as 
\begin{align}
Y^\* = \text{argmax}\_{Y \in \mathcal{Y}} p(Y \mid X) p(Y)^\alpha L(Y)^\beta.
\end{align}
The function $$L(\cdot)$$ computes the length of $$Y$$ in terms of the language
model tokens and serves as a word insertion bonus. The language model scores
are only included when a prefix is extended by a character (or word) and not at
every step of the algorithm. This causes the search to favor shorter prefixes,
as measured by $$L(\cdot)$$, since they do not have many language model
updates. The insertion bonus helps with this. The parameters $$\alpha$$ and
$$\beta$$ are usually set by cross-validation. 

## Properties of CTC

We briefly mentioned a few important properties of CTC so far. Here we'll go
into more depth on what these properties are and what trade-offs they offer. 

### Conditional Independence

One of the most commonly cited shortcomings of CTC is the conditional
independence assumption it makes. The cost function we wrote down above models
the conditional output distribution as
\begin{align}
p(Y \mid X) = \sum\_{A \in \mathcal{A}} \prod\_{t=1}^T p(a\_t \mid X).
\end{align}
The model assumes that a given output element is conditionally independent of
the other output elements given the input. For many sequence transduction
problems this is not a valid assumption. Say we had an audio clip of someone
saying "triple A".[^1] Another valid transcription could be "AAA". If the first
letter of the predicted transcription is 'A', then with high probability the
next letter should be 'A' and with low probability the next letter should be
'r'. The conditional independence assumption does not allow for this.

In fact speech recognizers using CTC are not able to learn an implicit language
model nearly as well as other models which do not make this conditional
independence assumption.[^2] However, this isn't always a bad trait. In some
cases, having the model implicitly learn a strong language model over the output
can make it less adaptable to new or altered domains. For example we might want
to adapt a model trained to transcribe on phone conversations between friends
to customer support calls. 

### Alignment Properties

While the CTC algorithm does make strong assumptions about the form of
alignments between $$X$$ and $$Y$$, technically the algorithm is
*alignment-free*. The objective function marginalizes over the allowed
alignments thus model is agnostic as to how probability is distributed amongst
them. In some problems CTC ends up allotting most of the probability towards a
single alignment, though this is not guaranteed. To force the model to upweight
a single alignment, we can replace the `sum` over the
alignments with a `max`,
\begin{align}
p(Y \mid X) = \max\_{A \in \mathcal{A}} \prod\_{t=1}^T p(a\_t \mid X).
\end{align}

As mentioned before, CTC allows only *strictly monotonic* alignments. In some
problems such as speech recognition this may be a valid assumption. For other
problems such as machine translation where a future word in a target sentence
can align to an earlier part of the source sentence, this assumption might be a
deal-breaker. The *strictly monotonic* property also implies that the length of
the output must be no greater than the length of the input.[^3] For problems
where this is often not the case, CTC will not work.

A final important property of CTC alignments is that they are *many-to-one*. In
other words, multiple input element can align to at most one output element. In
some cases this may not be desirable. We may want to enforce a strict
one-to-one correspondence between elements of $$X$$ and $$Y$$. Alternatively,
we may want to allow multiple output elements to align to a single input
element. For example the sound made by "th" might align to a single input frame
of audio, but a character based CTC model would not allow for this.

### Input Synchronous Inference

Inference with CTC is done in an *input synchronous* manner as opposed to an
*output synchronous* manner. This means the beam search is pruned after
processing each input element and the algorithm terminates when all of the
input has been seen. This is opposed to output synchronous decoding which
prunes the beam after each output time-step and typically terminates on an
end-of-sequence marker. Input synchronous inference makes streaming the
decoding process easier. For some applications streaming the inference
computation is critical to achieve low latency response times.

## CTC in Context 

In this section we'll discuss how CTC relates to other commonly used
algorithms for sequence transduction. 

### HMMs
*This section requires some familiarity with the HMM and is not critical to
understanding the CTC algorithm. Feel free to skip it on a first read.*

At a first glance a Hidden Markov Model (HMM) based sequence transducer does
not closely resemble a CTC model. However, the two algorithms have many
similarities. Understanding the relationship between the two models helps to
understand what exactly CTC does that couldn't be done before. Putting CTC in
this context will also allow us to understand how it can be changed and
potentially improved for various use cases.

We'll use the same notation from before, $$X$$ is the input sequence and $$Y$$
is the output sequence with lengths $$T$$ and $$U$$ respectively. Like before
we're interested in finding a "good" model for $$p(Y \mid X)$$. One way to
simplify the modeling problem is to transform this probability with Bayes' Rule
and compute
\begin{align}
p(Y \mid X) \propto p(X \mid Y) p(Y).
\end{align}
The $$p(Y)$$ term is straight-forward to model with a language model, so let's
focus on $$p(X \mid Y)$$. Like before we'll let $$\mathcal{A}$$ be a set of
allowed of alignments of $$Y$$ to $$X$$. In this case members of
$$\mathcal{A}$$ have length $$T$$.  Let's otherwise leave $$\mathcal{A}$$
unspecified for now. We'll come back to it later. We can marginalize over
$$\mathcal{A}$$ to get
\begin{align}
p(X \mid Y) = \sum\_{A \in \mathcal{A}} p(X, A \mid Y).
\end{align}
To simplify notation, let's remove the conditioning on $$Y$$, it will be
unchanging in every $$p(\cdot)$$. Using the HMM assumptions we can
write
\begin{align}
p(X) = \sum\_{A \in \mathcal{A}} \prod\_{t=1}^T p(x_t \mid a_t) p(a_t \mid a\_{t-1}).
\end{align}
Two assumptions have been made here. The first is the usual Markov property.
The state $$a_t$$ is conditionally independent of all historical states given
the previous state $$a_{t-1}$$. The second is that the observation $$x_t$$ is
conditionally independent of everything else given the current state $$a_t$$.

Let's assume that the transition probabilities $$p(a_t \mid a_{t-1})$$ are
uniform. This gives
\begin{align}
p(X) \propto \sum_{A \in \mathcal{A}} \prod\_{t=1}^T p(x_t \mid a_t).
\end{align}
This equation is starting to resemble the CTC loss function from above. In fact
there are only two differences. The first is that we are learning a model of
$$X$$ given $$Y$$ as opposed to $$Y$$ given $$X$$. The second is how the set
$$\mathcal{A}$$ is produced. Let's deal with each in turn.

The HMM can be used with discriminative models which estimate $$p(a \mid x)$$.
To do this, we apply Bayes' rule and rewrite the model as 
\begin{align}
p(X) &\propto \sum_{A \in \mathcal{A}} \prod\_{t=1}^T \frac{p(a_t \mid x_t)p(x_t)}{p(a_t)} \\\
&\propto \sum_{A \in \mathcal{A}} \prod\_{t=1}^T \frac{p(a_t \mid x_t)}{p(a_t)}. 
\end{align}

If we assume a uniform prior over the states $$a$$ and condition on all of
$$X$$ instead of a single element at a time, we arrive at 
\begin{align}
\sum_{A \in \mathcal{A}} \prod\_{t=1}^T p(a_t \mid X). 
\end{align}

The above equation is essentially the CTC loss function, assuming the set
$$\mathcal{A}$$ is the same. In fact, the HMM framework does not specify what
$$\mathcal{A}$$ should consist of. This part of the model can be designed on a
per-problem basis. In many cases the model doesn't condition on $$Y$$ and the
set $$\mathcal{A}$$ consists of all possible length $$T$$ sequences from the
output alphabet. In this case, the HMM can be drawn as an  *ergodic* state
transition diagram in which every state connects to every other state.  The
figure below shows this model with the alphabet or set of unique hidden states
as $$\{a, b, c\}$$.

In our case the hidden states of the model (the elements of $$A$$) are strongly
related to $$Y$$. We want the HMM to reflect this. One possible model could be
a simple linear state transition diagram. The figure below shows this with the
same alphabet as before and $$Y =$$ [a, b]. Another commonly used model is the
*Bakis* or left-right HMM. In this model any transition which proceeds from the
left to the right is allowed. 

<div class="figure">
<img src="{{ site.base_url }}/images/ctc/ergodic_hmm.svg" />
<img style="margin-bottom:30px;" src="{{ site.base_url }}/images/ctc/linear_hmm.svg" />
<img style="margin-bottom:30px;" src="{{ site.base_url }}/images/ctc/ctc_hmm.svg" />
<div class="caption" markdown="span">
Three different HMM state transition diagrams. **Left:** The ergodic HMM. Any
node can be either a starting or final state.  **Middle:** The linear HMM. The
first node is the starting state and the second node is the final state.
**Right:** The CTC HMM. The first two nodes are the starting states and the
last two nodes are the final states. 
</div>
</div>

In CTC we augment the alphabet with $$\epsilon$$ and the HMM model allows a
subset of the left-right transitions. In this model there are two start
states and two accepting states.

One possible source of confusion is that the HMM model differs for any unique
$$Y$$. This is in fact standard in applications such as speech recognition. The
state diagram changes based on the output $$Y$$. However, the functions which
estimate the observation and transition probabilities are shared.

Let's discuss how CTC improves on the original HMM model. First, we can think
of the CTC state diagram as a special case HMM which works well for many
problems of interest. Incorporating the blank as a hidden state in the HMM
allows us to use the alphabet of $$Y$$ as the other hidden states. This model
also gives a set of allowed alignments which may be a good prior for some
problems. Perhaps most importantly, CTC is discriminative as it models $$p(Y
\mid X)$$ directly. This allows us to train the model "end-to-end" and
unleashes the capacity of powerful models like the RNN.

### Encoder-Decoder Models

The neural encoder-decoder is perhaps the most commonly used framework for
sequence transduction. This class of models consists of an encoder and a
decoder. The encoder maps the input sequence $$X$$ into a hidden
representation. The decoder consumes the hidden representation and produces a
distribution over the output space $$\mathcal{Y}$$. We can write this as
\begin{align}
h &= \texttt{encode}(X) \\\
p(Y \mid X) &= \texttt{decode}(h).
\end{align}
The $$\texttt{encode}(\cdot)$$ and $$\texttt{decode}(\cdot)$$ functions are
typically RNNs. The decoder can optionally be equipped with an attention
mechanism. The hidden state $$h$$ usually has dimensions $$T \times d$$ where
$$d$$ is a hyperparameter. Sometimes the encoder subsamples the input. If the
encoder subsamples the input by a factor $$s$$ then the inner dimension of
$$h$$ will be $$\frac{T}{s}$$.

We can interpret CTC in the encoder-decoder framework. This is helpful to
understand the developments in encoder-decoder models that are applicable to
CTC. Also, it's useful to develop a common language for the properties of these
models.

**Encoder:** The encoder of a CTC model can be just about any encoder we find
in commonly used encoder-decoder models. For example the encoder could be a
multi-layer bidirectional RNN or a convolutional network. There is a constraint
on the CTC encoder that doesn't apply to the others. We can't subsample the
input at such a rate that $$\frac{T}{s}$$ is less than $$U$$. 

**Decoder:** We can view the decoder of a CTC model as a simple linear
transformation followed by a softmax normalization. This layer should project
all $$T$$ components of the encoder output $$h$$ into the dimensionality of the
output alphabet.

## Practitioner's Guide

So far we've mostly developed a conceptual understanding of CTC. Here we'll go
through a few implementation tips for practitioners.

**Software:** Even with a solid conceptual understanding of CTC, the
implementation is difficult. The algorithm has several edge cases and a fast
implementation needs to be written in a lower-level programming language.
Open-source software tools can make getting started with CTC much easier:

- Baidu Research has open-sourced [warp-ctc]. The package is written in C++ and
  CUDA. The CTC loss function runs on either the CPU or the GPU. Bindings are
  available for Torch, TensorFlow and [PyTorch].
- TensorFlow has built in [CTC loss] and [CTC beam search] functions. The
  TensorFlow version doesn't support the GPU yet.

**Numerical Stability:** Computing the CTC loss naively is numerically
instable.  One method to avoid this is to normalize the $$\alpha$$'s at each
time-step.  The [original publication][ctc-2006] has more detail on this
including the adjustments to the gradient. In practice this works well enough
for medium length sequences but can still underflow for long sequences.
Another solution is to compute the loss function in log-space with the
[log-sum-exp trick][log-sum-exp]. Inference should also be done in log-space
using the log-sum-exp trick.

**Beam Search:** There are a couple of good tips to know about when
implementing  and using the CTC beam search.

The correctness of the beam search can be tested as follows.
1. Run the beam search algorithm on an arbitrary input. 
2. Save the inferred output $$\bar{Y}$$ and the corresponding score $$\bar{c}$$. 
3. Compute the actual CTC score $$c$$ for $$\bar{Y}$$ using the same input. 
4. Check that $$\bar{c} \approx c$$ with the former being no greater than the later.
As the beam size increases the inferred output $$\bar{Y}$$ may change, but the two
numbers should grow closer.

A common question when using a beam search decoder is the size of the beam to
use.  There is a trade-off between accuracy and runtime. We can check if the
beam size is in a good range. To do this first compute the CTC score for the
inferred output $$c_i$$. Then compute the CTC score for the ground truth output
$$c_g$$. If the two outputs are not the same, we should have $$c_g \lt c_i$$.
If $$c_i << c_g$$ then the beam search is performing poorly and a large
increase in the beam size may be warranted.

### Bibliographic Notes
{:.no_toc}

The CTC algorithm was first published by [Graves et al., 2006][ctc-2006]. The
first experiments were on [TIMIT], a popular phoneme recognition benchmark.
Chapter 7 of Graves' [thesis] gives a more detailed treatment of CTC.

One of the first applications of CTC to large vocabulary speech recognition was
by [Graves et al., 2014][graves-2014] where they used a hybrid HMM and CTC
trained model to achieve state-of-the-art results. [Hannun et al., 2014]
subsequently demonstrated state-of-the-art CTC based speech recognition on
larger benchmarks. [Liwicki et al., 2007] achieved state-of-the-art results on
an online handwriting recognition task using the CTC algorithm with an RNN.

The Hidden Markov Model was developed in the 1960's with the first application
to speech recognition in the 1970's. For an introduction to the HMM and
applications to speech recognition see Rabiner's [canonical tutorial]. 

Encoder-decoder models were simultaneously developed by [Cho et al., 2014] and
[Sutskever et al., 2014].  [Olah & Carter, 2016] give an in-depth guide to
attention in encoder-decoder models in the online publication *Distill*.

### Acknowledgements
{:.no_toc}

### Footnotes
{:.no_toc}

[^0]:
    For a $$Y$$ of length $$U$$ without any repeat characters and an $$X$$ of
    length $$T$$ the size of the set is $${T + U \choose T - U}$$. For $$T=100$$
    and $$U=50$$ this number is almost $$10^{40}$$.

[^1]:
    This example is from [Chan et al., 2015](https://arxiv.org/pdf/1508.01211.pdf).

[^2]:
    This is demonstrated in e.g. [Battenberg et al.,
    2017](https://arxiv.org/pdf/1707.07413.pdf).

[^3]:
    Note that if there are $$r$$ consecutive repeats in $$Y$$, the length $$U$$
    must be less than $$T$$ by $$2r - 1$$.


[log-sum-exp]: https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
[ctc-2006]: ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
[TIMIT]: https://catalog.ldc.upenn.edu/ldc93s1
[thesis]: https://www.cs.toronto.edu/~graves/phd.pdf
[Hannun et al., 2014]: https://arxiv.org/pdf/1412.5567.pdf
[Liwicki et al., 2007]: https://www.cs.toronto.edu/~graves/icdar_2007.pdf
[canonical tutorial]: http://www.cs.ubc.ca/~murphyk/Software/HMM/rabiner.pdf
[Cho et al., 2014]: https://arxiv.org/pdf/1406.1078.pdf
[Sutskever et al., 2014]: https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
[Olah & Carter, 2016]: https://distill.pub/2016/augmented-rnns/
[warp-ctc]: https://github.com/baidu-research/warp-ctc
[PyTorch]: https://github.com/awni/warp-ctc
[CTC loss]: https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
[CTC beam search]: https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder
[graves-2014]: http://proceedings.mlr.press/v32/graves14.html
[decoder-gist]: https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0
