---
layout: post
title: The Label Bias Problem
katex: True
---

Many sequence classification models suffer from the *label bias problem*.
Understanding the label bias problem and when a certain model suffers from it
is subtle but is essential to understand the design of models like conditional
random fields and graph transformer networks.

The label bias problem mostly shows up in discriminative sequence models. At
its worst, label bias can cause a model to completely ignore the current
observation when predicting the next label. How and why this happens is the
subject of this section. How to fix it is the subject of the next section.

Suppose we have a task like predicting the parts-of-speech for each word in a
sentence. For example, take the sentence "the cat sat" which consists of
the tokens `[the, cat, sat]`. We'd like our model to output the sequence
`[ARTICLE, NOUN, VERB]`.

A classic discriminative sequence model for solving this problem is the maximum
entropy Markov model (MEMM).[^mcallum00] The graphical model for the MEMM is
shown below. Throughout, $X\\!=\\![x_1, \ldots, x_T]$ is the input or observation
sequence and $Y\\!=\\![y_1, \ldots, y_T]$ is the output or label sequence.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/label-bias/memm.svg" style="width:450px"/>
<div class="caption" markdown="span">
The graphical model for the maximum entropy Markov model (MEMM).
</div>
</div>

The MEMM makes two assumptions about the data. First, $y_t$ is conditionally
independent of all previous observations and labels given $x_t$ and $y_{t-1}$.
Second, the observations $x_t$ are independent of one another. The first
assumption is more central to the model while the second sometimes varies and
is easier to relax. These assumptions can be seen from the graphical model.
Mathematically, the model is written as 
\\[
 p(Y \mid X) = \prod_{t=1}^T p(y_t \mid x_t, y_{t-1}).
\\]
The probabilities are computed with a softmax:
\\[
 p(y_t \mid x_t, y_{t-1}) = \frac{e^{s(y_t, x_t, y_{t-1})}}{\sum_{i=1}^c e^{s(y_i, x_t, y_{t-1})}}
\\]
where $c$ is the number of output classes and $s(y_t, x_t, y_{t-1})$ is a
scoring function which should give a higher score for $y_t$ which are likely to
be the correct label. Since we normalize over the set of possible output labels
at each time step, we say the model is *locally normalized* and $p(y_t \mid
x_t, y_{t-1})$ are the *local probabilities*. The distribution $p(Y \mid X)$ is
valid since summing over all possible $Y$ of length $T$ equals one, and all
the values are non-negative.

Let's return to our example of `[the, cat, sat]`. We can represent the
inference process on this sequence using a graph. The states (or nodes)
represent the set of possible labels. The transitions (or edges) represent the
possible observations along with the associated probabilities $p(y_t~\\!\mid~\\!x_t,
y_{t-1})$. 

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/label-bias/memm_inference_normalized.svg" style="width:450px"/>
<div class="caption" markdown="span">
An inference graph for the MEMM with the label set `{ARTICLE, NOUN, VERB}`.
Each arc is labelled with the probability of transitioning to the corresponding
state given the observation. The sum of the probabilities on arcs leaving a
node for a given observation should sum to one. For simplicity not all
observation, score pairs are pictured.
</div>
</div>

To compute the probability of `[ARTICLE, NOUN, VERB]`, we just follow the
observations along each arc leading to the corresponding label. In this case
the probability would be $1.0\\!\times\\!0.9\\!\times\\!1.0$.

As another example, say we drop the article and just have the sequence `[cat
sat]`. While it's not a great sentence, the correct label should be `[NOUN,
VERB]`. However, if we follow the probabilities we see that the score for
`[ARTICLE, NOUN]` is $0.9\\!\times\\!0.3\\!=\\!0.27$ whereas the score for `[NOUN, VERB]` is
$0.1\\!\times\\!1.0\\!=\\!0.1$.

The model is not used to seeing `cat` at the start of a sentence, so the scores
leading from the start state are poorly calibrated. What we need is information
about how uncertain the model is for a given observation and previous label
pair. If the model has rarely seen the observation `cat` from the starting node
`<S>` then we want to know that, and it should be included in the scores.

It's actually possible that the model did at some point implicitly store this
uncertainty information. However, by normalizing the outgoing scores for a
given observation, we are forcing this information to be discarded. Take a look
at the following *unnormalized* inference graph.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/label-bias/memm_inference_unnormalized.svg" style="width:480px"/>
<div class="caption" markdown="span">
An *unnormalized* inference graph for the MEMM with the label set `{ARTICLE,
NOUN, VERB}`. Each arc is labelled with the score of transitioning to the
corresponding state given the observation. For simplicity not all observation,
score pairs are pictured.
</div>
</div>

The unnormalized inference graph corresponds exactly to the normalized
inference graph when the scores are exponentiated and normalized. However, we
see something interesting here. The outgoing scores for `cat` from `<S>` are
small. Recall lower scores are worse. This means the model is much less
confident about the observation `cat` from the start state than the observation
`the` which has a score of 100. This information is completely erased when we
normalize. That's one symptom of the label bias problem.

Notice in the unnormalized graph, the score for `[ARTICLE, NOUN]` is $5 + 21 =
26$ while the score for `[NOUN, VERB]` is $3 + 100 = 103$. The right answer
gets a better score in the unnormalized graph! Note, we are adding scores here
instead of multiplying them because the unnormalized scores are in log-space.
In other words $p(y_t \mid x_t, y_{t-1}) \propto e^{s(y_t, x_t, y_{t-1})}$.

We can see the label bias problem quantitatively. This observation is due to
Denker and Burges.[^denker94] Suppose our scoring function $s(y_t, x_t,
y_{t-1})$ factorizes into the sum of two functions $f(y_t, x_t, y_{t-1})$ and
$g(x_t, y_{t-1})$. Suppose further that $f(\cdot)$ mostly cares about how good
the predicted label $y_t$ is given the observation $x_t$, whereas $g(\cdot)$
mostly cares about how good the observation $x_t$ is given the previous label
$y_{t-1}$. If we compute the local probabilities using this factorization, we
get:
\\[
p(y_t \mid x_t, y_{t-1}) =
    \frac{e^{f(y_t, x_t, y_{t-1}) + g(x_t, y_{t-1})}}{
        \sum_{i=1}^c e^{f(y_i, x_t, y_{t-1}) + g(x_t, y_{t-1})}}
     = \frac{e^{f(y_t, x_t, y_{t-1})}}{\sum_{i=1}^c e^{f(y_i, x_t, y_{t-1})}}.
\\]
The contribution of $g(\cdot)$ in the numerator and denominator cancels. This
causes all the information about how likely the observation is given the
previous state to be erased. 

### Conservation of Score Mass

The label bias problem results from a "conservation of score mass" (Bottou,
1991).[^bottou91] Conservation of score mass just says that the outgoing scores
from a state for a given observation are normalized. This means that all of the
incoming probability to a state must leave that state. An observation can only
dictate how much of the incoming probability to send where. It cannot
change the total amount of probability leaving the state. The net result is any
inference procedure will bias towards states with fewer outgoing transitions.

Suppose we have three states, `A`, `B` and `C`, shown below. State
`A` has four outgoing (nonzero) transitions, state `B` only has two and state
`C` has just one. Suppose all three states distribute probability mass equally
among their successor states: $p(y_t \mid x_t, y_{t-1})$ is uniform. 

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/label-bias/number_transitions.svg" style="width:600px"/>
<div class="caption" markdown="span">
An example of three states, `A`, `B` and `C`, which have uniform outgoing
transition distributions. Label bias will cause the inference procedure to
favor paths which go through state `C`. 
</div>
</div>

Neither state `A`, `B` nor `C` are doing anything useful here, so we shouldn't
prefer one over the other. But the inference procedure will bias towards paths
which go through state `C` over `B` and `A`. Paths which go through `A` will be
the least preferred. To understand this, suppose that the same amount of
probability arrives at the three states. State `A` will decrease the
probability mass for any path by a factor of four, whereas state `B` will only
decrease a given path's score by a factor of two and state `C` won't penalize any
path at all. In every case the observation is ignored, but the state with the
fewest outgoing transitions is preferred.

Even if outgoing transitions from states `A` and `B` did not ignore their
observations, they would still reduce a paths score since the probabilities
aren't likely to be one. This would cause state `C` to be preferred even though
it always ignores it's observation.

### Entropy Bias

In a less contrived setting where the distribution $p(y_t \mid x_t, y_{t-1})$
is not the same for every observation, our model will bias towards states which
have a low entropy distribution over next states given the previous state.
Note this is distinct from the distribution $p(y_t \mid x_t, y_{t-1})$ which
can have low entropy without directly causing label bias. However, if the
conditional distribution $p(y_t \mid y_{t-1})$ has low entropy then we are
potentially in trouble. For example, in the figure above, $p(y \mid
\texttt{B})$ has lower entropy than $p(y \mid \texttt{A})$.

Consider the three cases below. In each case there are two possible
observations `a` and `b` and two possible successor states. We'd like to know
which one will introduce the most label bias into the model. To answer that
question, we need to make an assumption about the prior probability over
observations. Suppose that the prior, $p(x_t)$, is uniform (e.g. $p(a) = p(b) =
0.5$).
<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/label-bias/entropy_example.svg" style="width:600px"/>
<div class="caption" markdown="span">
An example of three states, `A`, `B` and `C`, each with two possible outgoing
transitions. Each transition is labelled with the observation and probability
pair for two observations, `a` and `b`.
</div>
</div>
We can calculate $p(y_t \mid y_{t-1})$ for any state since
\\[
p(y_t \mid y_{t-1}) = \sum_i p(y_t \mid x_i, y_{t-1}) p(x_i) = \frac{1}{n} \sum_i p(y_t \mid x_i, y_{t-1})
\\]
where we used the fact that $p(x_t)$ is uniform and there are $n$ possible
observations. In our case $n\\!~=\\!~2$. The following figure shows $p(y_t \mid
y_{t-1})$ on the corresponding arc for each example.
<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/label-bias/entropy_example_no_obs.svg" style="width:600px"/>
<div class="caption" markdown="span">
An example of three states, `A`, `B` and `C`, each with two possible outgoing
transitions.  Each transition is labelled with the probability $p(y_t \mid
y_{t-1})$ which are computed from the probabilities in the figure above.
</div>
</div>
Case (a) has the lowest entropy transition distribution whereas case (b) and
(c) are equivalent. Intuitively, we expect case (a) to be worse than case (b)
since it biases towards the upper path, whereas (b) does not bias towards
either. However, case (c) is interesting. In case (c), the observation can have
a large effect on the outcome. While this effect might be wrong if the
probabilities are poorly calibrated, this case doesn't cause label bias more
than case (b) under the assumption of a uniform prior.

### Revising Past Mistakes

Another description of the label bias problem is that it does not allow a model
to easily recover from past mistakes. This follows from the conservation of
score mass perspective. If the outgoing score mass of a path is conserved, then
at each transition the mass can only decrease. In the future, if a path
encounters new evidence which makes it very likely to be correct, it cannot
increase the path's score. The most this new evidence can do is to not decrease
the path's score by preserving all of the incoming mass for that path. So the
model's ability to promote a path given new evidence is limited even if we are
certain that the new observation makes this path the correct one.

## Overcoming the Label Bias Problem

As we observed in the `[cat, sat]` example above, if we don't normalize scores
at each step, then the path scores can retain useful information. This implies
that we should avoid locally normalizing probabilities.

One option is that we don't normalize at all. Instead, when training, we simply
tune the model parameters to increase the score
\\[
    s(X, Y) = \sum_{t=1}^T s(y_t, x_t, y_{t-1})
\\]
for every $(X, Y)$ pair in our training set. The problem with this is there is
no competition between possible outputs. This can result in the model
maximizing $s(Y, X)$ while completely ignoring its inputs. So we need some kind
of competition between outputs. If the score of one output goes up, then others
should feel pressure to go down.

This can be achieved with *global normalization*. We compute the probability as
$\gdef\Y{\mathcal{Y}}$
\\[
p(Y \mid X) = \frac{e^{\sum_{t=1}^T s(y_t, x_t, y_{t-1})}}{
    \sum_{Y^\prime \in \Y_T} e^{\sum_{t=1}^T s(y_t^\prime, x_t, y^\prime_{t-1})}}
\\]
where $\Y_T$ is the set of all possible $Y$ of length $T$.
This is exactly a linear chain conditional random field (CRF).[^lafferty01] The
graphical model and hence dependency structure is the same as the MEMM in the
previous section. The only difference is how we normalize the scores for a
given input.

With this normalization scheme, the label bias problem is no longer an issue.
In fact when performing inference, we need not normalize at all. The
normalization term is constant for a given $X$ and hence the ordering between
the possible $Y$ will be preserved without normalizing. What this means is that
the inference procedure operates on an unnormalized graph just like the one we
saw for the part-of-speech tagging example in the previous section.

Because transitions have unnormalized scores, they are free to affect the
overall path score anyway they please. If `cat` is very unlikely to follow
`<s>` the model can retain that information by keeping the scores for all
transitions out of `<s>` low. Then whenever we see `cat` following the state
`<s>`, the path score won't be affected much since the model is uncertain of
what the correct next label is.

This freedom from the label bias problem comes at a cost. Computing the
normalization term exactly is more expensive with a CRF than with an MEMM.
With an MEMM we normalize locally. Each local normalization costs
$\mathcal{O}(c)$ where $c$ is the number of classes, and we have to compute $T$
of them, so the total cost is $\mathcal{O}(cT)$. With a linear chain CRF, on
the other-hand, the total cost using an efficient dynamic programming algorithm
called the forward-backward algorithm, is $\mathcal{O}(c^2T)$.[^sutton12] If
$c$ is large this can be a major hit to training time. For more complex
structures where there can be longer-range dependencies between the outputs,
beam search is usually the only option to approximate the normalization
term.[^collobert19]

## A Brief History of the Label Bias Problem

The first recorded observation of the label bias problem was in Léon Bottou’s
PhD thesis.[^bottou91] The term "label bias" was coined in the seminal work of
Lafferty, McCallum and Pereira introducing conditional random
fields.[^lafferty01] Solving the label bias problem was one of the motivations
for developing the CRF. The CRF was one of earliest discriminative sequence
models to give a principled solution to the label bias problem. 

An even earlier sequence model which overcame the label bias problem was the
check reading system proposed by Denker and Burges[^denker94], though they did
not use the term label bias. This work motivated the graph transformer networks
of Bottou, LeCun, Bengio and others.[^bottou97] For more references on
graph transformer networks visit Léon Bottou's page on [structure learning
systems](https://leon.bottou.org/research/structured).

## A Few Examples

In this section we'll look at a few examples of models, some of which suffer
from label bias and some of which do not. 

### Hidden Markov Model

The hidden Markov model (HMM) is a generative model which makes two assumptions
about the data generating distribution. First, it assumes that the observation
$x_t$ is conditionally independent of all other $y$ and $x$ given the hidden
state (i.e. label) at time $t$, $y_t$. Second, the HMM makes the usual Markov
independence assumption that $y_t$ is conditionally independent of all previous
$y$ given $y_{t-1}$. In equations
\\[
p(X, Y) = p(y_0) \prod_{t=1}^T p(x_t \mid y_t) p(y_t \mid y_{t-1}).
\\]
This is a very different model from the MEMM. It's generative, not
discriminative, so we estimate $p(X, Y)$ and not $p(Y \mid X)$. Interestingly,
the only difference between the graphical model for an HMM and the MEMM is the
direction of the arrows between $x_t$ and $y_t$.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/label-bias/hmm.svg" style="width:450px"/>
<div class="caption" markdown="span">
The graphical model for the hidden Markov model (HMM).
</div>
</div>

As a simple rule of thumb, generative models do not suffer from label bias. One
way to see this for the HMM specifically is to look at the corresponding
inference graph.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/label-bias/hmm_inference.svg" style="width:450px"/>
<div class="caption" markdown="span">
An inference graph for the HMM with the label set `{ARTICLE, NOUN, VERB}`. The
scores for the given observation on each arc are not shown.
</div>
</div>

The scores on each edge associated with an observation are given by $p(x_t
\mid y_t) p(y_t \mid y_{t-1})$. In general the sum of these scores over all
possible next states is not required to be one:
\\[
\sum_{i=1}^c p(x_t \mid y_i) p(y_i \mid y_{t-1}) \ne 1.
\\]
More importantly, the sum is not a constant, but can change depending on the
observation and the previous state. This implies that we do not have
conservation of score mass and hence label bias is not an issue.

### Sequence-to-sequence Models

Sequence-to-sequence models with attention are very commonly used to label
sequences. These models are discriminative and compute the probability of an
output given an input using the chain rule
\\[
    p(Y \mid X) = \prod_{t=1}^T p(y_t \mid y_{\<t}, X)
      \quad \text{where} \quad p(y_t \mid y_{\<t}, X) \propto e^{s(y_t, y_{\<t}, X)}.
\\]
The score function $s(\cdot)$ is computed using a multi-layer neural network
with attention.

These models are locally normalized,
\\[
\sum_{i=1}^c p(y_i \mid y_{\<t}, X) = 1,
\\]
hence they can suffer from label bias. Whether or not this is an issue in
practice remains to be seen. Some attempts have been made to design globally
normalized versions of these models, though none are yet commonly used.[^wiseman16]

### Connectionist Temporal Classification

Connectionist Temporal Classification (CTC) is a discriminative sequence model
designed to solve problems where the correspondence between the input and
output sequence is unknown.[^graves06] This includes problems like speech and
handwriting recognition among others. (See my *Distill* article for an
in-depth tutorial.[^hannun17])

$\gdef\A{\mathcal{A}}$

For a given input-output pair $(X, Y)$, CTC allows a set of alignments
$\A_{X,Y}$. We let $A\\!=\\![a_1, \ldots, a_T] \in \A_{X,Y}$ be one such alignment.
Note, $A$ has the same length as $X$, namely $T$. The probability of a sequence
$Y$ given an input $X$ can then be computed as
\\[
    p(Y \mid X) = \sum_{A \in \A_{X,Y}} \prod_{t=1}^T p(a_t \mid X)
     \quad \text{where} \quad p(a_t \mid X) \propto e^{s(a_t, X)}.
\\]
Like sequence-to-sequence models with attention, the score function $s(\cdot)$
is usually computed with a multi-layer neural network. Notice that CTC assumes
the outputs $a_t$ are conditionally independent of one another given the input
$X$. 

The CTC model is a special case in that it is both locally normalized and
globally normalized. Because of the conditional independence assumption, the
two are equivalent. At the level of an individual alignment, we have
\\[
p(A \mid X) = \prod_{t=1}^T \frac{e^{s_t(a_t, X)}}{\sum_{i=1}^c e^{s_t(a_i, X)}} 
    = \frac{\prod_{t=1}^T e^{s_t(a_t, X)}}{\prod_{t=1}^T \sum_{i=1}^c e^{s_t(a_i, X)}}.
\\]
We can rewrite the denominator using the fact that 
\\[
\prod_{j=1}^m \sum_{i_j=1}^n a_{i_j} = 
    \left(\sum_{i_1=1}^n a_{i_1}\right) \ldots \left(\sum_{i_m=1}^n a_{i_m}\right)
    = \sum_{i_1=1}^m \ldots \sum_{i_m=1}^n \prod_{j=1}^m a_{i_j}
\\]
to get
\\[
p(A \mid X) = \frac{e^{\sum_{t=1}^T s_t(a_t, X)}}{\sum_{A^\prime} e^{\sum_{t=1}^T s_t(a^\prime_t, X)}}.
\\]

Used on its own, CTC does not suffer from label bias. There are a couple of
ways to see this. First, as we described, CTC is globally normalized at the
level of an alignment and label bias results from local normalization. 

Second, the conditional independence assumption made by CTC removes label bias.
If the next state prediction does not depend on any previous state, then there
is no label bias. The model acts as if the transition probabilities $p(y_t \mid
y_{t-1})$ are uniform and the same for all $y_{t-1}$.  This means the entropies
of these distributions are all the same and maximal.

The model does have conservation of score mass in the sense that $\sum_{i=1}^c
p(y_i \mid X) = 1$. However, new evidence can arbitrarily influence the
plausibility of a given path. The model can favor paths which have a certain
label at a given time step by giving the corresponding $p(y_t \mid X)$ a value
close to one. This will in turn make all paths which do not predict $y_t$ have
scores very close to zero. However, the expressiveness of the model is also limited
since it cannot select for paths based on previously predicted labels.

### References

[^mcallum00]:
    Andrew McCallum, Dayne Freitag, and Fernando C.N. Pereira. *Maximum Entropy
    Markov Models for Information Extraction and Segmentation*. ICML 2000.
    [(pdf)](http://www.ai.mit.edu/courses/6.891-nlp/READINGS/maxent.pdf)

[^denker94]:
    John Denker and Christopher Burges. *Image Segmentation and Recognition*, 1994. 
    [(link)](https://www.researchgate.net/publication/240039219_Image_Segmentation_and_Recognition)

[^bottou91]:
    Léon Bottou. *Une Approche théorique de l'Apprentissage Connexionniste:
    Applications à la Reconnaissance de la Parole*, Ph.D. thesis, Université de
    Paris XI, Orsay, France, 1991.
    [(link)](https://leon.bottou.org/papers/bottou-91a)

[^lafferty01]: 
    John Lafferty, Andrew McCallum, and Fernando C.N. Pereira. *Conditional
    Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data*, 2001.
    [(link)](https://repository.upenn.edu/cis_papers/159/)

[^sutton12]:
    Charles Sutton, and Andrew McCallum. *An introduction to conditional random
    fields.* Foundations and Trends® in Machine Learning 4.4 2012: 267-373.
    [(link)](https://www.nowpublishers.com/article/Details/MAL-013)

[^collobert19]:
    Ronan Collobert, Awni Hannun, and Gabriel Synnaeve. *A fully differentiable
    beam search decoder.* ICML 2019. [(link)](https://arxiv.org/abs/1902.06022)

[^bottou97]: 
    Léon Bottou, Yoshua Bengio, and Yann LeCun. *Global training of document
    processing systems using graph transformer networks.* Proceedings of IEEE
    Computer Society Conference on Computer Vision and Pattern Recognition. IEEE, 1997.
    [(link)](https://leon.bottou.org/papers/bottou-97)

[^wiseman16]:
    Sam Wiseman and Alexander M. Rush. *Sequence-to-sequence learning as beam-search optimization.* 2016.
    [(link)](https://arxiv.org/abs/1606.02960)

[^graves06]:
    Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. *Connectionist temporal
    classification: labelling unsegmented sequence data with recurrent neural networks.* ICML, 2006.
    [(link)](https://dl.acm.org/citation.cfm?id=1143891)

[^hannun17]:
    Awni Hannun. *Sequence modeling with CTC.* Distill (2017).
    [(link)](https://distill.pub/2017/ctc)
