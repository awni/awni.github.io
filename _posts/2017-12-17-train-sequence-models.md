---
layout: post
title: Training Sequence Models with Attention
katex: True
---

Here are a few practical tips for training sequence-to-sequence models with
attention. If you have experience training other types of deep neural networks,
pretty much all of it applies here. This article focuses on a few tips you
might not know about, even with experience training other models.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/train-seq2seq/seq2seq.svg" style="width:500px"/>
<div class="caption" markdown="span">
The architecture of a sequence-to-sequence model with attention. On the left is
the encoder network. On the right is the decoder network predicting an output
sequence. The hidden states of the encoder are attended to at each time-step of
the decoder.
</div>
</div>

## Learning to Condition 

The first thing we want to know is if the model is even working. Sometimes,
it's not so obvious. With sequence-to-sequence models, we typically optimize
the conditional probability of the output given the input,

\\[
p(Y \mid X) \; = \; \prod_{u=1}^U \; p(y_u \mid y\_{\lt u}, X).
\\]

Here $X = [x_1, \ldots, x_T]$ is the input sequence and $Y = [y_1, \ldots,
y_U]$ is the output sequence. The input $X$ is encoded into a sequence of
hidden states. The decoder network then incorporates information from these
hidden states via the attention mechanism.

One failure mode for a sequence-to-sequence model is it never learns to
condition on the input $X$. In other words, the model doesn't learn how to
attend to the encoded input in a useful way. When this happens, the model is in
effect optimizing

\\[
p(Y) \; = \; \prod_{u=1}^U \; p(y_u \mid y\_{\lt u}).
\\]

This is just a language model over the output sequences. Reasonable learning
can actually happen in this case even if the model never learns to condition on
$X$. That's one reason it's not always obvious if the model is truly working.

**Visualize Attention**: This brings us to our first tip. A great way to tell
if the model has learned to condition on the input is to visualize the
attention. Usually it's pretty clear if the attention looks reasonable.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/train-seq2seq/attention.svg" style="width:550px"/>
<div class="caption" markdown="span">
An example of the attention learned by two different models for a speech
recognition task. **Top:** A reasonable looking "alignment" between the input
and the output. **Bottom:** The model failed to learn how to attend to the
input even though the training loss was slowly reduced over time (the loss
didn't diverge).
</div>
</div>

I recommend setting up your model so that it's easy to extract the attention
vectors as an early debugging step. Make a Jupyter notebook or some other
simple method to load examples and visualize the attention.

## The Inference Gap

Sequence-to-sequence models are trained with *teacher forcing*. Instead of
using the predicted output as the input at the next step, the ground truth
output is used. Without teacher forcing these models are much slower to
converge, if they do so at all.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/train-seq2seq/seq2seq_teacher_forcing.svg" style="width:500px"/>
<div class="caption" markdown="span">
Sequence-to-sequence models are trained with *teacher forcing*. The input to
the decoder is the ground-truth output instead of the prediction from the
previous time-step. 
</div>
</div>

Teacher forcing causes a mismatch between training the model and using it for
inference. During training we always know the previous ground truth but not during 
inference. Because of this, it's common to see a large gap between error
rates on a held-out set evaluated with teacher forcing versus true inference.

**Scheduled Sampling:** A helpful technique to bridge the gap between training
and inference is scheduled sampling.[^scheduled_sampling] The idea is simple --
select the previous predicted output instead of the ground truth output with
probability $p$. The probability should be tuned for the problem. A typical
range for $p$ is between 10% and 40%. 

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/train-seq2seq/scheduled_sampling.svg" style="width:300px"/>
<div class="caption" markdown="span">
Scheduled sampling randomly chooses whether to use the predicted output or the
ground truth output as the input to the next time-step.
</div>
</div>

As a quick side note, the performance gap between teacher forcing at training
time and prediction at test time is still an active area of research. Scheduled
sampling works fairly well in practice; however, it has some undesirable
properties[^improper_ss] and alternative approaches have been
proposed.[^professor_forcing]

**Tune with Inference Rates:** There can be a big gap between the teacher
forced loss and the error rate when properly inferring the output. Also,
the correlation between the two metrics may not be perfect. Because of this, I
recommend performing model selection and hyper-parameter tuning based on the
inferred output error rates. If you save the model which performs best on a
development set during training, use this error rate as a performance
measure.

For example in speech recognition tune directly with the word (or character)
error rate computed on the predicted output. In machine translation, text
summarization and other tasks where many correct output sentences exist, use
the [BLEU] or [ROUGE] score.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/train-seq2seq/loss_cer.svg" style="width:500px"/>
<div class="caption" markdown="span">
The blue curve is the teacher forced cross entropy loss. The red curve is the
phoneme error rate (PER) of the inferred output (with a beam-size of 1). The
best achieved PER is quite a bit better than the PER corresponding to the best
teacher forced cross entropy loss and occurs much later in training. There is
quite a bit of variance in these curves so take this example with a grain of
salt. Even so, the two curves do look quite different which tells us that the
metrics aren't well correlated after a point.[^asr_experiments]
</div>
</div>

This tip is perhaps more important on smaller datasets when there is likely
more variance in the two metrics. However, in these cases it can make a big
difference. For example on the phoneme recognition task above we see a 13%
relative improvement by taking the model with the best inferred error rate
instead of the best teacher forced loss. This can be a key difference if you're
trying to reproduce a baseline.

## Efficiency 

One downside to using these models is that they can be quite slow. The
attention computation scales as the product of the input and output sequence
lengths, e.g. $O(TU)$. If the input sequence doubles in length and the output
sequence doubles in length the amount of computation quadruples.


**Bucket by Length:** When optimizing a model with a minibatch size greater
than 1, make sure to bucket the examples by length. For each batch, we'd like
the inputs to all be the same length and the outputs to all be the same length.
This won't usually be possible, but we can at least attempt to minimize the
largest length mismatch in any given batch.

One heuristic that works pretty well is to make buckets based on the input
lengths. For example, all the inputs with lengths 1 to 5 go in the first
bucket. Inputs with lengths 6 to 10 go in the second bucket and so on. Then
sort the examples in each bucket by the output length followed by the input
length.

Naturally, the larger the training set the more likely you are to have
minibatches with inputs and outputs that are mostly the same length.

**Striding and Subsampling:** When the input and output sequences are long
these models can grind to a halt. With long input sequences, a good practice is
to reduce the encoded sequence length by subsampling. This is common in speech
recognition, for example, where the input can have thousands of
time-steps.[^speech_recognition] You won't see it as much in word-based machine
translation since the input sequences aren't as long. However, with character
based models subsampling is more common.[^language_correction] The subsampling
can be implemented with a strided convolution and/or pooling operation or
simply by concatenating consecutive hidden states.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/train-seq2seq/subsampling.svg" style="width:500px"/>
<div class="caption" markdown="span">
A pyramidal structure in the encoder. Here the stride or subsampling factor is
2 in each layer. The number of time-steps in the input sequence is reduced by a
factor of 4 (give or take depending on how you pad the sequence to each layer).
</div>
</div>

Often subsampling the input doesn't reduce the accuracy of the model. Even with
a minor hit to accuracy though, the speedup in training time can be worth it.
When the RNN and attention computations are the bottleneck (which they usually
are), subsampling the input by a factor of 4 can make training the model 4
times faster. 

## That's All

As you can see, getting these models to work well requires the right basket of
tools. These tips are by no means comprehensive, my aim here is more for
precision over recall. Even so, they certainly won't generalize to every
problem. But, as a few first ideas to try when training and improving a
baseline sequence-to-sequence model, I strongly recommend all of them.

### Acknowledgements

Thanks to [Ziang Xie](https://twitter.com/ziangx1e) for useful feedback and
suggestions.

### Footnotes

[^scheduled_sampling]: See [Bengio et al., 2015](https://arxiv.org/abs/1506.03099)

[^improper_ss]:
    See [Huszar, 2015](https://arxiv.org/abs/1511.05101) for an explanation why
    scheduled sampling is improper and a suggested alternative.

[^professor_forcing]:
    [Lamb et al., 2016](https://arxiv.org/abs/1610.09038) introduce
    Professor Forcing, an alternative to scheduled sampling.

[^language_correction]:
    See [Xie et al., 2016](https://arxiv.org/abs/1603.09727) for an example of
    a character-based model for language correction which subsamples in the
    encoder.

[^speech_recognition]:
    See [Chan et al., 2015](https://arxiv.org/abs/1508.01211) for an example
    of this in speech recognition.

[^asr_experiments]: 
    Here's the [code](https://github.com/awni/speech/tree/master/examples/timit)
    for more details or if you want to reproduce this experiment.

[BLEU]: https://en.wikipedia.org/wiki/BLEU
[ROUGE]: https://en.wikipedia.org/wiki/ROUGE_(metric)
