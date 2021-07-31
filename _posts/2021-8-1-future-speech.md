---
layout: post
title: The History of Speech Recognition to the Year 2030
katex: True
---

Given the remarkable changes in the state of speech recognition over the
previous decade, what can we expect over the coming decade? I attempt to
forecast the state of speech recognition research and applications by the year
2030.

---

A longer version of this article with more technical details on the research
and more comprehensive references is available as a [PDF](www.google.com).

---

## Recap

The decade from 2010 to 2020 saw remarkable progress in speech recognition and
related technology. The figure below shows a timeline of some of the
major developments in the research, software, and application of speech
recognition over the previous decade. The decade saw the launch and spread of
phone-based voice assistants like Apple Siri. Far-field devices like Amazon
Alexa and Google Home were also released and proliferated.

<div class="figure">
<img src="{{ site.base_url }}/images/future-speech/asr_timeline.svg" style="width:700px"/>
<div class="caption" markdown="span">
A timeline of some of the major developments in speech recognition from the
years 2010 to 2020. The decade saw the launch of voice-based devices and voice
assistants, open-source and widely used speech recognition software like Kaldi,
and larger benchmarks like LibriSpeech. We also saw speech recognition models
improve starting from hybrid neural network architectures to more end-to-end
models including Deep Speech, Deep Speech 2, encoder-decoder models with
attention, and transducer-based speech recognition.
</div>
</div>

These technologies were enabled in-part by the remarkable improvement in the
word error rates of automatic speech recognition as a result of the rise of
deep learning. The key drivers of the success of deep learning in speech
recognition have been 1) the curation of massive transcribed data sets, 2) the
rapid rate of progress in graphics processing units, and 3) the improvement in
the learning algorithms and model architectures.

Thanks to these ingredients, the word error rate of speech recognizers improved
consistently and substantially throughout the decade. On two of the most
commonly studied benchmarks, automatic speech recognition word error rates have
surpassed those of professional transcribers.

This remarkable progress invites the question: what is left for the coming
decade to the year 2030? In the following, I attempt to answer this question.
But, before I begin, I'd first like to share some observations on the general
problem of predicting the future. These findings are inspired by the
mathematician (as well as computer scientist and electrical engineer) Richard
Hamming, who also happened to be particularly adept at forecasting the future
of computing.

## Predicting the Future

Richard Hamming in *The Art of Doing Science and Engineering* makes many
predictions, many of which have come to pass. Here are a few
examples:[^hamming]

- He stated that by "the year 2020 it would be fairly universal practice for
  the expert in the field of application to do the actual program preparation
  rather than have experts in computers (and ignorant of the field of
  application) do the program preparation."

- He predicted that neural networks "represent a solution to the
  programming problem," and that "they will probably play a large part in
  the future of computers."

- He predicted the prevalence of general-purpose rather than special-purpose
  hardware, digital over analog, and high-level programming languages all long
  before the field had decided one way or another.

- He anticipated the use of fiber-optic cables in place of copper wire for
  communication well before the switch actually took place.

These are just a few examples of Hamming's extraordinary prescience. Why was he
so good at predicting the future? Here are a few observations which I think
were key to his ability:

**Practice:** One doesn't get good at predicting the future without actually
practicing at it. Hamming practiced. He reserved every Friday afternoon "great
thoughts" in which he mused on the future. But he didn't just predict in
isolation. He made his predictions public, which forced him to put them in a
communicable form and held him accountable. For example, in 1960 Hamming gave a
talk titled "The History of Computing to the Year 2000" (you may recognize
the title) in which he predicted how computing would evolve over the next
several decades.

**Focus on fundamentals:** In some ways, forecasting the future development
of technology is just about understanding the present state of technology more
than those around you. This requires both depth in one field as well as
non-trivial breadth. This also requires the ability to rapidly assimilate new
knowledge. Mastering the fundamentals builds a strong foundation for both.

**Open mind:** Probably the most important trait Hamming exhibited, and in my
opinion the most difficult to learn, was his ability to keep an open mind.
Keeping an open mind requires constant self-vigilance. Having an open mind one
day does not guarantee having it the next. Having an open mind with respect to
one scientific field does not guarantee having it with respect to another.
Hamming recognized for example that one may be more productive in an office
with the door closed, but he kept his office door open as he believed an "open
mind leads to the open door, and the open door tends to lead to the open
mind."

I'll add to these a few more thoughts. First, the rate of change of progress in
computing and machine learning is increasing. This makes it harder to predict
the distant future today than it was 50 or 100 years ago. These days predicting
the evolution of speech recognition even ten years out strikes me as a
challenge. Hence my choosing to work with that time frame.

A common saying about technology forecasting is that short-term predictions
tend to be overly optimistic and long-term predictions tend to be overly
pessimistic. This is often attributed to the fact that progress in technology
has been exponential. The graph below shows how this can happen if we
optimistically extrapolate from the present assuming progress is linear with
time. Progress in speech recognition over the previous decade (2010-2020) was
driven by exponential growth along two key axes. These were compute (e.g.
floating-point operations per second) and data set sizes. Whether or not such
exponential growth will continue to apply to speech recognition for the coming
decade remains to be seen.

<div class="figure">
<img src="{{ site.base_url }}/images/future-speech/exponential_growth.svg" style="width:400px"/>
<div class="caption" markdown="span">
The graph depicts progress as a function of time. The linear extrapolation from
the present (dashed line) initially results in overly optimistic predictions.
However, eventually the predictions become pessimistic as they are outstripped
by the exponential growth (solid line).
</div>
</div>

I'm sure a lot of the following predictions will prove wrong. In some ways,
particularly when it comes to the more controversial predictions, these are
really more of an optimistic wishlist for the future. On that note, let me
close this section with the famous words of the computer scientist Alan
Kay:[^kay]

> The best way to predict the future is to invent it.

## Research Predictions

### Semi-supervised Learning

**Prediction:** Semi-supervised learning is here to stay. In particular,
self-supervised pretrained models will be a part of many machine-learning
applications, including speech recognition.

Part of my job as a research scientist is hiring, which means a lot of
interviews. I've interviewed more than a hundred candidates working on a
diverse array of machine-learning applications. Some large fraction,
particularly of the natural language applications, rely on a pretrained model
as the basis for their machine-learning enabled product or feature.
Self-supervised pretraining is already pervasive in language applications in
industry. I predict that by 2030 self-supervised pretraining will be just as
pervasive in speech recognition.

The main challenges with self-supervision are those of scale, and hence
accessibility. Right now only the most highly endowed industry research labs
*e.g.* Google Brain, Google DeepMind, Facebook AI Research, OpenAI, *etc.*)
have the funds to burn on the compute required to research self-supervision at
scale. As a research direction, self-supervision is only becoming less
accessible to academia and smaller industry labs.

**Research implications:** Self-supervised learning would be more accessible
given lighter-weight models which could be trained efficiently on less data.
Research directions which could lead to this include sparsity for
lighter-weight models, optimization for faster training, and effective ways of
incorporating prior knowledge for sample efficiency.

### On Device

**Prediction:** Most speech recognition will happen on the device or at the
edge.

There are a few reasons I predict this will happen. First, keeping your data on
your device rather than sending it to a central server is more private. The
trend towards data privacy will encourage on-device inference whenever
possible. If the model needs to learn from a user's data, then the training
should happen on the device.

The second reason to prefer on-device inference is latency. In absolute terms,
the difference between 10 milliseconds and 100 milliseconds is not much. But
the former is well below the perceptual latency of a human, and the latter well
above. Google has already demonstrated an on-device speech recognition system
with accuracy nearly as good as a server-side system. The latency differences
are easily noticeable.[^google_on_device] From a pragmatic standpoint, the
latency of the server-side recognizer is probably sufficient. However, the
imperceptible latency of the on-device system makes the interaction with the
device feel much more responsive and hence more engaging.

A final reason to prefer on-device inference is 100% availability. Having the
recognizer work even without an internet connection or in spotty service means
it will work all the time. From a user interaction standpoint there is a big
difference between a product which works most of the time and a product which
works every time.

**Research implications:** On-device recognition requires models with smaller
compute and memory requirements and which use less energy in order to preserve
battery life. Model quantization and knowledge distillation (training a smaller
model to mimic the predictions of a more accurate larger model) are two
commonly used techniques. Sparsity, which is less commonly used, is another
approach to generate lighter-weight models. In sparse models, most of the
parameters (*i.e.* connections between hidden states) are zero and can be
effectively ignored. Of these three techniques, I think sparsity is the
most promising research direction.

Weak supervision will be an important research direction for on-device training
for applications which typically require labeled data. For example, a users
interaction with the output of a speech recognizer or the actions they take
immediately afterward could be useful signal from which the model can learn in
a weakly-supervised manner.

### Word Error Rate

**Prediction:** By the end of the decade, possibly much sooner, researchers
will no longer be publishing papers which amount to "improved word error rate
on benchmark X with model architecture Y."

As you can see in graphs below, word error rates on the two most commonly
studied speech recognition benchmarks have saturated. Part of the problem is
that we need harder benchmarks for researchers to study.

<div class="figure">
<img src="{{ site.base_url }}/images/future-speech/librispeech_wer.svg" style="padding-right:30px;width:370px"/>
<img src="{{ site.base_url }}/images/future-speech/switchboard_wer.svg" style="width:340px"/>
<div class="caption" markdown="span">
The improvement in word error rate over time on the LibriSpeech (left) and
Switchboard Hub5'00 (right) benchmarks.[^wer_are_we] The dashed lines indicate
human-level performance.
</div>
</div>

Another part of the problem is that we have reached a regime where word error
rate improvements on academic benchmarks no longer correlate with practical
value. Speech recognition word error rates on both benchmarks surpassed human
word error rates several years ago. However, in most settings humans understand
speech better than machines do. This implies that word error rate as a measure
of the quality our speech recognition systems does not correlate well with an
ability to understand human speech.

A final issue is research in state-of-the-art speech recognition is becoming
less accessible as models and data sets are getting larger, and as computing
costs are increasing. A few well-funded industry labs are rapidly becoming the
only places that can afford this type of research. As the advances become more
incremental and further from academia, this part of the field will continue to
shift from research labs to engineering organizations.

### Richer Representations

<div class="figure">
<img src="{{ site.base_url }}/images/future-speech/lattice.svg" style="width:450px"/>
<div class="caption" markdown="span">
An example lattice used to encode mutliple hypotheses output from a speech
recognizer with differing weights.
</div>
</div>

**Prediction:** Transcriptions will be replaced by richer representations for
downstream tasks which rely on the output of a speech recognizer. Examples of
such downstream applications include conversational agents, voice-based search
queries, and digital assistants.

Downstream applications often don't care about a verbatim transcription; they
care about semantic correctness. Hence, improving the word error rate of a
speech recognizer often does not improve the objective of the downstream task.
One possibility is to develop a *semantic error rate* and use it to measure the
quality of the speech recognizer. This is a challenging albeit interesting
research problem.

I think a more likely outcome is to give downstream applications richer
representations from the speech recognizer. For example, instead of passing a
single transcription, passing a lattice of possibilities (as in graph below)
which captures the uncertainty for each could be much more useful.

**Research implications:** The exact structure used to encode the
representation is an open question. One possibility could be some sort of
weighted transducer which if differentiable could allow for fine-tuning the
recognizer to specific applications. This type of representation also requires
models which are able to ingest variable-sized graphs as input.

### Personalization

**Prediction:** By the end of the decade, speech recognition models will be
deeply personalized to individual users.

One of the main distinctions between the automatic recognition of speech and
the human interpretation of speech is in the use of context. Humans rely on a
lot of context when speaking to one another. This context includes the topic
of conversation, what was said in the past, the noise background, and visual
cues like lip movement and facial expressions. We have, or will soon
reach, the Bayes error rate for speech recognition on short (*i.e.* less
than ten second long) utterances taken out of context. Our models are using the
signal available in the data to the best of their ability. To continue to
improve the machine understanding of human speech will require leveraging
context as a deeper part of the recognition process.

One way to do this is with personalization. Personalization is already used to
improve the recognition of utterances of the form "call `<NAME>`".  As another
example, personalizing models to individual users with speech disorders
improves word error rates by 64% relative.[^sim2019] Personalization can make
a huge difference in the quality of the recognition, particularly for groups or
domains that are underrepresented in the training data. I predict we will see
much more pervasive personalization by the end of the decade.

**Research implications:** On-device personalization requires on-device
training which in itself requires lighter-weight models and some form of weak
supervision. Personalization requires models which can be easily customized to
a given user or context. The best way to incorporate such context into a model
is still a research question.

## Application Predictions

### Transcription Services

**Prediction:** By the end of the decade, 99% of transcribed speech services
will be done by automatic speech recognition. Human transcribers will perform
quality control and correct or transcribe the more difficult utterances.
Transcription services include, for example, captioning video, transcribing
interviews, and transcribing lectures or speeches.

### Voice Assistants

**Prediction:** Voice assistants will get better, but incrementally, not
fundamentally. Speech recognition is no longer the bottleneck to better voice
assistants. The bottlenecks are now fully in the language understanding domain
including the ability to maintain conversations, multi-ply contextual
responses, and much wider domain question and answering. We will continue to
make incremental progress on these so-called AI-complete
problems,[^shapiro]
but I don't expect them to be solved by 2030.

Will we live in smart homes that are always listening and can respond to our
every vocal beck and call? No. Will we wear augmented reality glasses on our
faces and control them with our voice? Not by 2030.


## Summary

Here is a summary of some of my predictions for the progress in speech
recognition research and applications by the year 2030:

- Self-supervised learning and pretrained models are here to stay.
- Most speech recognition (inference) will happen at the edge.
- On-device model training will be much more common.
- Sparsity will be a key research direction to enable on-device inference and training.
- Improving word error rate on common benchmarks will fizzle out as a research goal.
- Speech recognizers will output richer representations (graphs) for use by downstream tasks.
- Personalized models will be commonplace.
- Most transcription services will be automated.
- Voice assistants will continue to improve, but incrementally.

The predictions show that the coming decade could be just as exciting and
important to the development of speech recognition and spoken language
understanding as the previous one. We still have many research problems to
solve before speech recognition will reach the point where it works all the
time, for everyone. However, this is a goal worth working toward, as speech
recognition is a key component to more fluid, natural, and accessible
interactions with technology.

### Acknowledgements

Thanks to Chris Lengerich, Marya Hannun, Sam Cooper, and Yusuf Hannun for their
feedback.

### Footnotes

[^hamming]:
    Quotes and predictions are from chapters 2, 4, and 21 of Richard Hamming's
    *The Art of Doing Science and Engineering: Learning to Learn*.

[^kay]:
    Alan Kay is best known for developing the modern graphical user interface
    and also object-oriented programming in the Smalltalk programming langauge.

[^google_on_device]:
    For an example of the perceptual difference in latencies see the blog post
    on [Google's on-device speech recognizer](https://ai.googleblog.com/2019/03/an-all-neural-on-device-speech.html).

[^wer_are_we]:
    The data for these figures is from [wer_are_we](https://github.com/syhw/wer_are_we).

[^sim2019]:
    Khe Chai Sim, *et al.* *Personalization of end-to-end speech recognition on
    mobile devices for named entities.* 2019. [(link)](https://arxiv.org/abs/1912.09251)

[^shapiro]:
    Shapiro in the *Encyclopedia of Artificial Intelligence* defines an AI task as
    AI-complete if solving it is equivalent to "solving the general AI problem",
    which he defines as "producing a generally intelligent computer program".
