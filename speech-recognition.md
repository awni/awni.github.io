---
layout: post
title: Speech Recognition Is Not Solved 
---

Since Deep Learning hit the scene in speech recognition, word error rates have
fallen dramatically. I'll discuss a few reasons why, despite what you may have
read, we don't yet have human-level speech recognition. Acknowledging these
problems and taking steps towards solving them is critical to progress in ASR.
It's the only way we can go from ASR which works for *some people, most of the
time* to ASR which works for *all people, all the time*.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/speech-recognition/wer.svg" style="width:680px"/>
<div class="caption" markdown="span">
Improvements in word error rate over time on the Switchboard conversational
speech recognition benchmark. The test set was collected in 2000. It consists
of 40 phone conversations between two random native English speakers. The topic
of conversation was pre-suggested by the operator.
</div>
</div>

Recent claims suggest we've achieved human-level parity in conversational
speech recognition. In some ways, this is similar to saying an autonomous car
drives as well as a human after testing it in one town on a sunny day
without traffic. The recent improvements on conversational speech are
astounding. The innovations and improvements are real. But, the claims about
human-level performance are too broad.

## Accents and Noise

One of the most visible deficiencies in speech recognition is dealing with
accents[^scottish_accent] and background noise. The most straight-forward
reason for this is that most of the training data consists of American accented
English with high signal-to-noise ratios. For example, the Switchboard
conversational training and test sets only have native English speakers (mostly
American) with little background noise.

But, more training data likely won't solve this problem on it's own. There are
a lot of languages many with a lot of dialects and accents. It's not feasible
to collect enough annotated data for all of these cases. Building high quality
speech recognizers just for American accented English needs upwards of 10
thousand hours of transcribed audio.

<div class="figure" style="margin-top:20px;margin-bottom:20px">
<img src="{{ site.base_url }}/images/speech-recognition/human_model.svg" style="width:650px"/>
<div class="caption" markdown="span">
Comparison of human transcribers to Baidu's Deep Speech 2 model on various
types of speech.[^data_details] The UK accent group includes accents from the
UK, Australia, New Zealand, Ireland and South Africa. The European accent group
is any accent from continental Europe. Notice the humans are worse at
transcribing the non-American accents. This is probably due to an American bias
in the transcriber pool. I would expect transcribers native to a given region
to have much lower error rates for that regions accents.
</div>
</div>

Because of background noise, it's not uncommon for the SNR in a moving car to
be as low as -5dB. We humans don't have much trouble understanding one another
in these environments. Speech recognizers, on the other hand, degrade more
rapidly with noise. In the figure above we see the gap between humans and the
model error rates increases dramatically between the low SNR and high SNR
audio. 

## Semantic Errors

Often the word error rate is not the actual objective in a speech recognition
system. What we care about is the *semantic error rate*. That's the fraction
of utterances in which we misinterpret the meaning.

We have to be careful when using the word error rate as a proxy. Let me give a
worst-case example to show why. A WER of 5% roughly corresponds to 1 missed
words for every 20. If each sentence has 20 words, the sentence error rate
could be as high as 100%. Hopefully the mistaken words don't change the
semantic meaning of the sentences. Otherwise the recognizer could misinterpret
every sentence even with a 5% WER.

When comparing models to humans, it's important to check the nature of the
mistakes and not just look at the WER as a conclusive number. In my own
experience, human transcribers tend to make fewer and less drastic semantic
errors than speech recognizers. 

Researchers at Microsoft recently compared mistakes made by humans and their
human-level speech recognizer.[^human_comparison] One discrepancy they found
was that the model confuses "uh" with "uh huh" much more frequently than
humans. The two terms have very different semantics: "uh" is just filler
whereas "uh huh" is a *backchannel* acknowledgement. The model and humans also
made a lot of the same types of mistakes.

## Single-channel, Multi-speaker

The Switchboard conversational task is also easier because each speaker has
their own microphone and there are only two speakers. There's no overlap of
multiple speakers in the same audio stream. Humans on the other hand can
converse with multiple speakers at the same time.

A good conversational speech recognizer must be able to segment the audio based
on who is speaking (*diarisation*). It should also be able to make sense of
audio with overlapping speakers (*source separation*). This should be doable
without needing a microphone close to the mouth of each speaker, so
conversational speech can work well in arbitrary locations.

## Domain Variation

Accents and background noise are just two factors a speech
recognizer should be robust to. Here are a few more:

- Reverberation from varying the acoustic environment.
- Artefacts from the hardware.
- The codec used for the audio and compression artefacts.
- The sample rate.
- The age of the speaker.

Most people wouldn't even notice the difference between an `mp3` and a plain
`wav` file. Before we can claim human-level performance, speech recognizers
need to be robust to these sources of variability as well.

## Context

You'll notice the human-level error rate on benchmarks like Switchboard is
actually quite high. If you were conversing with a friend and they
misinterpreted 1 of every 20 words, you'd have a tough time communicating.

One reason for this is that the evaluation is done *context-free*. In real life
we use many other cues to help us understand what someone is saying. Some
examples of context that we use that speech recognizers don't include:

- The history of the conversation and the topic being discussed.
- Visual cues of the person speaking including facial expressions and lip movement.
- Prior knowledge about the person we are speaking with.

For example, Android's speech recognizer has knowledge of your contact list so
it can recognize your friends names.[^contacts] The voice search in maps
products can use geolocation to narrow down the possible points-of-interest
you might be asking to navigate to.[^geo_location]

The accuracy of ASR systems definitely improves when incorporating this type of
signal. We've just begun to scratch the surface on the type of context we can
include and how it's used.

## Deployment

The recent improvements in conversational speech are not deployable. When
thinking about what makes a new speech algorithm deployable, it's helpful to
consider both latency and compute. The two are definitely related. Algorithms
which increase compute tend to increase latency. But for simplicity I'll
discuss each separately.

**Latency**: With latency, I mean the time from when the user is done
speaking to when the transcription is complete. Low latency is a common product
constraint in ASR. It can significantly impact the user experience. Latencies
requirements in the tens of milliseconds aren't uncommon for ASR systems. While
may sound extreme, remember that the producing the transcript is usually the
first step a series of expensive computations. For example in voice search the
actual search has to be done after the speech recognition.

Bidirectional recurrent layers are a good example of a latency killing
improvement. All the recent state-of-the-art results in conversational speech
use them. The problem is we can't compute anything after the first
bidirectional layer until the user is done speaking. So the latency scales with
the length of the utterance.

<div class="figure" style="margin-bottom:30px;">
<img src="{{ site.base_url }}/images/speech-recognition/forward_only.svg" style="padding-right:30px;width:370px"/>
<img src="{{ site.base_url }}/images/speech-recognition/bidirectional.svg" style="width:340px"/>
<div class="caption" markdown="span" style="margin-top:10px">
**Left:** With a forward only recurrence we can start computing the
transcription immediately. **Right:** With a bidirectional recurrence we have to
wait until all the speech arrives before beginning to compute the
transcription.
</div>
</div>

A good way to efficiently incorporate future information in speech
recognition is still an open problem.

**Compute**: The amount of computational power needed to transcribe an
utterance is an economic constraint. We have to consider the *bang-for-buck* of
every accuracy improvement to a speech recognizer. If an improvement doesn't
meet our threshold, then it can't be deployed.

A classic example of consistent improvement that never gets deployed is an
ensemble. The 1% or 2% decrease in error is rarely worth the 2-8x increase in
compute. Big RNN language models are also usually in this category since they
really slow down the beam search. I expect this will change in the future.

As a caveat, I'm not suggesting research which improves accuracy at great
computational cost isn't useful. We've seen the pattern of "first slow but
accurate, then fast" work well before. The point is just that until an
improvement is sufficiently fast, it's not usable.

## The Next Five Years

Claims like "human-level performance on conversational speech" aren't
necessarily good for the field. Aspiring researchers looking to innovate on
hard problems and have a big impact may think all the problems in speech
recognition have been solved. And they will focus their attention elsewhere.  

There are still a lot of open and challenging problems in speech understanding.
These include:
- Broadening the capabilities to new domains, accents and far-field, low SNR speech.
- Incorporating more context into the recognition process.
- Diarisation and source-separation.
- Semantic error rates and innovative methods of evaluating recognizers.
- Super low-latency and efficient inference.

I look forward to the next five years of progress on these and other fronts.

### Footnotes
[^scottish_accent]:
    Just ask anyone with a [Scottish accent].

[^data_details]:
    These results are from [Amodei et al, 2016](https://arxiv.org/abs/1512.02595).
    The accented speech comes from [VoxForge]. The noise-free and noisy speech
    comes from the third [CHiME] challenge.

[^contacts]: 
    See [Aleksic et al., 2015](http://ieeexplore.ieee.org/document/7178957/)
    for an example of how to improve contact name recognition.

[^geo_location]:
    See [Chelba et al., 2015](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43817.pdf)
    for an example of how to incorporate location.

[^human_comparison]:
    [Stolcke and Droppo, 2017](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/paper-revised2.pdf)

[Scottish accent]: https://www.youtube.com/watch?v=5FFRoYhTJQQ
[VoxForge]: http://www.voxforge.org/
[CHiME]: http://ieeexplore.ieee.org/document/7404837/
