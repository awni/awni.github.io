---
layout: post
title: PyTorch or TensorFlow?
---

This is a guide to the main differences I've found between
[PyTorch](http://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/).
This post is intended to be useful for anyone considering starting a new
project or making the switch from one deep learning framework to another.  The
focus is on programmability and flexibility when setting up the components of the
training and deployment deep learning stack. I won't go into performance (speed
/ memory usage) trade-offs.

### Summary 

PyTorch is better for rapid prototyping in research, for hobbyists and for
small scale projects. TensorFlow is better for large-scale deployments,
especially when cross-platform and embedded deployment is a consideration.

### Ramp-up Time 
*Winner: PyTorch*

PyTorch is essentially a GPU enabled drop-in replacement for NumPy equipped
with higher-level functionality for building and training deep neural networks.
This makes PyTorch especially easy to learn if you are familiar with NumPy,
Python and the usual deep learning abstractions (convolutional layers,
recurrent layers, SGD, etc.).

On the other hand, a good mental model for TensorFlow is a programming language
embedded within Python. When you write TensorFlow code it gets "compiled" into
a graph by Python and then run by the TensorFlow execution engine. I've seen
newcomers to TensorFlow struggle to wrap their head around this added layer of
indirection. Also because of this, TensorFlow has a few extra concepts to
learn such as the session, the graph, variable scoping and placeholders. Also
more boilerplate code is needed to get a basic model running. The ramp-up time
to get going with TensorFlow is definitely longer than PyTorch. 

### Graph Creation and Debugging
*Winner: PyTorch*

Creating and running the computation graph is perhaps where the two frameworks
differ the most. In PyTorch the graph construction is dynamic, meaning the
graph is built at run-time.  In TensorFlow the graph construction is static,
meaning the graph is "compiled" and then run. As a simple example, in PyTorch
you can write a for loop construction using standard Python syntax
```
for _ in range(T):
    h = torch.matmul(W, h) + b
```
and `T` can change between executions of this code. In TensorFlow this requires
the use of [control flow
operations](https://www.tensorflow.org/api_guides/python/control_flow_ops#Control_Flow_Operations)
in constructing the graph such as the `tf.while_loop`. TensorFlow does have the
`dynamic_rnn` for the more common constructs but creating custom dynamic
computations is more difficult.

The simple graph construction in PyTorch is easier to reason about, but perhaps
even more importantly, it's easier to debug. Debugging PyTorch code is just
like debugging Python code. You can use `pdb` and set a break point anywhere.
Debugging TensorFlow code is not so easy. The two options are to request the
variables you want to inspect from the session or to learn and use the
TensorFlow debugger (tfdbg).

### Coverage
*Winner: TensorFlow*

As PyTorch ages, I expect the gap here will converge to zero. However, there is
still some functionality which TensorFlow supports that PyTorch doesn't. A few
features that PyTorch doesn't have (at the time of writing) are:
- Flipping a tensor along a dimension (`np.flip`, `np.flipud`, `np.fliplr`)
- Checking a tensor for NaN and infinity (`np.is_nan`, `np.is_inf`)
- Fast Fourier transforms (`np.fft`)

These are all supported in TensorFlow. Also the TensorFlow `contrib` package
has many more higher level functions and models than PyTorch. 

### Serialization
*Winner: TensorFlow*

Saving and loading models is simple in both frameworks. PyTorch has an
especially simple API which can either save all the weights of a model or
pickle the entire class. The TensorFlow `Saver` object is also easy to use
and exposes a few more options for check-pointing. 

The main advantage TensorFlow has in serialization is that the entire graph can
be saved as a protocol buffer. This includes parameters as well as operations.
The graph can then be loaded in other supported languages (C++, Java). This is
critical for deployment stacks where Python is not an option. Also this can, in
theory, be useful when you change the model source code but want to be able to
run old models.

### Deployment
*Winner: TensorFlow*

For small scale server-side deployments both frameworks are easy to wrap
in e.g. a Flask web server.

For mobile and embedded deployments TensorFlow
[works](https://www.tensorflow.org/mobile/). This is more than can be said of
most other deep learning frame-works including PyTorch. Deploying to Android or
iOS does require a non-trivial amount of work in TensorFlow but at least you
don't have to rewrite the entire inference portion of your model in Java or
C++.

For high-performance server-side deployments there is [TensorFlow
Serving](https://www.tensorflow.org/serving/). I don't have experience with
TensorFlow Serving, so I can't write confidently about the pros and cons. For
heavily used machine learning services, I suspect TensorFlow Serving could be a
sufficient reason to stay with TensorFlow. Other than performance, one of the
noticeable features of TensorFlow Serving is that models can be hot-swapped
easily without bringing the service down. Checkout this [blog
post](https://medium.com/zendesk-engineering/how-zendesk-serves-tensorflow-models-in-production-751ee22f0f4b)
from Zendesk for an example deployment of a QA bot with TensorFlow serving.

### Documentation
*Winner: Tie*

I've found everything I need in the docs for both frameworks. The Python APIs
are well documented and there are enough examples and tutorials to learn either
framework.

One edge case gripe is that the PyTorch C library is mostly undocumented.
However, this really only matters when writing a custom C extension and perhaps
if contributing to the software.

### Data Loading
*Winner: PyTorch*

The APIs for data loading are well designed in PyTorch. The interfaces are
specified in a dataset, a sampler, and a data loader. A data loader takes a
dataset and a sampler and produces an iterator over the dataset according to
the sampler's schedule. Parallelizing data loading is as simple as passing a
`num_workers` argument to the data loader. 

I haven't found the tools for data loading in TensorFlow (readers, queues,
queue runners, etc.) especially useful. In part this is because adding all the
preprocessing code you want to run in parallel into the TensorFlow graph is not
always straight-forward (e.g. computing a spectrogram). Also, the API itself is
more verbose and harder to learn.

### Device Management
*Winner: TensorFlow*

Device management in TensorFlow is about as seamless as it gets. Usually you
don't need to specify anything since the defaults are set well. For example,
TensorFlow assumes you want to run on the GPU if one is available. In PyTorch
you have to explicitly move everything onto the device even if CUDA is enabled. 

The only downside with TensorFlow device management is that by default it
consumes all the memory on all available GPUs even if only one is being used.
The simple workaround is to specify `CUDA_VISIBLE_DEVICES`. Sometimes people
forget this, and GPUs can appear to be busy when they are in fact idle.

In PyTorch, I've found my code needs more frequent checks for CUDA availability
and more explicit device management. This is especially the case when writing
code that should be able to run on both the CPU and GPU. Also converting say a
PyTorch Variable on the GPU into a NumPy array is somewhat verbose.
```
numpy_var = variable.cpu().data.numpy()
```

### Custom Extensions
*Winner: PyTorch*

Building or binding custom extensions written in C, C++ or CUDA is doable with
both frameworks. TensorFlow again requires more boiler plate code though is
arguably cleaner for supporting multiple types and devices. In PyTorch you
simply write an interface and corresponding implementation for each of the CPU
and GPU versions. Compiling the extension is also straight-forward with both
frameworks and doesn't require downloading any headers or source code outside
of what's included with the pip installation.

### A note on TensorBoard

TensorBoard is a tool for visualizing various aspects of training machine
learning models. It's one of the most useful features to come out of the
TensorFlow project. With a few code snippets in a training script you can view
training curves and validation results of any model. TensorBoard runs as a web
service which is especially convenient for visualizing results stored on
headless nodes.

This was one feature that I made sure I could keep (or find an alternative to)
before using PyTorch. Thankfully there are, at least, two open-source projects
which allow for this. The first is
[tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger) and
the second is [crayon](https://github.com/torrvision/crayon). The
tensorboard_logger library is even easier to use than TensorBoard "summaries"
in  TensorFlow, though you need TensorBoard installed to use it. The crayon
project is a complete replacement for TensorBoard but requires more setup
(docker is a prerequisite).

### A note on Keras

[Keras](https://keras.io/) is a higher-level API with a configurable back-end.
At the moment TensorFlow, Theano and CNTK are supported, though perhaps in the
not too distant future PyTorch will be included as well. Keras is also
distributed with TensorFlow as a part of `tf.contrib`.

Though I didn't discuss Keras above, the API is especially easy to use. It's
one of the fastest ways to get running with many of the more commonly used deep
neural network architectures. That said, the API is not as flexible as PyTorch
or core TensorFlow.

### A note on TensorFlow Fold 

Google announced [TensorFlow
Fold](https://research.googleblog.com/2017/02/announcing-tensorflow-fold-deep.html)
in February of 2017. The library is built on top of TensorFlow and allows for
more dynamic graph construction. The main advantage of the library appears to
be the dynamic batching. Dynamic batching automatically batches computations on
inputs of varying size (think recursive networks on parse trees). In terms of
programmability, the syntax is not as straightforward as PyTorch, though in a
some cases the performance improvements from batching may be worth the cost. 

