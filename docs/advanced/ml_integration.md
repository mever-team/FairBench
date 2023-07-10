# ML Integration

FairBench integrates smoothly into machine learning workflows
by being able to process their primitives and providing
backpropagate-able computations. This is achieved by running 
[eagerpy](https://eagerpy.jonasrauber.de/) 
under the hood to abstract the operations of different
frameworks with the same interfaces.

1. [Backend selection](#backend-selection)
2. [Buffering batch predictions](#buffering-batch-predictions)

## Backend selection

If you provide primitives of various frameworks (e.g., tensors),
these will be internally converted to FairBench's selected
backend. Switch to a different backend as shown below.

!!! info 
    The `numpy` backend is selected by default.

```python
import fairbench as fb

backend_name = "numpy" # or "torch", "tensorflow", "jax"
fb.setbackend(backend_name)
```

For simple fairness/bias quantification, it does not matter how
computations run internally. However, you may want to 
backpropagate results, for example to
add them to the loss. In this case, you need to set 
internal operations to run on
the same backend as the one that runs your AI.

## Buffering batch predictions

When training machine learning algorithms, you may want
to concatenate the same report arguments generated across 
several batches. You can keep arguments by calling
`fairbench.todict` to convert a set of keyword arguments
to a fork of dictionaries, which reports
automatically unpack internally.
Entries of such forks (e.g.,
`predictions` in the example below) can be concatenated
via a namesake method. iteratively concatenate such dictionaries
with previous concatenation outcomes to generate a final
dictionary of keyword arguments to pass to reports like so:

```python
data = None
for batch in range(batches):
    yhat, y, sensitive = ...  # compute for the batch
    data = fb.concatenate(data, fb.todict(predictions=yhat, labels=y, sensitive=sensitive))
report = fb.multireport(data)
```
