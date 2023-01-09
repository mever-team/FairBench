## :brain: Advanced usage
Depending on the modality being assessed, 
you can have different training or test data, 
or you might want to use different predictive models.
For instance, in the above example you can write:

```python
classifier = fb.Fork(case1=LogisticRegression(), case2=LogisticRegression(tol=1.E-6))
```
which makes a different classifier's outcome be assessed 
in each case.

Accessing modal values -if present- is done as class fields,
for example like `yhat = (yhat.case1+yhat.case2)/2`. This 
computation produces a factual tensor that is not
bound to any modality, but if was not there `yhat.case1`
and `yhat.case2` would be used during assessment of
case1 sensitive attribute values and case2 sensitive
attribute values. Here is a visual view of how data 
are organized between branches:

![branches](branches.png)

:bulb: You can use branches to run several computations
pipelines concurrently.