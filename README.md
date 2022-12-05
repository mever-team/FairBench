# FairBench
Bringing together existing and new frameworks
for fairness exploration.

**This project is in its pre-alpha phase.**

## :rocket: Quickstart

```python
import fairbench as fb
from sklearn.linear_model import LogisticRegression

x_train, y_train, x_test, y_test, s_train, s_test = ...

x = fb.Modal(train=fb.array(x_train), test=fb.array(x_test))
y = fb.Modal(train=fb.array(y_train), test=fb.array(y_test))
s = fb.Modal(train=fb.array(s_train), test=fb.array(s_test))

classifier = fb.instance(LogisticRegression)
classifier = fb.fit(classifier, x, y).train  # get the classifier to be fit on the train mode of data
yhat = fb.predict(classifier, x)

print(fb.report(yhat, y=y, sensitive=s).test())  # generate report for the test mode of predictions
```
```
accuracy  	 0.750
dfnr      	 0.667
dfpr      	 0.000
prule     	 0.500
```

## :brain: Overview
`FairBench` makes use of forward-oriented programming
to declare interoperable interfaces that are easier 
to use. It suffices to keep only two things in mind:
1. code is not executed immediately but later on 
when actually used (lazy execution)
2. you can define parameters and metadata,
including sensitive information, only once
at any point of the code (even at the end)
and these are automatically handled by the whole
code segment.


To see this in action let's load some data first. 
After loading we also convert these data into
appropriate array. Using the library's method
for doing so (e.g. instead of using `numpy.array`)
lets us write code that can be run on any other
compatible backend later on. But you could just
create numpy arrays if you are more comfortable with those.

```python
import fairbench as fb


x, y, sensitive = load()

x = fb.array(x)
y = fb.array(y)
sensitive = fb.array(sensitive)
```

Next, let's try to do something simple,
such as instantiating an `sklearn` classifier. 
Notably the  instantiation is not required to 
pass parameters to classifier's constructor yet.
Up to this point you can also use a normal 
constructor, though using lazy instantiation
can be instructive:

```python
classifier = fb.instance(LogisticRegression)
```

Now on to the first important task;
fitting the classifier on the data
needs to be performed with the interface 
provided by `fairbench`. Let's do this
together with predictions:

```python
classifier = fb.fit(classifier, x=x, y=y)
yhat = classifier.predict(x)
```

If you look at the source code of the `fit` function
you will see that this just wraps the respective 
sklearn method. It just uses a more verbose
naming convention similar to the language of
other code components of the library.
Notably, the above
is a lazy definition of training and predictions
and will be conducted whenever appropriate.

If you want to see what would happen by running
your code, you can just write `print(yhat())` 
(yes, use the output as a function: you
can put additional arguments there at this
point that can be retroactively applied to
any methods of the previous code). But doing
so is not needed.

Let's do something simple and try to measure
disparate mistreatment on the above code.
This requires the ground truth labels to
compute misclassification rates, but these
will be automatically retrieved from the
arguments of the `fit` function. Instead,
we only set (once) information about
the sensitive attribute:

```python
mistreatment = abs(fb.dfnr(yhat, sensitive=sensitive)) + abs(fb.dfpr(yhat))
print(mistreatment())
```

If you ran the previous print statementtoo , you
will notice that this does not spend any more 
effort on training. This happens because
lazy execution outcomes are cached.

Finally, we do something a little more fun:
we apply an iterative method for bias
mitigation. This reruns the whole classification
scheme while tuning sample weights to eventually
produce a fairer classification. We pass
the lazy (not yet computed) prediction and
any desired evaluation outcome to the algorithm:

```python
yhat = fb.culep(yhat, fb.accuracy(yhat)+fb.prule(yhat))
```

That's it. You don't need to provide the 
classification scheme, since this is automatically 
rerun whenever yhat is internally computed. 
You also don't need to set the ground truth,
since it's also automatically retrieved.

Some components of this process (both
prule and culep) require knowledge of the sensitive
attribute. Even if you have used this in calculating
mistreatment earlier, there is no interaction between
the two and we need to define it again. For style points,
we define this at the final print statement
to be retroactively applied to all earlier methods
needing it:

```python
print(fb.prule(yhat)(sensitive=sensitive))
```

