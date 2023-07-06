# Stamps and model cards

When disseminating models, it is often important to summarize
evaluations under specific fairness perspectives, and whether
they meet specific thresholds. 

`FairBench` calls such evaluations *stamps*, and allows their
automated extraction from [reports](reports.md) (reports hold a lot of 
information, some of which can be matched to speccific popular
definitions of fairness to be added to model cards). Several
stamps can be aggregated into forks 
and outputted in various model card formats.

1. [Stamps](#stamps)
2. [Combine stamps](#combine-stamps)
3. [Export to model cards](#export-to-model-cards)


## Stamps
Stamps are callables that take as inputs reports,
extract specific fields, and apply potential threshold
checks. Calling a stamp returns an `Explainable` object
(or an `ExplainableError` in case it cannot be retrieved
from the report).  This holds either boolean or numeric values, 
respectively asserting that some property is met and quantifying how
well a model performs.

Several stamps are provided within the `fairbench.stamps`
backage. To avoid confusion with code library functionalities,
stamps are not accessed from the global level. Run
a stamp per:

```python
import fairbench as fb
report = ...
stamp = fb.stamps.four_fifths_rule(report)
print(stamp)
# 3/4ths ratio: False
```

## Combine stamps
You may want to check a model for various characteristics,
and therefore assess it with various stamps.
To do so, combine the stamps into one entity like in the following
snippet. Each stamp's value is an `Explainable` object
under the hood, but they all contain different explanation data;
avoid mass-explanations (e.g., with `stamps.explain`)
and instead export collections of stamps to model cards 
for a more thorough explanation (see below).

```python
stamps = fb.combine(
    fb.stamps.prule(report),
    fb.stamps.accuracy(report),
    fb.stamps.four_fifths_rule(report)
)
print(stamps)
```

```
prule: 0.212
worst accuracy: 0.799
4/5 rule: False
```

## Export to model cards
Combined stamps can be exported in popular formats 
that are often used to represent model cards. The
exports can be treated as fairness model
cards or be integrated to the full cards of models
being tested. Export methods can produce new 
files if given a path, but also return the respective
string conversion they would write to the file.

To avoid confusion with core library functionalities,
model card exports can only be accessed from 
the module `fairbench.modelcards` and not from the
top level. For example, use the following snippet to
export markdown:

```python
print(fb.modelcards.tomarkdown(stamps))  # or toyaml or tohtml
```

The output will be [this markdown](../images/example_modelcard.md).
Notice that the original stamp outputs keep track of factors,
descriptions, and recommendations to add to respective report 
fields.

The `fairbench.modelcards.tohtml` method can also let you open
the generated html in your browser (without necessarily 
creating a file) like so:

```python
fb.modelcards.tohtml(stamps, show=True)
```
