# Stamps and model cards

When disseminating models, it is often important to summarize
evaluations under specific fairness perspectives, and whether
they meet specific thresholds. 

`FairBench` calls such evaluations *stamps*, and allows their
automated extraction from [reports](../basics/reports.md) (reports hold a lot of 
information, some of which can be matched to speccific popular
definitions of fairness to be added to model cards). Several
stamps can be aggregated into forks 
and outputted in various model card formats.


## Stamps
Stamps are callables that take as inputs reports,
extract specific fields, and apply potential threshold
checks. Calling a stamp returns an `Explainable` object
(or an `ExplainableError` in case it cannot be retrieved
from the report).  This holds either boolean or numeric values, 
respectively asserting that some property is met and quantifying how
well a model performs.

Several stamps are provided within the `fairbench.stamps`
package. Find them all [here](../record/stamps.md).
To avoid confusion with core library functionalities,
stamps are not accessed from the global level. Run
a stamp per:

```python
import fairbench as fb
report = ...
stamp = fb.stamps.four_fifths(report)
print(stamp)
# 3/4ths ratio: False
```

!!! tip
    List all available stamps with `fb.stamps.available()`.

!!! warning
    Stamps require an internet connection to work
    because they are updated each time based on an 
    online database maintained in FairBench's repository.

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
    fb.stamps.four_fifths(report)
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
export markdown, where `file` is either a string path
of where to export or None (default). The method
returns the string representation of the generated
modelcard in the specified format.
The output of the snippet bellow 
is shown [here](../images/example_modelcard.md).
Notice that it kept track of factors,
descriptions, and caveats and recommendations 
to add to respective report fields.

```python
print(fb.modelcards.tomarkdown(stamps, file=None))  # or toyaml or tohtml
```

Below are the supported modelcard formats. 
You can also display a model card in a browser window
without creating a file per `fb.modelcards.tohtml(stamps, show=True)`.

| Export     | Description                                  |
|------------|----------------------------------------------|
| tomarkdown | Exports the model card to a Markdown format. |
| toyaml     | Exports the model card to a YAML format compatible with [HuggingFace](https://huggingface.co/docs/hub/en/model-cards).    |
| tohtml     | Exports the model card to an HTML format.    |
