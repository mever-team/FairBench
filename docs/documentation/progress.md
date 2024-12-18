# Aggregate reports

A mechanism provided by FairBench to gather
fairness reports and aggregate them into one larger assessment
is the `Progress` class. This offers a builder pattern in which new reports are registered
sqeuentially, and at any point an amalgamation can be extracted.
The same mechanism can be used for reports that show the evolution
of datasets and algorithms over time 
(both tracked progress and reports can be serialized to and from Json,
which allows for persistence if needed).

## Gather instances

Here is how you can add report/value instances to progress and build
a report that contains all of them:

```python
import fairbench as fb

settings = fb.Progress("settings")
for name, experiment in experiments.items():
    y, yhat, sensitive = experiment()
    sensitive = fb.Dimensions(fb.categories @ sensitive)
    report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
    settings.instance(name, report)
comparison = settings.build()

comparison.show(fb.export.ConsoleTable)
```

## Persistence

Serialization and deserialization allow you to maintain fairness report or collections
of reports in `Progress` instances across
multiple application runs. For example this features can help investigate algorithm or dataset evolution
over time by retaining previous reports. To perform serialization, use the `status` property of a `Progress`
object to peek at its current build outcome without clearing gathered instances. Then perform the serialization
by converting the status into dictionaries and lists and using any framework. Here
is an example where json is used to perform the serialization:

```python
import json

dict_form = comparison.status.to_dict()
with open("comparison.json", "w") as file:
    json.dump(serialized, file)
```

Deserializing from the file entails the inverse procedure, in which a `Value` holding
the report outcome is deserialized and then passed to the `Progress` constructor to
continue building.

```python
with open("comparison.json", "r") as file:
    dict_form = json.load(file)

status = fb.core.Value.from_dict(dict_form)
comparison = fb.Progress(status)
```

## Applying reductions

FairBench already provides reduction mechanisms to aggregate fairness measures across
groups or subgroups. These can also be applied on filters to values, granted that they
encounter the same units across all dimensions when attempting a reduction (this check
is enforced for sanity).

```python
mean_comparison = comparison.status.filter(fb.reduction.mean)
mean_comparison.show(fb.export.ConsoleTable)
```