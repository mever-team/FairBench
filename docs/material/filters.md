# Filters

Here are available filters that you can pass as positional arguments to 
the `filter` method of reports. Pass either the classes
or instantiated class objects to that method. Explanations
for filter arguments are provided too. Everything is imported
from the `investigate` module. 

## DeviationsOver

Its main role is to simplify reports so that only unacceptable numerical deviations stand out.
These are its constructor arguments:

- `limit` Mandatory. A number that indicates maximally allowed deviations from ideal values. If it is not exceeded, report values are either omitted as unimportant or colored green.
- `action` Either "keep" or "colorize". The first makes this filter remove all report values that do not satisfy the criteria. The second changes how the report is colorized based on the criteria.
- `shallow` A boolean value that is relevant on when a "keep" action is selected. If True (default), only the top numeric values are filtered. Otherwise the filter checks all internal values too.

## IsBias

Keeps only report entries where values close to zero are considered better.
This way, all high values in the filtered report indicate violations of ideal behavior.

## Stamps

It contains a collection of fairness assessment stamps, which correspond to popular literature definitions.
Only known stamps are retained from the report. However, these are enriched with caveats and recommendations 
coming from a socio-technical database of the MAMMOth EU project.
