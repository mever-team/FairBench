
# About

AI is everywhere and responsible for many sensitive decisions.

Consider a benign system that predicts whether
persons prefer eating doughnuts :doughnut: rather than 
icecream :cake: by looking at them. We want to
handout their preferred food in a promotion.

To _test_ the
system we need to create some predictions for a bunch of people
and compare with what they actually like; a checkmark :heavy_check_mark:
shows correct predictions and a cross :x: wrong ones.

(We are not going into detail on how we
created that system.)


:girl: &rarr; :cake: :heavy_check_mark:<br>
:girl: &rarr; :cake: :x:<br>
:girl: &rarr; :doughnut: :x:<br>
:boy: &rarr; :doughnut: :heavy_check_mark:<br>
:boy: &rarr; :doughnut: :heavy_check_mark:<br>
:boy: &rarr; :cake: :heavy_check_mark:<br>

Would these predictions be unbiased? There are not a lot of people, so any
imbalances can be random. But if we still want to check for fairness,
we need to check which demographic groups to compare and what are the 
things we want to balance between those demographics.

Let's balance per gender and get the chance of giving donuts:

66% :boy: &rarr;  :doughnut:<br>
33% :girl: &rarr;  :doughnut:

There is some clear imbalance there called disparate impact.
Is it justified? We need
some input from experts. It's hazardous to make judgement on our own,
but can say that what is 
known as the p-rule = smallest/largest = 33%/66% = 0.5 is less than 0.8
(a common threshold) and therefore further investigation is needed.

Let's see what other kinds of comparison we could make. We may want
to compare the error rates instead, to give everybody the food
they like. 

0% :boy: &rarr;  :x:<br>
66% :girl: &rarr;  :x:

This time things are even worse! Our system makes a lot of mistake
for one gender, and perfect predictions for the other. This difference
in errors is called disparate mistreatment. Again,
we need to consult with domain experts (e.g., sociologists,
legal experts) to check that this is actually
an issue. 

There are many more ways of comparing groups of people, or even to
check whether things are fair at the level for individuals. We may
want to consider information other than gender to protect, such as
the race or even things that indicate those (called proxy attributes)
like clothe style.
