# dice-calculator
 
When designing table top role playing games, it is important to understand the full complexity of the dice rolls that your design employs. However, it can be difficult to intuit the behavior of a somewhat complicated dice roll at a glance.

This tool takes arbitrary dice roll expressions and outputs information such as the probability mass function.

# Work In Progress

Its kinda three-quarters finished right now. Use at your own risk.

## core functionality
* expression compiler

## qol functionality
* expression simplifier

## extended dice mechanics
* keep highest/lowest X in pool, sum those X
* reroll lowest/highest 1 in pool, sum entire pool
* reroll lowest/highest X in pool, sum entire pool
* reroll any instance of minValue/maxValue in pool, sum entire pool
* count successes over/under X in pool
* mixed size pools for extended functions (example: roll N1dS1 and N2dS2, keep highest X from entire pool of S1+S2, sum those X)
* mixed function pools (example: roll NdS, reroll any instance of minValue, count success over X)
