# Dyploma project: building an ensemble of classifiers and regressors with TPOT and auto-sklearn using genetic alorithm 
**This project was carried out under the supervision of PhD [Andriy Konovalov](https://github.com/kamua)**

This project implements functions for building an ensemble of classifiers and regressors using the TPOT and auto-sklearn libraries with the help of a genetic algorithm evolved using the deap library. The functions allow using these libraries to optimise machine learning models and build an ensemble of them.

Below is a flowchart demonstrating the stages of the developed methodology:

![image](https://github.com/yevIbrahimov/gen_ensemble/assets/61506686/a21ab112-4b3e-4ef5-a162-627c7b551b91)


**Usage**.
Import the functions from this file into your project.
1. Use the gen_ensemble_builder_clf function to build a classifier ensemble.
2. Use the gen_ensemble_builder_regr function to build a regressor ensemble.

```python
# load data
X_train, X_test, y_train, y_test = load_data()

# run classification func
result = gen_ensemble_builder_clf(X_train, X_test, y_train, y_test)

# print result
print(result)
```

Using own TPOT object for regression

```python
pipeline_optimizer = TPOTClassifier(max_time_mins=5, scoring='r2')
result = gen_ensemble_builder_regr(X_train, X_test, y_train, y_test, TPOT_object=pipeline_optimizer, use_aurosklearn=False)
```

Using own crossover and mutation genetic operators

```python
def cxSet(ind1, ind2):
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2

def mutSet(individual):
    """Mutation that pops or add an element."""
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(NBR_ITEMS))
    return individual,

result = gen_ensemble_builder_clf(X_train, X_test, y_train, y_test, cx_operator=cxSet, mut_operator=mutSet)
```
