# Choquet-classifier

Implementation of the Choquet classifier. This model was presented in "Learning monotone nonlinear models using the Choquet integral" [[1]](#1)
- 🚧Under Construction🚧

## Installation

### Dependencies
- Python (>=3.9)
- NumPy (>=1.23.3)
- SciPy (>=1.9.1)
- scikit-learn (>=1.1.1)

### User installation
- clone this repository or
- Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the Choquet classifier.

```bash
pip install choquet-classifier-glenscalai
```

# Usage
The application is compatible to [scikit-learn](https://scikit-learn.org/stable/) and can be used like every other classifier from the scikit-learn library. In order to use the Choquet classifier, import the class **ChoquetClassifier** from the module **choquet_classifier** from the package **classifier**. Some examples are shown below:

## Default example

Use the constructor and the function **fit** to initialize the Choquet classifier for a given dataset.

```python
>>> from choquet_classifier_glenscalai.choquet_classifier import ChoquetClassifier
>>> X = [[1, 3, 2],
...      [1, 0, 3]]
>>> y = [1, 0]
>>> cc = ChoquetClassifier()
>>> cc.fit(X, y)
```
Use the function **predict** to classify samples.

```python
>>> Z = [[1, 1, 2],
...      [2, 1, 3]]
>>> cc.predict(Z)
array([0, 0])
```

## Examples with hyper-parameters

```python
>>> from choquet_classifier_glenscalai.choquet_classifier import ChoquetClassifier
>>> X = [[1, 3, 2],
...      [1, 0, 3]]
>>> y = [1, 0]
>>> cc = ChoquetClassifier(additivity=3, regularization=1)
>>> cc.fit(X, y)
```
Again, the function **predict** can be used to classify samples. Note the different output compared to the first example.

```python
>>> Z = [[1, 1, 2],
...      [2, 1, 3]]
>>> sc.predict(Z)
array([0, 1])
```

## Example with different class labels

The classes do not have to be labeled with 0 and 1; any integer numbers or strings may be used instead. The smaller label in terms of the relation or lexicographic ordering is given to the negative class, whereas the other label is assigned to the positive class.

The first example contains the class labels 2 and 1. Label 2 is assigned to the positive class and label 1 is assigned to the negative class since 2>1.

```python
>>> from choquet_classifier_glenscalai.choquet_classifier import ChoquetClassifier
>>> X = [[1, 3, 2],
...      [1, 0, 3]]
>>> y = [2, 1]
>>> cc = ChoquetClassifier()
>>> cc.fit(X, y)
>>> Z = [[1, 1, 2],
...      [2, 1, 3]]
>>> sc.predict(Z)
array([1, 1])
```

The second example contains the class labels 'one' and 'two'. Label 'one' is assigned to the negative class and label 'two' is assigned to the positive class because 'one' comes lexicographically first.

```python
>>> from choquet_classifier_glenscalai.choquet_classifier import ChoquetClassifier
>>> X = [[1, 3, 2],
...      [1, 0, 3]]
>>> y = ['two', 'one']
>>> cc = ChoquetClassifier()
>>> cc.fit(X, y)
>>> Z = [[1, 1, 2],
...      [2, 1, 3]]
>>> sc.predict(Z)
array(['one', 'one'])
```

# Licence
[MIT](https://choosealicense.com/licenses/mit/)

# Reference
[1] Ali Fallah Tehrani, Weiwei Cheng, Krzysztof Dembczynski and Eyke Hüllermeier. Learning monotone nonlinear models using the Choquet integral. 2012.
