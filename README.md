# Finding Donors for *CharityML*
## (Supervised Learning)


# Introduction

## Background

CharityML is a fictitious charity organization located in the heart of Silicon Valley that was established to provide financial support for people eager to learn machine learning. After nearly 32,000 letters were sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually.

To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity. With nearly 15 million working Californians, CharityML needs an algorithm to best identify potential donors and reduce overhead cost of sending mail.

The goal of this project is to determine which algorithm will provide the highest donation yield, while also reducing the total number of letters being sent.

## Goal
 To construct a model that accurately predicts whether an individual makes **more than $50,000**.


## Tools

Language: **Python**

Software: Google Colab

### Libraries


```
import numpy as np
import pandas as pd
from time import time #Return the time in seconds
from IPython.display import display

# Prepare Data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Evaluate
from sklearn.metrics import f1_score, fbeta_score, accuracy_score

# Learning Models
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Improve
from sklearn.model_selection import GridSearchCV # rather than sklearn.grid_search
from sklearn.metrics import make_scorer
from sklearn.base import clone

# Visualise
import seaborn as sns
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches

# Import supplementary visualization code visuals.py
import visuals as vs
```
## Data

`census.csv`

- Individuals' income collected from the 1994 U.S. Census.

 - The modified census dataset consists of approximately 45222 data points, with each datapoint having 13 features.

  - There are small changes to the original dataset, such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries.

 - The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. You can find [the article](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf) by Ron Kohavi online.

## Supportive Documents

- `visuals.py`: This Python script provides supplementary visualizations for the project. (Provided by Udacity)

## Progress

- Employed several supervised algorithms to accurately model individuals' income.

- Chose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data.

# Exploring the Data

### Features
* **age**: continuous.
* **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
* **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
* **education-num**: continuous.
* **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
* **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
* **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
* **race**: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other.
* **sex**: Female, Male.
* **capital-gain**: continuous.
* **capital-loss**: continuous.
* **hours-per-week**: continuous.
* **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

### Target Variable
- **income**: Income Class (<=50K, >50K)

![explore data1](https://github.com/yanglinjing/dsnd_p1_finding_donors_for_charity/blob/master/readme_img/1.png)
![explore data2](https://github.com/yanglinjing/dsnd_p1_finding_donors_for_charity/blob/master/readme_img/2.png)

# Preparing the Data

- Transform **skewed** continuous features with **outliers**: apply a <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">`logarithmic transformation`</a>
```
skewed_feature.apply(lambda x: np.log(x + 1))
```

![log transform](https://github.com/yanglinjing/dsnd_p1_finding_donors_for_charity/blob/master/readme_img/3.png)

- **Normalise** numerical variables by [`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html):
```
MinMaxScaler().fit_transform(numerical_feature)
```

- Encode **categorical** variables by [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies):
```
pd.get_dummies(features)
```

- Split data into **training** and **testing** sets
```
X_train, X_test, y_train, y_test = train_test_split(features_final, income, test_size = 0.2, random_state = 0)
```

# Evaluating Model Performance

## Naive Predictor

The purpose of generating a naive predictor is simply to show what **a base model** without any intelligence would look like.

- In the real world, ideally a base model would be
 - either the results of **a previous model **or
 - could be based on **a research paper** upon which we are looking to improve.
- When there is no benchmark model set, getting a result** better than random** choice is a place we could start from.

Thus, I chose a model that always predicted an individual made more than $50,000 as the Navie Predictor, and calculated its  [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).**accuracy-score** and **F-score**.

```
# Calculate accuracy, precision and recall
accuracy = (TP + FN) / n
recall = TP / (TP + FN)
precision = TP / (TP + FP)

# Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta = 0.5
fscore = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
```

Naive Predictor:
- Accuracy score: 0.2478,
- F-score: 0.2917


## Creating a Training and Predicting Pipeline
To properly **evaluate** the performance of each model, I created **a training and predicting pipeline**, so I can quickly and effectively train models using various sizes of training data and perform predictions on the testing data.

```
f_beta = 0.5

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
    '''

    results = {}

    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time

    # Calculate the training time
    results['train_time'] = end - start

    # Get the predictions on the test set(X_test),
    # then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time

    # Calculate the total prediction time
    results['pred_time'] = end - start

    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(predictions_train, y_train[:300])

    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(predictions_test, y_test)

    # Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta = f_beta)

    # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = f_beta)

    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    return results
```

## Initial Model Evaluation

- Initialize models and store them in `'clf_A'`, `'clf_B'`, `'clf_C'`, etc.
  - Use a `'random_state'` for each model if applicable

```
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(random_state=0)
clf_C = SVC(kernel = 'rbf', random_state=0)
clf_D = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state=0), random_state=0)
clf_E = RandomForestClassifier(random_state=0)
clf_F = GradientBoostingClassifier(random_state=0)

clfs = [clf_A, clf_B, clf_C, clf_D, clf_E, clf_F]
```

- Calculate the number of records equal to 1%, 10%, and 100% of the training data.
  - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.

```
samples_100 = len(y_train)
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)
```

```
results = {}

for clf in clfs:

    clf_name = clf.__class__.__name__
    results[clf_name] = {}

    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)

```

![models](https://github.com/yanglinjing/dsnd_p1_finding_donors_for_charity/blob/master/readme_img/4.png)


- I also used **TPOT** result as a reference.

```
# find the best pipeline by tpot
from tpot import TPOTClassifier
tpot=TPOTClassifier(generations=6, population_size=20, verbosity=3,
                    max_time_mins = 60,
                    scoring = 'f1', cv = 5, random_state = 0)
tpot.fit(X_train,y_train)
tpot.score(X_test,y_test)
tpot.export('pipeline.py')
```

OUTPUT:

63.2257694 minutes have elapsed. TPOT will close down.
TPOT closed prematurely. Will use the current best pipeline.

Best pipeline: GradientBoostingClassifier(input_matrix, learning_rate=0.5, max_depth=4, max_features=0.35000000000000003, min_samples_leaf=3, min_samples_split=17, n_estimators=100, subsample=0.4)
True

# Improving Results
I chose the ***best*** model and performed **a grid search optimization** for it over the entire training set (`X_train` and `y_train`) by tuning parameters to improve upon the untuned model's F-score.

## Best model


### **F-score ** on the testing (100% training data)

Gradient Boosting have the highest F-score (*F*$_{0.5}$ = 0.74) on the testing when 100% of the training data is used, followed by SVM  (*F*$_{0.5}$ = 0.67), Ada Boost  (*F*$_{0.5}$ = 0.67) and Random Forest (*F*$_{0.5}$ = 0.67).

The F-score of Ada Boost drops more than the other 3 algorithms when the sample size decreases to 10%.

Though SVM performs well when there is 100% and 10% sample data, its F-score decreases to extremely tiny when 1% sample size is used).


### Prediction / Training **Time**

In these 3 algorithms, SVM costs the longest time - 24.3 and 166.8 seconds for  predicting and training, respectively, which is time-consuming. Gradient Boosting used 0.03 and11.31 seconds, while Random Forest spent 0.05 and 0.89.

### Algorithm's **Suitability**

**Gradient Boosting** would be the *best* model for the data, as it shows the highest F-score and accuracy while using relatively short time.


## Model Tuning

- Initialize the best model and store it in `clf`.
 - Set a `random_state` same as before.
- Create a dictionary of parameters to tunel.
 - Example: `parameters = {'parameter' : [list of values]}`.

- Use `make_scorer` to create an `fbeta_score` scoring object (with $\beta = 0.5$).
- Perform grid search on the classifier `clf` using the `'scorer'`, and store it in `grid_obj`.
- Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_fit`.

```
# Initialize the classifier
clf = GradientBoostingClassifier(random_state=0)

# Create the parameters list you wish to tune, using a dictionary if needed.
parameters = {
    'learning_rate': [0.1, 0.5, 1],
    'max_depth': [3, 4, 5],
    'max_features': [0.35],
    'min_samples_leaf': [2, 3, 4],
    'min_samples_split': [17],
    'n_estimators': [100],
    'subsample': [0.4]
}

# Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta = f_beta)

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scorer)

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-after scores
acc_before = accuracy_score(y_test, predictions)
f_before = fbeta_score(y_test, predictions, beta = 0.5)

acc_after = accuracy_score(y_test, best_predictions)
f_after = fbeta_score(y_test, best_predictions, beta = 0.5)
```

![model tuning results](https://github.com/yanglinjing/dsnd_p1_finding_donors_for_charity/blob/master/readme_img/5.png)

# Feature Selection

## Extracting Feature Importance

 - Import a supervised learning model from `sklearn` with a `feature_importance_` attribute.
  - This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.
 - Train the model on the entire training set.
 - Extract the feature importances using `'.feature_importances_'`.

```
# Train the supervised model on the training set
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)

# Extract the feature importances using .feature_importances_
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)
```

![top 5 features](https://github.com/yanglinjing/dsnd_p1_finding_donors_for_charity/blob/master/readme_img/6.png)

## Reduce Feature Space

With **less features** required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics.

From the visualization above, we see that the **top five** most important features contribute more than half of the importance of all features present in the data. This hints that we can attempt to ***reduce the feature space*** and simplify the information required for the model to learn.

Thus, I trained the model on the same training set *with only the top five important features*.

```
# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores

acc_reduced = accuracy_score(y_test, reduced_predictions)
f_reduced = fbeta_score(y_test, reduced_predictions, beta = 0.5)

```

![scores by reduced features](https://github.com/yanglinjing/dsnd_p1_finding_donors_for_charity/blob/master/readme_img/7.png)
