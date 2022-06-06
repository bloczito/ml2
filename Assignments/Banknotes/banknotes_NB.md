---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{raw-cell}

---
output: github_document
---
```

# Counterfeit detection

+++

The task in this assignment is to detect the  counterfeit banknotes. The data set is based on [banknote authentication Data Set ](https://archive.ics.uci.edu/ml/datasets/banknote+authentication#) from UCI Machine Learning repository. The first three columns denote different parameters obtained from the photographs of the banknotes and last colum provides the label. Frankly as the dataset does not have any description I don't know  which labels corresponds to real and which to counterfeited banknotes. let's assume that label one (positive) denotes the clounterfeits. The set  "banknote_authentication.csv" can be found in the data  directory.

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
# import scrapbook as sb
```

```{code-cell} ipython3
import  matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(8,8)
```

Please insert you  firstname  and name below

```{code-cell} ipython3
# sb.glue("Who", ["Bartosz", "Krawiec"])
# sb.glue("Who", ["Weronika", "Materna"])
```

```{code-cell} ipython3
:tags: []

from  sklearn.model_selection import train_test_split
seed = 31287
```

```{code-cell} ipython3
data = pd.read_csv('data/banknotes_data.csv')
```

```{code-cell} ipython3
:tags: []

data.head()
```

```{code-cell} ipython3
:tags: [skip]

data.describe()
```

```{code-cell} ipython3
:tags: [skip]

data.info()
```

```{code-cell} ipython3
:tags: []

data_train, data_test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data.loc[:,'counterfeit'], random_state=seed)
```

```{code-cell} ipython3
:tags: []

lbls_train = data_train['counterfeit']
print(lbls_train)
print(type(lbls_train))
```

```{code-cell} ipython3
:tags: []

fig, ax = plt.subplots(1,4, figsize=(22,5))
for i in range(4):
    ax[i].hist(data_train[lbls_train==0].iloc[:,i], bins=32, histtype='step', color='blue')
    ax[i].hist(data_train[lbls_train==1].iloc[:,i], bins=32, histtype='step', color='red')
    ax[i].hist(data_train[lbls_train==0].iloc[:,i], bins=32, histtype='bar', color='lightblue', alpha=0.25)
    ax[i].hist(data_train[lbls_train==1].iloc[:,i], bins=32, histtype='bar', color='orange', alpha =0.25)
```

+++ {"tags": []}

You will have to install a popular plotting library `seaborn`

```{code-cell} ipython3
:tags: []

import seaborn
```

```{code-cell} ipython3
:tags: []

seaborn.pairplot(data_train.iloc[:,0:5], hue='counterfeit');
```

```{code-cell} ipython3
:tags: []

len(data_train)
```

## Problem 1

+++

Implement Gaussian  Bayes classifier using only one feature. Which feature will you choose? Calculate the confusion matrix (normalized as to show rates), ROC AUC score and plot ROC curve. Do this bot for training and validation set. Plot both curves on the same plot. Save everything using `scrapbook`.

+++

__Hint__ For calculating metrics and plotting ROC curves you may use functions from scikit-learn: `roc_curve`, `roc_auc_score` and `confusion matrix`. For estimating normal distribution parameters  use `norm.fit` `from scipy.stats`. Use `norm.pdf` for normal probability density function.

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy.stats import norm


data = pd.read_csv('data/banknotes_data.csv')

seed = 31287
data_train, data_test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data.loc[:,'counterfeit'], random_state=seed)

lbls_train = data_train['counterfeit']
lbls_test = data_test['counterfeit']
```

```{code-cell} ipython3
norm_counterfeit = norm(*norm.fit(data_train.a0[lbls_train == 1]))
norm_legit = norm(*norm.fit(data_train.a0[lbls_train == 0]))

p_counterfeit = len(data_train.a0[lbls_train == 1]) / len(data_train)
p_legit = 1 - p_counterfeit
```

```{code-cell} ipython3
def pdf_counterfeit_cond(x):
    return norm_counterfeit.pdf(x) * p_counterfeit / (norm_counterfeit.pdf(x) * p_counterfeit + norm_legit.pdf(x) * p_legit)
```

```{code-cell} ipython3
x_axis = np.linspace(-10, 10, 100)
plt.plot(x_axis, pdf_counterfeit_cond(x_axis))
plt.axhline(0.5, color="orange")
plt.axvline(0, color="red")
```

```{code-cell} ipython3
tn, fp, fn, tp = confusion_matrix(lbls_train, pdf_counterfeit_cond(data_train.a0) > 0.5).ravel()
p = tp + fn
n = tn + fp
tpr = tp / p
fpr = fp / n
confusion_matrix(lbls_train, pdf_counterfeit_cond(data_train.a0) > 0.5, normalize="true")
```

```{code-cell} ipython3
tn_test, fp_test, fn_test, tp_test = confusion_matrix(lbls_test, pdf_counterfeit_cond(data_test.a0) > 0.5).ravel()
p_test = tp_test + fn_test
n_test = tn_test + fp_test
tpr_test = tp_test / p_test
fpr_test = fp_test / n_test
confusion_matrix(lbls_test, pdf_counterfeit_cond(data_test.a0) > 0.5, normalize='true')
```

```{code-cell} ipython3
roc_auc_score(lbls_train, pdf_counterfeit_cond(data_train.a0))
```

```{code-cell} ipython3
roc_auc_score(lbls_test, pdf_counterfeit_cond(data_test.a0))
```

```{code-cell} ipython3
fprs, tprs, thds = roc_curve(lbls_train, pdf_counterfeit_cond(data_train.a0))
test_fprs, test_tprs, test_thds = roc_curve(lbls_test, pdf_counterfeit_cond(data_test.a0))

fig, ax = plt.subplots(figsize=[10, 15])
ax.set_aspect(1)
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')

ax.set_title("ROC curve")

ax.plot(fprs, tprs, linewidth=1, label="Train")
ax.plot(test_fprs, test_tprs, linewidth=1, label="Test")

ax.scatter([fpr], [tpr])
ax.scatter([fpr_test], [tpr_test])
```

## Problem 2

+++

Same as Problem 1 but now implement Gaussian Naive Bayes using two features. Compare ROC curves on the test set. What is teh improvement of AUC score on the test set?

```{code-cell} ipython3
f1 = "a0"
f2 = "a1"

norm_counter_f1 = norm(*norm.fit(data_train[f1][lbls_train == 1]))
norm_legit_f1 = norm(*norm.fit(data_train[f1][lbls_train == 0]))

norm_counter_f2 = norm(*norm.fit(data_train[f2][lbls_train == 1]))
norm_legit_f2 = norm(*norm.fit(data_train[f2][lbls_train == 0]))

p_counter = len(data_train.a0[lbls_train == 1]) / len(data_train)
p_legit = 1 - p_counter
```

```{code-cell} ipython3
def pdf_counterfeit_cond2(f1, f2):
    return norm_counter_f1.pdf(f1) * norm_counter_f2.pdf(f2) * p_counter / \
           (norm_counter_f1.pdf(f1) * norm_counter_f2.pdf(f2) * p_counter + norm_legit_f1.pdf(f1) * norm_legit_f2.pdf(f2) * p_legit)
```

```{code-cell} ipython3
confusion_matrix(lbls_train, pdf_counterfeit_cond2(data_train[f1], data_train[f2]) > 0.5, normalize="true")
```

```{code-cell} ipython3
tn2, fp2, fn2, tp2 = confusion_matrix(lbls_test, pdf_counterfeit_cond2(data_test[f1], data_test[f2]) > 0.5).ravel()
p2 = tp2 + fn2
n2 = tn2 + fp2
tpr_test2 = tp2 / p2
fpr_test2 = fp2 / n2


confusion_matrix(lbls_test, pdf_counterfeit_cond2(data_test[f1], data_test[f2]) > 0.5, normalize="true")
```

```{code-cell} ipython3
roc_auc_score(lbls_train, pdf_counterfeit_cond2(data_train[f1], data_train[f2]))
```

```{code-cell} ipython3
roc_auc_score(lbls_test, pdf_counterfeit_cond2(data_test[f1], data_test[f2]))
```

```{code-cell} ipython3
fprs2, tprs2, _ = roc_curve(lbls_test, pdf_counterfeit_cond2(data_test[f1], data_test[f2]))
fprs_test, tprs_test, _ = roc_curve(lbls_test, pdf_counterfeit_cond(data_test.a0))

fig, ax = plt.subplots(figsize=[12, 12])
ax.set_aspect(1)
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title("ROC curve", fontsize=14)

roc_test = ax.plot(fprs_test, tprs_test)
roc_test2 = ax.plot(fprs2, tprs2)

ax.scatter([fpr_test], [tpr_test], label="One parameter")
ax.scatter([fpr_test2], [tpr_test2], label="Two parameters")
ax.legend()
```

## Problem 3

```{code-cell} ipython3

```

Same as Problem 2 but now implement Gaussian Naive Bayes using all features.
