#%%

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %%

iris = datasets.load_iris()
x = iris['data'][:, (2,3)]
y = (iris['target'] == 2).astype(np.float64)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= .6, random_state = 1232)



#%%

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_csv", LinearSVC(C=15, loss = 'hinge'))
])

svm_clf.fit(x_train,y_train)
# %%

preds = svm_clf.predict(x_test)
print(confusion_matrix(y_test, preds))
print(classification_report(y_test,preds))
# %%


