import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
def plot_roc_curve(estimator, X, y, title):
    # Determine the false positive and true positive rates
    fpr, tpr, _ = roc_curve(y, estimator.predict_proba(X)[:,1])

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)
    print ('ROC AUC: %0.2f' % roc_auc)

    # Plot of a ROC curve for a specific class
    plt.figure(figsize=(10,6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - {}'.format(title))
    plt.legend(loc="lower right")
    plt.show()



train = pd.read_csv('Synthetic_cargo_data_1K.csv')
print ("Dimension of train data {}".format(train.shape))
from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(random_state=42, n_estimators=500, min_samples_leaf=12, min_samples_split=5,
#                               criterion='entropy', oob_score=True, n_jobs=5)
#model = SVC(random_state=47, C=25, gamma=0.0055, coef0=0.1,  kernel='poly', probability=True)
#model = tree.DecisionTreeClassifier()
model = LogisticRegression(random_state=42, penalty='l2', C=25)

'''
X = train.drop(['Label'], axis=1)
X = train.drop(['Date'], axis=1)
print(train.axes)
y = train.Label
'''

#print(train[train.Label==1].sum())
features = [u'Customs Code', u'Weight', u'Temperature', u'Origin', u'Bank',
            u'Payment Value', u'Currency', u'Cash', u'Prepaid', u'Shipping Line']
X = train[features]
y = train['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model.fit(X_train, y_train)
#print(model.support_vectors_)
print (accuracy_score(y_test, model.predict(X_test)))

print(cross_val_score(model, X, y, cv=5))
plot_roc_curve(model,X,y,"")
'''
skf = StratifiedKFold(n_splits=5)
for train, test in skf.split(X, y):
    model.fit(X.iloc[train], y.iloc[train])
    print (accuracy_score(y.iloc[test], model.predict(X.iloc[test])))
#
#print(train.describe())
'''
