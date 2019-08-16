import pandas as pd

df = pd.read_csv('./data.csv')
df.head()

X = df.drop(['gesture'], axis=1)
y = df['gesture']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
#svclassifier = SVC(kernel='poly', degree=3)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

from sklearn import svm, datasets
import pickle
import numpy as np

svmFile = open( './gesture.pckl', 'wb')
pickle.dump(svclassifier, svmFile)
svmFile.close()
