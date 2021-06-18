import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import  KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.target_names)
# print(cancer.feature_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# print(x_train)
# print(y_train)
classes = ['malignant' 'benign']

clf = svm.SVC(kernel="linear")  # use kernel = linear, rbf,poly, signmoid to check time and accuracy
# c is soft margin, c=0 means hard margin
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_predict)
print("accruacy from svm:",acc)

kneighbor = KNeighborsClassifier(n_neighbors=7)
kneighbor.fit(x_train, y_train)

y_predict1 = kneighbor.predict(x_test)

acc1 = metrics.accuracy_score(y_test, y_predict1)
print("accuracy from knn:", acc1)
'''comparing accuracy between knn and svm'''