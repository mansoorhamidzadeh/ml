import sklearn
from sklearn import svm
from sklearn import datasets

canser = datasets.load_breast_cancer()
x = canser.data
y = canser.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
clf=svm.SVC(kernel='linear',C=1,)
clf.fit(x_train,y_train)
predict=clf.predict(x_test)
acc=sklearn.metrics.accuracy_score(y_test,predict)
print(acc)