import pandas as pd
import sklearn.model_selection
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import preprocessing

data = pd.read_csv('car.data')
# print(data)

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data['maint']))
doors = le.fit_transform(list(data['doors']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
clas = le.fit_transform(list(data['class']))

x = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(clas)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.7)

model=KNeighborsClassifier(n_neighbors=7)
model.fit(x_train,y_train)
acc=model.score(x_test,y_test)
predicted=model.predict(x_test)
print(acc)
cl=['unac','acc','good','vgood']

for x in range(len(predicted)):
    print(cl[predicted[x]],cl[y_test[x]])