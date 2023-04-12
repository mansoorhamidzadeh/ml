import numpy as np
import sklearn.model_selection as sk
import pandas as pd
import sklearn.linear_model
import pickle

data =pd.read_csv('student-mat.csv',sep=';')
data=data[["G1","G2","G3","studytime","failures","absences"]]

predict="G3"

x=np.array(data.drop(["G3"],axis=1))
y=np.array(data[predict])

# best=0
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
# for _ in range(10):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#     linra=sklearn.linear_model.LinearRegression()
#     linra.fit(x_train,y_train)
#     acc=linra.score(x_test,y_test)
#     print(acc)
#     if acc>best:
#         best=acc
#         with open('student.pickle','wb') as f:
#             pickle.dump(linra,f)
# print(best)
newmodel=pickle.load(open('student.pickle','rb'))
result=newmodel.predict(x_test)
for x in range(len(result)):
    print(result[x],  x_test[x],  y_test[x])