#Techiespod- Simple Machine Learning Model demonstration

# Packages and modules needed for Program
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load data
dataset = loadtxt('car details from car dekho.csv', delimiter=",");
# split data into X and y
X = dataset[:, 0:5];
Y = dataset[:, 6];
# split data into train and test sets
for i in range(1,42):
    seed = i;
    #30% testing data and 70% training data
    test_size = 0.30
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    #Assigninig variables to pre defined models
    m1 = LogisticRegression(solver='liblinear', multi_class='ovr')
    m2 = LinearDiscriminantAnalysis()
    m3 = KNeighborsClassifier()
    m4 = DecisionTreeClassifier()
    m5 = GaussianNB()
    m6 = SVC(gamma='auto')
    z=str(i);
    print("\nFor seed-"+z)
    m1.fit(X_train, y_train)
    # make predictions for test data
    print("\nLinear Regression");
    y_pred1 = m1.predict(X_test)
    predictions1 = [round(value) for value in y_pred1]
    # evaluate predictions
    accuracy1 = accuracy_score(y_test, predictions1)
    print("Accuracy: %.2f%%" % (accuracy1 * 100.0))
    #Linear Discriminant Analysis
    print("\nLinear Discriminant Analysis");
    m2.fit(X_train, y_train)
    # make predictions for test data
    y_pred2 = m2.predict(X_test)
    predictions2 = [round(value) for value in y_pred2]
    # evaluate predictions
    accuracy2 = accuracy_score(y_test, predictions2)
    print("Accuracy: %.2f%%" % (accuracy2 * 100.0))
    #KNeighbors Classifier
    print("\nKNeighbors Classifier");
    m3.fit(X_train, y_train)
    # make predictions for test data
    y_pred3 = m3.predict(X_test)
    predictions3 = [round(value) for value in y_pred3]
    # evaluate predictions
    accuracy3 = accuracy_score(y_test, predictions3)
    print("Accuracy: %.2f%%" % (accuracy3 * 100.0))
    #Descision Tree Classifier
    print("\nDescision Tree Classifier");
    m4.fit(X_train, y_train)
    # make predictions for test data
    y_pred4 = m4.predict(X_test)
    predictions4 = [round(value) for value in y_pred4]
    # evaluate predictions
    accuracy4 = accuracy_score(y_test, predictions4)
    print("Accuracy: %.2f%%" % (accuracy4 * 100.0))
    #Gaussian NB
    print("\nGaussian NB");
    m5.fit(X_train, y_train)
    # make predictions for test data
    y_pred5 = m5.predict(X_test)
    predictions5 = [round(value) for value in y_pred5]
    # evaluate predictions
    accuracy5 = accuracy_score(y_test, predictions5)
    print("Accuracy: %.2f%%" % (accuracy5 * 100.0))
    #SVC
    print("\nSVC");
    m6.fit(X_train, y_train)
    # make predictions for test data
    y_pred6 = m6.predict(X_test)
    predictions6 = [round(value) for value in y_pred6]
    # evaluate predictions
    accuracy6 = accuracy_score(y_test, predictions6)
    print("Accuracy: %.2f%%" % (accuracy6 * 100.0))
