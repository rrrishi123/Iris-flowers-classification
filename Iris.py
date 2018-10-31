# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

path=r"Iris.csv"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=pandas.read_csv(path,names=names)
print("Shape is",dataset.shape,"\n")

print("First five data (head)\n",dataset.head(),"\n")

print("Desciption of Data\n",dataset.describe(),"\n")

print("Dataset grouped by",dataset.groupby('class').size(),"\n")

print("Plot of the Individual Variable (univariate plots)")
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()

print("\nHistogram of the individual variable dataset are-")
dataset.hist()
plt.show()

print("\nScatter plots of all variables, for structured relaionship between variables")
scatter_matrix(dataset)
plt.show()

array=dataset.values
X=array[:,0:4]
Y=array[:,4]

validation_size=0.20
seed=7
X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)
seed=7
scoring='accuracy'
models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))

results = []
names = []
print("Algorithms Comparision for LR(Linear Regression), LDA(Linear Discriminant Analysis), KNN(K Neighbours Classifier), CART(Decision Tree Classifier), NB(Gaussian NB),SVM(Support Vector Machine)")
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison on Graphs')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show() 
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)

print("Classification Accuracy is",accuracy_score(Y_validation, predictions),"\n")
print("misclassification accuracy is",1-accuracy_score(Y_validation,predictions),"\n")
print("Confusion Matrix is\n",confusion_matrix(Y_validation, predictions),"\n")
print("Classification Report is\n",classification_report(Y_validation, predictions))
