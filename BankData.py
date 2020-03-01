# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing
#import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix

dataset=pd.read_excel('defaultcredit.xls')
#drop first row
df = dataset.iloc[1:,]
df.head()
df.columns
x=df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11',
       'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21',
       'X22', 'X23']].values
      


y=df[['Y']].values.astype('int')    


#Normalizing the data
x=preprocessing.StandardScaler().fit(x).transform(x.astype(float))

#Train test split
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=4)
print("train set:",xtrain.shape,ytrain.shape)
print("test set:",xtest.shape,ytest.shape)

##         KNN         ############
##         KNN         ############
##         KNN         ############

##Classification
from sklearn.neighbors import KNeighborsClassifier

##Training
k=7
model=KNeighborsClassifier(n_neighbors=k)
model.fit(xtrain,ytrain)
model


##Predicting
yhat=model.predict(xtest)
yhat

##Accuracy Evaluation

test_accu=metrics.accuracy_score(ytest,yhat)
print("test accuracy:accuracy_score",test_accu)


train_accu=metrics.accuracy_score(ytrain,model.predict(xtrain))
print("train accuracy:accuracy_score",train_accu)

test_accu=metrics.accuracy_score(ytest,yhat)
print("test accuracy:accuracy_score",test_accu)


####COnfusion Matrix of KNN

from sklearn.metrics import classification_report,confusion_matrix
import itertools

def plot_confusion_matrix(cm,classes,normalize=False,title='confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
        
    print(cm)

    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
print(confusion_matrix(ytest,yhat,labels=[1,0]))

cnf_matrix=confusion_matrix(ytest,yhat,labels=[1,0])
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['Defaulter=1','Defaulter=0'],normalize=False,title='Confusion Matrix')

#ROC Of KNN

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(ytest, yhat)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(ytest, yhat)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot

plt.title("ROC of KNN")
plt.show()

##AUC of KNN
from sklearn import model_selection

scoring = 'roc_auc'
results = model_selection.cross_val_score(model, ytest, yhat, cv=3,scoring=scoring)

print("AUC: %.3f (%.3f)", (results.mean(), results.std()))


###try to plot the scatter graph
xplot=df[['X1']].values.astype('int')
yplot=df[['X5']].values.astype('int')
plt.scatter(yplot[:50],xplot[:50])
plt.xlabel("Age")
plt.ylabel("Balance")


##Log loss

from sklearn.metrics import log_loss
log_loss(ytest,yhat)

from matplotlib.colors import ListedColormap

##         Logistic Regression        ############
##         Logistic Regression        ############
##         Logistic Regression        ############


from sklearn.linear_model import LogisticRegression

###Training
model=LogisticRegression(C=0.02,solver='liblinear').fit(xtrain,ytrain)

##Prediction
yhat=model.predict(xtest)

#Evaluation


##Accuracy Evaluation

test_accu=metrics.accuracy_score(ytest,yhat)
print("test accuracy:accuracy_score",test_accu)

#jaccard index

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(ytest,yhat)
#Confusion Matrix

from sklearn.metrics import classification_report,confusion_matrix
import itertools

def plot_confusion_matrix(cm,classes,normalize=False,title='confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
        
    print(cm)

    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
print(confusion_matrix(ytest,yhat,labels=[1,0]))

cnf_matrix=confusion_matrix(ytest,yhat,labels=[1,0])
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['Defaulter=1','Defaulter=0'],normalize=False,title='Confusion Matrix')

########ROC



from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(ytest, yhat)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(ytest, yhat)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.title("ROC of Logistic Regression")
plt.show()



##AUC of 
from sklearn import model_selection

scoring = 'roc_auc'
results = model_selection.cross_val_score(model, ytest, yhat, cv=3,scoring=scoring)

print("AUC: %.3f (%.3f)", (results.mean(), results.std()))




#logloss

from sklearn.metrics import log_loss
log_loss(ytest,yhat)

######         Decision Tree      ############
######         Decision Tree      ############
######         Decision Tree      ############

from sklearn.tree import DecisionTreeClassifier

##Training
model=DecisionTreeClassifier(criterion="entropy",max_depth=3)
model.fit(xtrain,ytrain)

##Prediction
yhat=model.predict(xtest)

print(yhat[0:5])
print(ytest[0:5])

#Evaluation

accuracy=metrics.accuracy_score(ytest,yhat)
print("acuuracy",accuracy)


##Accuracy Evaluation

test_accu=metrics.accuracy_score(ytest,yhat)
print("test accuracy:accuracy_score",test_accu)


from sklearn.metrics import classification_report,confusion_matrix
import itertools

def plot_confusion_matrix(cm,classes,normalize=False,title='confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
        
    print(cm)

    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
print(confusion_matrix(ytest,yhat,labels=[1,0]))

cnf_matrix=confusion_matrix(ytest,yhat,labels=[1,0])
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['Defaulter=1','Defaulter=0'],normalize=False,title='Confusion Matrix')



#ROC Of

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(ytest, yhat)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(ytest, yhat)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.title("ROC of DIcision Tree")
plt.show()



##AUC of 
from sklearn import model_selection

scoring = 'roc_auc'
results = model_selection.cross_val_score(model, ytest, yhat, cv=3,scoring=scoring)

print("AUC: %.3f (%.3f)", (results.mean(), results.std()))


from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data=StringIO()
filename="bank.png"
featureNames=df.columns[0:23]
targetNames=df["Y"].unique().tolist()

out=tree.export_graphviz(model,feature_names=featureNames,out_file=dot_data,class_names=np.unique(ytest),filled=True,special_characters=True,rotate=False)

graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)

######         Support Vector Machines      ############
######         Support Vector Machines      ############
######         Support Vector Machines      ############

ax=df[df['Y']==1][0:50].plot(kind='scatter',x='X1', y='X5', color='DarkBlue', label='Yes')
df[df['Y'] == 0][0:50].plot(kind='scatter', x='X1', y='X5', color='Yellow', label='No', ax=ax);
plt.show()

from sklearn import svm

##Training
model=svm.SVC(kernel='linear')
model.fit(xtrain,ytrain)

##Prediction
yhat=model.predict(xtest)


yhat

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(ytest,yhat)

##Accuracy Evaluation

test_accu=metrics.accuracy_score(ytest,yhat)
print("test accuracy:accuracy_score",test_accu)


from sklearn.metrics import classification_report,confusion_matrix
import itertools

def plot_confusion_matrix(cm,classes,normalize=False,title='confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
        
    print(cm)

    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
print(confusion_matrix(ytest,yhat,labels=[1,0]))

cnf_matrix=confusion_matrix(ytest,yhat,labels=[1,0])
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['Defaulter=1','Defaulter=0'],normalize=False,title='Confusion Matrix')



#ROC Of SVM
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(ytest, yhat)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(ytest, yhat)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.title("ROC of Support Vector Machine")
plt.show()



##AUC of SVM
from sklearn import model_selection

scoring = 'roc_auc'
results = model_selection.cross_val_score(model, ytest, yhat, cv=3,scoring=scoring)

print("AUC: %.3f (%.3f)", (results.mean(), results.std()))


######         Artificial Neural Networks     ############
######         Artificial Neural Networks     ############
######         Artificial Neural Networks     ############

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(xtrain)

xtrain=scaler.transform(xtrain)
xtest=scaler.transform(xtest)

from sklearn.neural_network import MLPClassifier

###Training
model=MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
model.fit(xtrain,ytrain)

##Prediction
yhat=model.predict(xtest)

###confusion matrix
##Accuracy Evaluation

test_accu=metrics.accuracy_score(ytest,yhat)
print("test accuracy:accuracy_score",test_accu)


from sklearn.metrics import classification_report,confusion_matrix
import itertools

def plot_confusion_matrix(cm,classes,normalize=False,title='confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
        
    print(cm)

    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
print(confusion_matrix(ytest,yhat,labels=[1,0]))

cnf_matrix=confusion_matrix(ytest,yhat,labels=[1,0])
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['Defaulter=1','Defaulter=0'],normalize=False,title='Confusion Matrix')



#ROC Of KNN

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(ytest, yhat)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(ytest, yhat)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.title("ROC of Artificial Neural Network")
plt.show()



##AUC of KNN
from sklearn import model_selection

scoring = 'roc_auc'
results = model_selection.cross_val_score(model, ytest, yhat, cv=3,scoring=scoring)

print("AUC: %.3f (%.3f)", (results.mean(), results.std()))


# The magic happens here
import matplotlib.pyplot as plt
import scikitplot as skplt
skplt.metrics.plot_cumulative_gain(ytest, pred)
plt.show()

######         Gaussian Naive Bayes      ############
######         Gaussian Naive Bayes      ############
######         Gaussian Naive Bayes      ############


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Training
model = GaussianNB()
model.fit(xtrain,ytrain)

#Predicting
yhat= model.predict(xtest) 


# 0:Overcast, 2:Mild
print ("Predicted Value:" ,yhat)

print("Accuracy:",metrics.accuracy_score(ytest, yhat))


##Accuracy Evaluation

test_accu=metrics.accuracy_score(ytest,yhat)
print("test accuracy:accuracy_score",test_accu)


from sklearn.metrics import classification_report,confusion_matrix
import itertools

def plot_confusion_matrix(cm,classes,normalize=False,title='confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
        
    print(cm)

    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
print(confusion_matrix(ytest,yhat,labels=[1,0]))

cnf_matrix=confusion_matrix(ytest,yhat,labels=[1,0])
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['Defaulter=1','Defaulter=0'],normalize=False,title='Confusion Matrix')



#ROC Of KNN

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(ytest, yhat)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(ytest, yhat)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.title("ROC of Guassian Naive Bayesian")
plt.show()



##AUC of KNN
from sklearn import model_selection

scoring = 'roc_auc'
results = model_selection.cross_val_score(model, ytest, yhat, cv=3,scoring=scoring)

print("AUC: %.3f (%.3f)", (results.mean(), results.std()))
