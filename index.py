import numpy as nphy
from sklearn.naive_bayes import GaussianNB
from sklearn import decomposition
import matplotlib.pyplot as matplt
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV


"""
Avant de commencer, il faut placer les données dans un dossier "data" à la racine du projet

"""


X_tr = nphy.loadtxt('data/Train/X_train.txt',delimiter=' ')
Y_tr = nphy.loadtxt('data/Train/y_train.txt')

train_size = int(X_tr.shape[0]*0.8)

X_train = X_tr[0:train_size]
Y_train = Y_tr[0:train_size]

X_v = X_tr[train_size:]
Y_v = Y_tr[train_size:]

X_test = nphy.loadtxt('data/Test/X_test.txt',delimiter=' ')
Y_test = nphy.loadtxt('data/Test/y_test.txt')

print('Data importé')


gnb = GaussianNB()

nb_clf = gnb.fit(X_train, Y_train)

prediction = nb_clf.predict(X_test)


presc = nphy.sum(prediction == Y_test) / float(X_test.shape[0])

print('Précision de Naive Bayes : %.4f' % presc)

index=0
presc = nphy.zeros(11)

for i in range(-5,5):
        lsvm_clf = svm.SVC(C=2**i,kernel='linear')
        lsvm_clf.fit(X_train, Y_train)

        prediction = lsvm_clf.predict(X_train)
        acc = nphy.sum(prediction == Y_train)/float(X_train.shape[0])
        print('Précison SVM Train : %.4f' % acc)

        prediction = lsvm_clf.predict(X_v)
        presc[index] = nphy.sum(prediction == Y_v)/float(X_v.shape[0])
        print('Précision SVM linéraire C=%f : %.4f' % (2**i, presc[index]))
        index += 1

matplt.figure()

matplt.plot(presc)

matplt.tick_params(labelright=True)

matplt.title('Précision validé et C linéaire SVM')


C_range = nphy.logspace(-5,5,10)

G_range = nphy.logspace(-5,-5,10)

param_grid = dict(gamma=G_range,C=C_range)

Cv = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=42)

grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=Cv)

grid.fit(X_test, Y_test)

print("Les meilleur parametre %s avec le score de %0.2f" % (grid.best_params_, grid.best_score_))

gsvm_clf = svm.SVC(kernel='poly', C = 1,gamma=1e-5)

gsvm_clf.fit(X_train,Y_train)
prediction = gsvm_clf.predict(X_train)
presc = nphy.sum(prediction == Y_train) / float(X_train.shape[0])
print('Précision SVM polynomial : %.4f' % presc)


pca = decomposition.PCA()

pca.fit(X_tr)

presc = nphy.zeros(12)

index = 0
for n_components in range(10, 561, 50):
        print(n_components)

        X_red = nphy.dot(X_tr,pca.components_[:,0:n_components])

        lsvm_clf = svm.SVC(C=1, kernel='linear')
        lsvm_clf.fit(X_red, Y_tr)

        prediction = lsvm_clf.predict(nphy.dot(X_test, pca.components_[:,0:n_components]))
        presc[index] = nphy.sum(prediction == Y_test) / float(X_test.shape[0])
        index += 1

matplt.figure()

matplt.plot(nphy.arange(10, 561, 50), presc)

matplt.grid()

matplt.xlabel('Nombre de fonctionnalités')

matplt.ylabel('Précision (%)')

matplt.savefig('Précision_Fonctionalités.eps',format='eps',dpi=1000)

matplt.show()
