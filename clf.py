import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pickle 
def png2vec(filename):
    img = mpimg.imread(filename)    
    return np.dot(img[...,:3], [0.299, 0.587, 0.114]).reshape(-1)

    
def RandomForest(n_tree, X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    for dep in range(1,20):
        trainx = X_train.copy()
        trainy = y_train.copy()
        clf = RandomForestClassifier(n_estimators = 50, random_state = 88, max_depth = dep, class_weight = 'balanced')
        clf.fit(trainx,trainy)
        print("max_depth =",dep)
        print(clf.score(X_train,y_train))
        print(clf.score(X_test, y_test))
        print()

def rbfsvm(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVC

    for Cval in range(1,10):
        trainx = X_train.copy()
        trainy = y_train.copy()
        clf = SVC(C = Cval, kernel = 'rbf', gamma= 'scale', class_weight = 'balanced')
        clf.fit(trainx,trainy)
        print("C =",Cval)
        print(clf.n_support_)
        print(clf.score(X_train,y_train))
        print(clf.score(X_test, y_test))
        print()

def polysvm(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVC
    for Cval in range(1, 5):
        for Deg in range(1, 5):
            trainx = X_train.copy()
            trainy = y_train.copy()
            clf = SVC(C = Cval, kernel = 'poly', degree = Deg, gamma= 'scale', class_weight = 'balanced')
            clf.fit(trainx,trainy)
            print("C =",Cval)
            print(clf.n_support_)
            print(clf.score(X_train,y_train))
            print(clf.score(X_test, y_test))
            print()

def knnclf(X_train, y_train, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    for k in range(5, 10):
        trainx = X_train.copy()
        trainy = y_train.copy()
        clf = KNeighborsClassifier(n_neighbors = k, weights = 'uniform')
        clf.fit(trainx,trainy)
        print("k =", k)
        print(clf.score(X_train,y_train))
        print(clf.score(X_test, y_test))
        print()

if __name__ == "__main__":
    '''
    vec = []
    lab = []
    for root, dirs, files in os.walk('./images'):
        for name in files:
            if name.endswith('.png'):
                vec.append(png2vec(root + '/' + name))
                lab.append(int(name[0]))
        
    vec = np.array(vec)
    lab = np.array(lab).reshape(len(lab), 1)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    sl = StandardScaler()
    vec = sl.fit_transform(vec)
    print(vec.shape)
    print(lab)
    print("PCA:")
    pca = PCA(n_components = 0.85)
    vec = pca.fit_transform(vec)
    print(pca.explained_variance_ratio_.shape)
    print(vec.shape)
    fvec = open('./fvec.txt', 'wb')
    flab = open('./flab.txt', 'wb')
    pickle.dump(vec,fvec)
    pickle.dump(lab,flab)
    fvec.close()
    flab.close()
    '''
    fvec = open('./fvec.txt','rb')
    flab = open('./flab.txt','rb')
    vec = pickle.load(fvec)
    lab = pickle.load(flab).ravel()
    fvec.close()
    flab.close()
    
    from sklearn.preprocessing import StandardScaler
    sl = StandardScaler()
    vec = sl.fit_transform(vec)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(vec, lab ,random_state=88)
    knnclf(X_train, y_train, X_test, y_test)