import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pickle 
def png2vec(filename):
    img = mpimg.imread(filename)    
    return np.dot(img[...,:3], [0.299, 0.587, 0.114]).reshape(-1)

def drawroc(y_test, P_test):
    from sklearn.metrics import roc_curve, auc
    # ROC Curve
    fpr, tpr, threshold = roc_curve(y_test, P_test)
    roc_auc = auc(fpr,tpr)
    print("auc:", roc_auc)
    return 
    lw = 1
    plt.figure()
    #ax = plt.gca()
    #ax.set_aspect(1)
    plt.plot(fpr, tpr, color='darkorange',
        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()
    #plt.savefig('./roc_c=%0.1f.png'%Cval,dpi=300)
    plt.clf()
def draw_distribution(X1, X2, fout=None):
    plt.figure()
    plt.hist([X1, X2], bins = 50, color = ['r','b'])
    plt.grid()
    plt.ylim(0,80)
    plt.title('data distribution')
    if fout:
        plt.savefig(fout,dpi = 300)
    else:
        plt.show()
    plt.clf()

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
    train_set = []
    test_set = []
    index = []
    for Cval in range(1,2):
        Cval = 2
        index.append(Cval)
        trainx = X_train.copy()
        trainy = y_train.copy()
        clf = SVC(C = Cval, kernel = 'rbf', gamma= 'scale', class_weight = {1:1,0:2})
        P_test = clf.fit(trainx,trainy).decision_function(trainx)
        
        neg = P_test[trainy==0]
        pos = P_test[trainy==1]
        draw_distribution(neg, pos)
        #print(P_test)
        #print(np.max(P_test), np.min(P_test),np.mean(P_test))
        #return clf.score(X_test, y_test)
        
        #print("C =",Cval)
        #print(clf.n_support_)
        #print(clf.score(X_train, y_train))
        #print(clf.score(X_test, y_test))
        #train_set.append(clf.score(X_train,y_train))
        #test_set.append(clf.score(X_test, y_test))
        #drawroc(y_test, P_test)

    '''
    plt.figure(figsize = (10, 10))
    xticks = list(map(lambda x: 'C = ' + str(x), index))
    plt.title('Correct rate on SVM with C range from 1 to 9')
    plt.xticks(index, xticks)
    plt.ylim([0.7, 1.1])
    plt.xlim([0, 10])
    plt.plot(index, train_set, lw=1, label = 'Correct rate on training set', color = 'r')
    plt.plot(index, test_set, lw=1, label = 'Correct rate on test set', color = 'g')
    plt.legend(loc = 'lower right')
    for x, y, z in zip(index, train_set, test_set):
        plt.text(x, y, '%0.2f %%' % (y * 100), ha='center', va='top',bbox=dict(facecolor='r', alpha=0.2))
        plt.text(x, z, '%0.2f %%' % (z * 100), ha='center', va='top',bbox=dict(facecolor='g', alpha=0.2))
    plt.savefig('./CorrectRate.png',dpi = 300)
    '''
    


def polysvm(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVC
    Cval = 2
    Deg = 2
    clf = SVC(C = Cval, kernel = 'poly', degree = Deg, gamma = 'scale', class_weight = {1: 1,0: 1.5}) 
    P_test = clf.fit(X_train,y_train).decision_function(X_test)
    return clf.score(X_test,y_test)
    print("C =", Cval, "degree:", Deg)
    print(clf.n_support_)
    print(clf.score(X_train,y_train))
    print(clf.score(X_test, y_test))
    print()
    drawroc(y_test, P_test)

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
            if name.endswith('.png') and name[2] == 's':
                vec.append(png2vec(root + '/' + name))
                lab.append(int(name[0]))
        
    vec = np.array(vec)
    lab = np.array(lab)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    sl = StandardScaler()
    vec = sl.fit_transform(vec)
    print(vec.shape)
    print("PCA:")
    pca = PCA(n_components = 0.85)
    vec = pca.fit_transform(vec)
    print(pca.explained_variance_ratio_.shape)
    print(vec.shape)
    fvec = open('./specvec.txt', 'wb')
    flab = open('./speclab.txt', 'wb')
    pickle.dump(vec,fvec)
    pickle.dump(lab,flab)
    fvec.close()
    flab.close()
    '''
    fvec = open('./specvec.txt','rb')
    flab = open('./speclab.txt','rb')
    vec = pickle.load(fvec)
    lab = pickle.load(flab).ravel()
    fvec.close()
    flab.close()
    
    from sklearn.preprocessing import StandardScaler
    sl = StandardScaler()
    vec = sl.fit_transform(vec)
    '''
    from sklearn.model_selection import GridSearchCV
    para = [{'kernel': ['rbf'], 'C': list(range(1,5)), 'class_weight': [{1: 1, 0: 2}, {1: 1, 0: 1}]},
    {'kernel': ['poly'], 'C': list(range(1,5)), 'class_weight': [{1: 1, 0: 2}, {1: 1, 0: 1}],'degree': list(range(1,5))}]
    from sklearn.svm import SVC
    import pandas as pd
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(vec, lab, random_state = 6)
    clf = GridSearchCV(SVC(), para)
    clf.fit(X_train, y_train)
    data = pd.DataFrame(clf.cv_results_)
    data.to_csv('./cv_results.csv')
    print(clf.cv_results_['params'][clf.best_index_])
    '''
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 10, shuffle = True, random_state = 66)
    lower = []
    upper = []
    total = []
    cas=0
    for train_index, test_index in kf.split(vec):
        cas+=1
        X_train = vec[train_index]
        y_train = lab[train_index]
        X_test = vec[test_index]
        y_test = lab[test_index]
        
        #print(test_index)
        print(cas)
        X_l = X_test[y_test == 0]
        y_l = y_test[y_test == 0]
        X_u = X_test[y_test == 1]
        y_u = y_test[y_test == 1]
        upper.append(polysvm(X_train, y_train, X_u, y_u))
        lower.append(polysvm(X_train, y_train, X_l, y_l))
        total.append(polysvm(X_train, y_train, X_test, y_test))
        #RandomForest(100, X_train, y_train, X_test, y_test)
        #polysvm(X_train, y_train, X_test, y_test)
        
        #rbfsvm(X_train,y_train,X_test,y_test)
        
    
    print("Up:", np.mean(upper))
    print("Low:", np.mean(lower))
    print("all:", np.mean(total))
    