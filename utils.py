def plot_decision_boundary(X, y, clf, title):

    h = 0.02 
    x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
    y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.4)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.show()
    

 def Gini_Index(L):
    d = {x:L.count(x) for x in L}
    n = len(L)
    p = []
    for i in d.keys():
        p.append(d[i]/n)
    GI = []
    for i in p:
        GI.append(i*i)
    return 1-np.sum(GI)
  
  
def cont_to_cat(X, y, feature):
    ''' 
    converts continuous prediction values into categorical data (classes)
    using maximisation of Gini Index gain
    '''
    sorted_feat = np.argsort(X[:, feature])

    X_sorted = X[sorted_feat, feature]
    y_sorted = y[sorted_feat]

    bins = [X_sorted[0], X_sorted[-1]]

    gini_original = Gini_Index(y_sorted)
    best_split = 0
    best_diff = 0

    for i in range(1, len(X_sorted)):

        part = (X_sorted[i-1] + X_sorted[i]) / 2          #partition/boundary for the split
        left_split = X_sorted < part
        left_gini = Gini_Index(y_sorted[left_split])        # gini impurity for left side of the split
        right_split = X_sorted >= part
        right_gini = Gini_Index(y_sorted[right_split])        # gini impurity for right side of the split

        gini_final = (np.sum(left_split)*left_gini + np.sum(right_split)*right_gini) / len(X_sorted)        # total gini impurity after the split

        diff = gini_original - gini_final   # reduction in gini impurity

        if diff > best_diff:
            best_split = part
            best_diff = diff
    
    bins.insert(1, best_split)
    X_categorised = np.digitize(X[:, feature], bins) - 1          # assigning labels to all optimal split points
    return X_categorised, bins


def covariance(x,y):
    x_mean = sum(x)/len(x)
    y_mean = sum(y)/len(y)
    sum1 = 0
    for i in range(len(x)):
        x_diff = x[i] - x_mean
        y_diff = y[i] - y_mean
        prod = x_diff*y_diff
        sum1 += prod
    return sum1/((len(x))-1)
