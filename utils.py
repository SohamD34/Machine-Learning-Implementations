def gradient_descent(x_train,y_train,model):

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    w = tf.Variable(0.5, name="weights")

    #  model = tf.add(tf.multiply(x, w), 0.5)         # x*w + 0.5
    
    cost = tf.reduce_mean(tf.square(model - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            sess.run(train,feed_dict={x:x_train, y:y_train})
        w_val = sess.run(w)

    return w_val
    

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


def Mahalanobis(x, y, cov_mat):
    xx=x-y
    cov_inv=np.linalg.inv(cov_mat)
    mah=np.matmul(np.matmul(xx.T,cov_inv),xx)
    mah_d=mah**0.5
    return mah_d


def split_class_wise(frame,uniq_classes):
    class_wise_data = []
    for i in range(len(uniq_classes):
        class_data = df[df['Class']==uniq_classes[i]]
        class_wise_data[i] = (class_data.to_numpy())
    return class_wise_data

                   
def within_class_means(class_data):
    class_data = pd.DataFrame(class_data, columns = df.columns.values)
    names = class_data.columns.values
    class_means = []
    for i in names:
        attribute_values = class_data[i]
        attribute_mean = np.mean(attribute_values)
        class_means.append(attribute_mean)
    return class_means
                   
                   
def overall_mean(X):
    col_names = X.columns.values
    overall_mean_vec = []
    for i in col_names:
        L = list(data[i])
        mean = np.mean(L)
        overall_mean_vec.append(mean)
    return overall_mean_vec
                   
                   
def within_class_scatter_matrix(class_mean,class_data):

    mean = np.array(class_mean.drop("Class"))
    data = class_data.drop("Class",axis=1)
    scatter_matrix = np.zeros((13,13))

    for i in range(len(data)):

        datapoint = np.array(data.iloc[i,:])

        diff = np.matrix(datapoint - mean)
        diff_T = diff.T

        point_mat = np.dot(diff.T,diff)
        scatter_matrix+=point_mat

    return np.matrix(scatter_matrix)
                   
                   
def between_class_scatter_matrix(class_means, df_mean, class_sizes):

    scatter_matrix = np.zeros((13,13))

    for i in range(len(class_sizes)):

        mi = np.array(class_means[i][1:])
        m = np.array(df_mean)
        Ni = np.array(class_sizes[i])
        diff = np.matrix(mi - m)
        diff_T = diff.T
        class_mat = Ni*(np.dot(diff_T,diff))
        scatter_matrix += class_mat

    return np.matrix(scatter_matrix)
                   
                   
