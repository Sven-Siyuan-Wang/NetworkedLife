import numpy as np
import projectLib as lib

# shape is movie,user,rating
training = lib.getTrainingData()

#some useful stats
trStats = lib.getUsefulStats(training)
rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset
def getA(training):
    A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    # ???
    for i in range(trStats["n_ratings"]):
        A[i][trStats["movies"][i]] = 1
        A[i][trStats["n_movies"] + trStats["users"][i]] = 1
    return A

# we also get c
def getc(rBar, ratings):
    # ???
    c = np.array([ratings[i]-rBar for i in range(len(ratings))])
    return c

# apply the functions
A = getA(training)
print(A)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c):
    # ???
    inverse = np.linalg.pinv(A.transpose().dot(A))
    b = inverse.dot(A.transpose()).dot(c)
    b = np.array(b).flatten()
    return b

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    # ???
    print(A.shape)
    inverse = np.linalg.pinv(A.transpose().dot(A) + l * np.identity(A.shape[1]))  # inv(ATA + lamda*I)
    b = inverse.dot(A.transpose()).dot(c)
    return b

# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p

# Unregularised version
b = param(A, c)

# Regularised version
l = 1
b = param_reg(A, c, l)

# print(b)
print "Linear regression, l = %f" % l
print lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"])
