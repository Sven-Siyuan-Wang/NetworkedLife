import numpy as np
from numpy.linalg import pinv


import pprint
pp = pprint.PrettyPrinter(indent=4)
R = np.matrix('5 0 5 4; 0 1 1 4; 4 1 2 4; 3 4 0 3; 1 5 3 0')

# construct matrix A

c = []
sum = 0
count = 0
A = []
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R.item(i,j) != 0:
            c.append(R.item(i,j))
            sum += R.item(i,j)
            count += 1
            newA = [0] * (R.shape[0]+R.shape[1])
            newA[i] = 1
            newA[R.shape[0]+j] = 1
            A.append(newA)

A = np.matrix(A)
mean = sum / count
c = np.array(c)
# print(c)
c = c - mean
# print(c)
# print(mean)
# print(A)
# print(A.transpose().dot(A))
inverse = pinv(A.transpose().dot(A))
b = inverse.dot(A.transpose()).dot(c)
b = np.array(b).flatten()
bu = b[:R.shape[0]]
bi = b[R.shape[0]:]
print(bu, bi)

R_pred = [[0 for i in range(R.shape[1])] for j in range(R.shape[0])]
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        temp = mean + bu[i] + bi[j]
        R_pred[i][j] = int(round(temp, 2)*100) /100
        # print(R_pred[i][j])

pp.pprint(R_pred)

print(np.linalg.norm(A.dot(b.transpose())-c)**2)