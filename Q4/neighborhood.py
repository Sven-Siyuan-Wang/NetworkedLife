import numpy as np
from numpy.linalg import pinv
import pprint

pp = pprint.PrettyPrinter(indent=4)
R = np.matrix('5 0 5 4; 0 1 1 4; 4 1 2 4; 3 4 0 3; 1 5 3 0')
# R = np.matrix('5 2 0 3; 3 5 1 0; 5 0 4 2; 0 3 2 5')
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
c = c - mean
print('c', c)
print('mean: ', mean)
inverse = pinv(A.transpose().dot(A))
b = inverse.dot(A.transpose()).dot(c)
b = np.array(b).flatten()
bu = b[:R.shape[0]]
bi = b[R.shape[0]:]
print(b)

R_pred = [[0 for i in range(R.shape[1])] for j in range(R.shape[0])]
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        R_pred[i][j] = mean + bu[i] + bi[j]

        # print(R_pred[i][j])
R_error = [[99 for i in range(R.shape[1])] for j in range(R.shape[0])]
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if(R.item(i,j)) != 0:
            R_error[i][j] = R.item(i,j) - R_pred[i][j]
print('R_pred')
pp.pprint(R_pred)
print('R_error')
pp.pprint(R_error)

# similarity matrix
D = [[0 for i in range(len(bi))] for j in range(len(bi))]
for a in range(len(bi)):
    for b in range(a+1, len(bi)):
        ab = 0
        sqsuma = 0
        sqsumb = 0
        for u in range(len(bu)):
            if R_error[u][a]!=99 and R_error[u][b]!=99:
                ab += R_error[u][a] * R_error[u][b]
                sqsuma += R_error[u][a]**2
                sqsumb += R_error[u][b]**2
        D[a][b] = ab / (sqsuma * sqsumb)**0.5
        D[b][a] = ab / (sqsuma * sqsumb)**0.5
print('D')
pp.pprint(D)
R_npred = R.tolist()
# print(R_npred)

for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        # if R_npred[i][j]==0:
            # find top 2 neighbors
            l = [abs(D[j][other]) for other in range(R.shape[1])]
            print(l)
            top1 = l.index(max(l))
            l[top1] = 0
            top2 = l.index(max(l))
            print(top1, top2)
            if R_error[i][top1]==99:
                R_error[i][top1] = 0
            if R_error[i][top2]==99:
                R_error[i][top2] = 0
            R_npred[i][j] = R_pred[i][j] \
                            + D[j][top1]*R_error[i][top1] / (abs(D[j][top1])+abs(D[j][top2])) \
                            + D[j][top2]*R_error[i][top2] / (abs(D[j][top1])+abs(D[j][top2]))
            R_npred[i][j] = int(round(R_npred[i][j], 2)*100)/ 100
pp.pprint(R_npred)

