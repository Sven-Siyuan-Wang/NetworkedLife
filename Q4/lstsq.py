import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt

A = np.matrix('1 0 2; 1 1 0; 0 2 1; 2 1 1')
c = np.array([[2],[1],[1],[3]])

# b = np.linalg.lstsq(A, c)
sqerror = []
bsq = []
for lamda in np.arange(0, 5.1, 0.2):
    inverse = pinv(A.transpose().dot(A) + lamda * np.identity(3)) # inv(ATA + lamda*I)
    b = inverse.dot(A.transpose()).dot(c)
    # print(lamda, b)
    sqerror.append(np.linalg.norm(np.dot(A, b)-c)**2)
    bsq.append(np.linalg.norm(b)**2)

print(sqerror)
print(bsq)

fig, ax = plt.subplots()
x = np.arange(0, 5.1, 0.2)
ax.plot(x, sqerror, label='Squared Error')
ax.plot(x, bsq, label='b squared')
legend = ax.legend(loc='upper center', shadow=True)
# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')
ax.set_xlabel('lambda')
plt.show()


