import matplotlib.pyplot as plt
s = []
i = []
r = []

b = 1
k = 1/3
v = 1/50
i.append(0.1)
s.append(0.9)
r.append(0)

for t in range(1, 201):
    s.append(s[t-1]+(-b*s[t-1]*i[t-1]+v*r[t-1]))
    i.append(i[t-1] + (b*s[t-1]*i[t-1]-k*i[t-1]))
    r.append(r[t-1] + (k*i[t-1]-v*r[t-1]))

print(s[200],i[200],r[200])

fig, ax = plt.subplots()
# ax.plot(p[0], label='Link 1')
# ax.plot(p[1], label='Link 2')
# ax.plot(p[2], label='Link 3')
ax.plot(s, label='S')
ax.plot(i, label='I')
ax.plot(r, label='R')

legend = ax.legend(loc='upper right', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()