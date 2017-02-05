import matplotlib.pyplot as plt


# return SIR_i[t]
def calculateSIR(G, p, n, r, t, i):
	top = G[i][i] * p[i][t]
	bottom = n[i]
	for j in range(len(G)):
		if j!=i:
			bottom += G[i][j] * p[j][t]
	return top/bottom

G = [[1, 0.1, 0.3],[0.2, 1, 0.3],[0.2, 0.2, 1]]
r = [1, 1.5, 1]
p = [[1]] * 3
n = [0.1] * 3
SIR = [[]] * 3

# iterations
for t in range(10):
	for i in range(3):
		SIR_i = SIR[i][:]
		SIR_i.append(round(calculateSIR(G,p,n,r,t,i),8))
		SIR[i] = SIR_i
		p_i = p[i][:]
		p_i.append(round(r[i]*p[i][t]/SIR[i][t],8))
		p[i] = p_i
# final SIR
for i in range(3):
	SIR_i = SIR[i][:]
	SIR_i.append(round(calculateSIR(G,p,n,r,10,i),2))
	SIR[i] = SIR_i
		
print (p)
print (SIR)
fig, ax = plt.subplots()
# ax.plot(p[0], label='Link 1')
# ax.plot(p[1], label='Link 2')
# ax.plot(p[2], label='Link 3')
ax.plot(SIR[0], label='Link 1')
ax.plot(SIR[1], label='Link 2')
ax.plot(SIR[2], label='Link 3')

# Now add the legend with some customizations.
legend = ax.legend(loc='upper right', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')
ax.set_ylabel('SIR')
ax.set_xlabel('Iteration')
# # Set the fontsize
# for label in legend.get_texts():
#     label.set_fontsize('large')

# for label in legend.get_lines():
#     label.set_linewidth(1.5)  # the legend line width
plt.show()
