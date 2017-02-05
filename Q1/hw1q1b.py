import matplotlib.pyplot as plt


# return SIR_i[t]
def calculateSIR(G, p, n, r, t, i):
	top = G[i][i] * p[i][t]
	bottom = n[i]
	for j in range(len(G)):
		if j!=i:
			bottom += G[i][j] * p[j][t]
	return top/bottom
size = 4
# G = [[1, 0.1, 0.3],[0.2, 1, 0.3],[0.2, 0.2, 1]]
G = [[1, 0.1, 0.3,0.1],[0.2, 1, 0.3,0.1],[0.2, 0.2, 1,0.1],[0.1,0.1,0.1,1]]
# G = [[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5, 1]]
r = [1, 1.5, 1, 1]
# r = [1,2,1]
p = [[0.19],[0.3],[0.2],[1]]
n = [0.1] * size
SIR = [[]] * size

# iterations
for t in range(10):
	for i in range(size):
		SIR_i = SIR[i][:]
		SIR_i.append(round(calculateSIR(G,p,n,r,t,i),4))
		SIR[i] = SIR_i
		p_i = p[i][:]
		p_i.append(round(r[i]*p[i][t]/SIR[i][t],4))
		p[i] = p_i
# final SIR
for i in range(size):
	SIR_i = SIR[i][:]
	SIR_i.append(round(calculateSIR(G,p,n,r,10,i),2))
	SIR[i] = SIR_i
		
print (p)
print (SIR)
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
for i in range(size):
	ax.plot(p[i], label='Link '+str(i+1))
	ax2.plot(SIR[i], label='Link '+str(i+1))


# Now add the legend with some customizations.
legend = ax.legend(loc='upper right', shadow=True)
legend2 = ax2.legend(loc='upper right', shadow=True)
# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')
frame2 = legend2.get_frame()
frame2.set_facecolor('0.90')
ax2.set_ylabel('SIR')
ax.set_ylabel('power (mW)')
ax.set_xlabel('Iteration')
ax2.set_xlabel('Iteration')
# # Set the fontsize
# for label in legend.get_texts():
#     label.set_fontsize('large')

# for label in legend.get_lines():
#     label.set_linewidth(1.5)  # the legend line width
plt.show()
