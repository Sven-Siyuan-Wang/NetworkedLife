import numpy as np
import rbm
import projectLib as lib
import matplotlib.pyplot as plt

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

K = 5

# SET PARAMETERS HERE!!!
# number of hidden units
F = 7
epochs = 50
gradientLearningRate = 0.0001

regularization = 0.001
# Initialise all our arrays
W = rbm.getInitialWeights(trStats["n_movies"], F, K)
posprods = np.zeros(W.shape)
negprods = np.zeros(W.shape)

best_epoch = {"weights":W,"rmse_t":100,"rmse_v":100}

tr_rmse = []
vl_rmse = []

for epoch in range(1, epochs):
    # in each epoch, we'll visit all users in a random order
    visitingOrder = np.array(trStats["u_users"])
    np.random.shuffle(visitingOrder)

    for user in visitingOrder:
        # get the ratings of that user
        ratingsForUser = lib.getRatingsForUser(user, training)

        # build the visible input
        v = rbm.getV(ratingsForUser)

        # get the weights associated to movies the user has seen
        weightsForUser = W[ratingsForUser[:, 0], :, :]

        ### LEARNING ###
        # propagate visible input to hidden units
        posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser)

        # get positive gradient
        # note that we only update the movies that this user has seen!
        posprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(v, posHiddenProb)

        ### UNLEARNING ###
        # sample from hidden distribution
        sampledHidden = rbm.sample(posHiddenProb)
        # propagate back to get "negative data"
        negData = rbm.hiddenToVisible(sampledHidden, weightsForUser)
        # propagate negative data to hidden units
        negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser)
        # get negative gradient
        # note that we only update the movies that this user has seen!
        negprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(negData, negHiddenProb)

        # we average over the number of users
        grad = gradientLearningRate * ((posprods - negprods) / trStats["n_users"] - regularization*W)

        W += grad

    # Print the current RMSE for training and validation sets
    # this allows you to control for overfitting e.g
    # We predict over the training set
    tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training)
    trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

    # We predict over the validation set
    vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training)
    vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

    tr_rmse.append(trRMSE)
    vl_rmse.append(vlRMSE)

    if(vlRMSE < best_epoch["rmse_v"]):
        best_epoch["rmse_v"] =vlRMSE
        best_epoch["rmse_t"] = trRMSE
        best_epoch["weights"] = W

    print ("### EPOCH %d ###" % epoch)
    print ("Training loss = %f" % trRMSE)
    print ("Validation loss = %f" % vlRMSE)



# print ("Best Training loss = %f" % best_epoch["rmse_t"])
print ("Best Validation loss = %f" % best_epoch["rmse_v"])
### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
predictedRatings = np.array([rbm.predictForUser(user, best_epoch["weights"], training) for user in trStats["u_users"]])
np.savetxt("predictedRatings.txt", predictedRatings)


fig, ax = plt.subplots()
ax.plot(tr_rmse,label = "training RMSE")
ax.plot(vl_rmse,label = "validation RMSE")
legend = ax.legend(loc='upper right', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
ax.set_xlabel("epoch")
ax.set_ylabel("RMSE")

plt.show()
