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
F = 8
epochs = 30
gradientLearningRate = 0.0001
momentum = 0.5



# Parameter tuning
rmse_m = []
mrange = [0, 0.1, 0.8, 0.9]
for momentum in mrange:
    min_rmse = 2

    # Initialise all our arrays
    W = rbm.getInitialWeights(trStats["n_movies"], F, K)
    posprods = np.zeros(W.shape)
    negprods = np.zeros(W.shape)
    grad = np.zeros(W.shape)

    for epoch in range(1, epochs):
        # in each epoch, we'll visit all users in a random order
        visitingOrder = np.array(trStats["u_users"])
        np.random.shuffle(visitingOrder)

        # adaptiveLearningRate = gradientLearningRate / epoch
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
            grad = momentum*grad + gradientLearningRate * (posprods - negprods) / trStats["n_users"]
            # grad = gradientLearningRate * (posprods - negprods) / trStats["n_users"]
            W += grad

        # Print the current RMSE for training and validation sets
        # this allows you to control for overfitting e.g
        # We predict over the training set
        tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training)
        trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

        # We predict over the validation set
        vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training)
        vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

        print "### Momentum %f EPOCH %d ###" % (momentum,epoch)
        print "Training loss = %f" % trRMSE
        print "Validation loss = %f" % vlRMSE

        if vlRMSE < min_rmse:
            min_rmse = vlRMSE
            best_epoch = epoch
            best_F = F
    rmse_m.append(min_rmse)

print(rmse_m)
print(min_rmse, best_epoch, best_F)

plt.plot(mrange, rmse_m, 'ro')
plt.axis([min(mrange), max(mrange), 1.1, 1.2])
plt.title("Best RMSE vs Momentum")
plt.show()



# (1.1396005045845585, 29, 6)


### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
# predictedRatings = np.array([rbm.predictForUser(user, W, training) for user in trStats["u_users"]])
# np.savetxt("predictedRatings.txt", predictedRatings)
