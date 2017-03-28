import numpy as np
import rbm as rbm
import random
import projectLib as lib

full = lib.getTrainingData()
# You could also try with the chapter 4 data
# full = lib.getChapter4Data()
(training, validation) = lib.splitDataset(full)

fullStats = lib.getUsefulStats(full)
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

K = 5
F = 90
epochs = 10000
gradientLearningRate = 0.0001

# Initialise all our arrays
# We use full stats because we want ALL the movies in the training dataset
# to appear in our weights
W = rbm.getInitialWeights(fullStats["n_movies"], F, K)

for epoch in range(epochs):
    # we get a random user to apply gradient descent
    user = trStats["u_users"][random.randint(0, len(trStats["u_users"])-1)]

    # get the ratings of that user
    ratingsForUser = lib.getRatingsForUser(user, training)

    # build the visible input
    v = rbm.getV(ratingsForUser)

    # get the weights associated to movies the user has seen
    weightsForUser = W[ratingsForUser[:, 0], :, :]

    ### LEARNING ###
    # propagate visible input to hidden units
    posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser, hidBias)
    # get positive gradient
    posGrad = rbm.probProduct(v, posHiddenProb)

    ### UNLEARNING ###
    # sample from hidden distribution
    sampledHidden = rbm.sample(posHiddenProb)
    # propagate back to get "negative data"
    negData = rbm.hiddenToVisible(sampledHidden, weightsForUser, visBiasForUser)
    # propagate negative data to hidden units
    negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser, hidBias)
    # get negative gradient
    negGrad = rbm.probProduct(negData, rbm.sample(negHiddenProb))

    grad = np.zeros(W.shape)
    # we only fill gradient for movies user has seen! Otherwise it is zero
    grad[ratingsForUser[:, 0], :, :] = gradientLearningRate * (posGrad - negGrad)

    W += grad

    # every 100 epochs, print the current RMSE for training and validation sets
    # this allows you to control for overfitting e.g
    if epoch % 100 == 0:
        # We predict over the training set
        tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, hidBias, visBias, training)
        trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

        # We predict over the validation set
        vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, hidBias, visBias, training)
        vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

        print "### EPOCH %d ###" % epoch
        print "Training loss = %f" % trRMSE
        print "Validation loss = %f" % vlRMSE

### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
