import numpy as np
import rbm
import projectLib as lib
import matplotlib.pyplot as plt
import csv
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
alpha = 0.03
momentum = 0
B = 10
regularization = 0.0001

# Parameter tuning
mrange = [0.6, 0.75, 0.9]
rrange = [0.00001, 0.0001,0.0003, 0.001, 0.01]
arange = [0.01, 0.03, 0.1]
brange = [5, 10, 20]
frange = [6, 8, 10]

def getBatches(array, B):
    ret = []
    for i in range(int(len(array)/B)):
        ret.append(array[i*B:i*B+B])
    if len(array)%B != 0:
        ret.append(array[len(array)/B:])
    return ret

# output file

csvfile = open("tuningParams.csv",'w')

writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer.writerow(['Momentum', 'Regularization', 'Alpha', 'B', 'F', 'epoch', 'RMSE'])

best_momentum = 0
best_reg = 0
best_epoch = 0
best_alpha = 0
best_B = 0
best_F = 0

for momentum in mrange:
    for regularization in rrange:
        for alpha in arange:
            for B in brange:
                for F in frange:
                    # reset best params
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
                        adaptiveLearningRate = alpha / epoch**2
                        batches = getBatches(visitingOrder, B)
                        for batch in batches:
                            prev_grad = grad
                            grad = np.zeros(W.shape)
                            for user in batch:
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
                                grad += adaptiveLearningRate * ((posprods - negprods) / trStats["n_users"] - regularization * W)
                                # grad += adaptiveLearningRate * (posprods - negprods) / trStats["n_users"]

                            # mini-batch update of weights
                            W += grad + momentum * prev_grad

                        # Print the current RMSE for training and validation sets
                        # this allows you to control for overfitting e.g
                        # We predict over the training set
                        tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training)
                        trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

                        # We predict over the validation set
                        vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training)
                        vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

                        print "### momentum %.1f regularization %.5f alpha %.3f B %d F %d epoch %d ###" % (momentum, regularization, alpha, B, F, epoch)
                        print "Training loss = %f" % trRMSE
                        print "Validation loss = %f" % vlRMSE

                        if vlRMSE < min_rmse:
                            best_momentum = momentum
                            best_reg = regularization
                            best_epoch = epoch
                            best_alpha = alpha
                            best_B = B
                            best_F = F
                            min_rmse = vlRMSE
                    writer.writerow([best_momentum, best_reg, best_alpha, best_B, best_F, best_epoch, min_rmse])






# (1.1396005045845585, 29, 6)


### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
# predictedRatings = np.array([rbm.predictForUser(user, W, training) for user in trStats["u_users"]])
# np.savetxt("predictedRatings.txt", predictedRatings)
