from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. (3072,10)
    - X: A numpy array of shape (N, D) containing a minibatch of data. (500,3072)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C. (500,)
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero,(3072,10)
    delta=1.0
    # compute the loss and the gradient
    num_classes = W.shape[1]       #C=10
    num_train = X.shape[0]         #N=500
    for i in range(num_train):     #for every instance in 500
        scores = X[i].dot(W)       #(1,3072)dot(3072,10)=(1,10) one single instance in 500
        correct_class_score = scores[y[i]]     #eg y[490]=3, scores[3] is the prob
        for j in range(num_classes):       #10
            if j == y[i]:
                margin=0
            else:
                margin = scores[j] - correct_class_score + delta
            if margin > 0:
                loss += margin
                dW[:,j]+=X[i,:]/num_train
                dW[:,y[i]]*= -1

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero in (3072,10)
    delta = 1.0

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.dot(X, W)       #(500,10)
    num_train = X.shape[0]      #N=500
    # compute the margins for all classes in one vector operation
    margins = np.maximum(0, scores - scores[y] + delta)     #(500,10)
    # on y-th position scores[y] - scores[y] canceled and gave delta. We want
    # to ignore the y-th position and only consider margin on max wrong class
    margins[range(num_train),y] = 0
    loss = np.sum(margins)
 
    loss /= num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margins[margins > 0] = 1.0
    row_sum = np.sum(margins, axis=1) #(500,)
    margins[range(num_train), y] = -row_sum        
    dW += np.dot(X.T, margins)/num_train +reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
