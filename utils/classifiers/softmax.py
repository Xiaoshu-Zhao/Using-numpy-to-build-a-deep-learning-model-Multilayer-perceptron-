import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    
    N = X.shape[0]
    C = W.shape[1]
    
    for i in range(N):
        score = X[i].dot(W)
        score -= np.max(score)
        softmax = np.exp(score) / np.sum( np.exp(score) )
        Li = -np.log(softmax[y[i]])
        loss += Li
        
        for j in range(C):
            if j == y[i]:
                dW[:,j] += (-1+ softmax[y[i]])*X[i].T
            else:
                dW[:,j] += (softmax[j])*X[i].T

        
    loss /= N
    dW /= N
    loss += reg*np.sum(W*W)
    dW += 2*reg*W
    
        
        
        
    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    
    N = X.shape[0]
    scores = X.dot(W)
    
    #Numerical stability
    max_scores = np.max(scores, axis = 1, keepdims = True)
    scores -= max_scores
    
    #compute softmax matrix:
    sum_of_scores_for_each_row = np.exp(scores).sum(axis = 1, keepdims = True)
    softmax_matrix = np.exp(scores) / sum_of_scores_for_each_row
    loss = np.sum( -np.log(softmax_matrix[np.arange(N), y]) )
    
    #get the Weight Gradient:
    softmax_matrix[np.arange(N), y] -= 1
    dW = X.T.dot(softmax_matrix)
    
    loss /= N
    dW /= N
    
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW
