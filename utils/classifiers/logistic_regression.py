import numpy as np
from random import shuffle

def sigmoid(x):
    h = np.zeros_like(x)
    #############################################################################
    # TODO: Implement sigmoid function.                                         #         
    #############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    size = h.shape[0]
    for i in range(size):
        h[i] = 1 / (1 + np.exp(-x[i]))
    

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    
    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    N = X.shape[0]
    D = X.shape[1]
   
    for i in range(N):
        loss += -( y[i]*np.log(sigmoid(np.dot(X[i],W))) + (1-y[i])*np.log(1 - sigmoid(np.dot(X[i],W))) )
        
    loss = loss / N + reg * sum(W*W)
    
    
    y = list(map(lambda x:[x], y))
    for i in range(D):
        dW[i] = np.dot(np.transpose(X)[i], sigmoid(np.dot(X,W)) - y) / N
        
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no     # 
    # explicit loops.                                                          #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    ############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    N = X.shape[0]
    D = X.shape[1]    
    
    loss = -(np.dot(y, np.log(sigmoid(np.dot(X,W)))) +  np.dot((1-y),np.log(1 - sigmoid(np.dot(X,W)))))
    loss = loss / N + reg * sum(W*W)
        
    y = list(map(lambda x:[x], y))
    dW = (np.dot(np.transpose(X), sigmoid(np.dot(X,W)) - y))/N

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW
