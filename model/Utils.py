import numpy as np 
from abc import ABC, abstractmethod

class Layer(ABC):
    """
    RNN cells and embedding layers will inherit from this abstract class
    """ 
    @abstractmethod
    def _initWeights(self):
        pass

    @abstractmethod
    def _forward(self):
        pass

    @abstractmethod
    def _backward(self):
        pass




class GradientDescentMomentum(object):
    def __init__(self, b1 = 0.95):
        self.b1 = b1
        self.running_avgs = []

    def __call__(self, learn_rate, params, dparams, grad_clip = False):
        if not self.running_avgs:
            for param in params:
                self.running_avgs.append(np.zeros_like(param))
        
        if grad_clip:
            for dparam in dparams:
                np.clip(dparam, -5, 5, out=dparam)
        
        output_params = [] 
        for i,param, dparam, running_avg in zip(range(len(params)),params, dparams, self.running_avgs):
            running_avg = self.b1*running_avg + (1-self.b1)*dparam
            param = param - learn_rate*running_avg
            self.running_avgs[i] = running_avg
            output_params.append(param)
        
        return output_params if len(output_params) >1 else output_params[0]


class smoothLoss(object):
    """
    This class simply keeps the moving average of a loss during 
    training.
    """
    def __init__(self, b1=0.9):
        self.b1= b1
        self.movingAvgLoss = 1
    
    def __call__(self, loss):
        self.movingAvgLoss = self.b1 * self.movingAvgLoss + (1-self.b1)*loss
        return self.movingAvgLoss



def crossEntropy(y,yhat, mask = None):
    """
    This function computes the cross entropy between a reference probability
    distribution and a estimated probability distribution.

    Inputs:
        - y (np.array): List of integers containing the labels for every single input
        - yhat (np.matrix): Matrix of shape (M,C) where M is the number of examples and 
        C is the number of classes.
        - mask (np.array | None):  Vector representing which of the idxs were
        masked vectors, or None indicating all inputs are valid. 
    Outputs:
        - int representing the loss 
    """
    loss = -np.log(yhat[np.arange(yhat.shape[0]), y]+1e-7)
    # number of examples in the input is equivalent to the number of non-padding vectors if we have a mask 
    batch_size = y.shape[0] if mask is None else np.sum(mask)
    return np.sum(loss,axis=0)/batch_size if mask is None else np.sum(mask.astype(int)*loss,axis=0)/batch_size


def softmax(matrix_in):
    # numerically stable - subtract the largest value of the logits off before softmaxing 
    matrix_in -= matrix_in.max(axis=1, keepdims=True)
    return np.exp(matrix_in)/np.sum(np.exp(matrix_in), axis=1, keepdims=True)


def getMask(vector, pad_idx):
    # True - include this vector when computing loss, updating gradients and 
    # calculating activations as the idx is not equal to the pad_idx 
    return vector != pad_idx


def exponentialDecaySchedule(decay_rate, decay_after_epochs):
    def decay_lr(learn_rate, epoch):
        return learn_rate * decay_rate**(epoch/decay_after_epochs)
    return decay_lr


