import numpy as np 
class GradientDescentMomentum(object):
    def __init__(self, b1 = 0.9):
        self.b1 = b1
        self.running_avgs = []
    
    def __call__(self, learn_rate, params, dparams, grad_clip = False):
        if not self.running_avgs:
            for param in params:
                self.running_avgs.append(np.zeros_like(param))
        
        output_params = [] 
        for i,param, dparam, running_avg in zip(range(len(params)),params, dparams, self.running_avgs):
            running_avg = self.b1*running_avg + (1-self.b1)*dparam
            param = param - learn_rate*running_avg
            self.running_avgs[i] = running_avg
            output_params.append(param)
        
        if grad_clip:
            for param in output_params:
                np.clip(param, -0.2, 0.2, out=param)
        return output_params 


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
    loss = -np.log(yhat[np.arange(yhat.shape[0]), y])
    return np.sum(loss,axis=0) if mask is None else np.sum(mask.astype(int)*loss,axis=0)


def softmax(matrix_in):
    return np.exp(matrix_in)/np.sum(np.exp(matrix_in), axis=1, keepdims=True)

def getMask(vector, mask_idx):
    return vector != mask_idx



yhat = np.array([[0.5,0.1,0.1,0.3], [0.8,0.1,0.05,0.05], [0.3,0.4,0.2,0.1]])
y = np.array([1,2,3])
mask = np.array([True, True, False])
ce = crossEntropy(y, yhat, mask)

print(ce)


params = [np.random.randn(3,5) for i in range(2)]
dparams = [np.random.randn(3,5) for i in range(2)]
obj = GradientDescentMomentum()
obj(0.7, params, dparams, True)


seq = np.array(([3,4,9,8,7,1,1,1,1],[4,5,6,7,8,9,1,1,1]))
out = getMask(seq, 1)
print(out)


obj1 = smoothLoss()
print(obj1(5.32))


logits_prac = np.array([[0.3, 0.2, 0.4, 0.1], [0.6,0.2,0.1,0.1], [0.5,0.3,0.1,0.1]])
print(softmax(logits_prac))

logits_prac2 = np.array([[0.8,0.2], [0.9,0.1]])
print(softmax(logits_prac2))

