from Utils import Layer
import numpy as np 

class Embedding_layer(Layer):
    """
    This class represents an embedding layer. The purpose of the layer is to transport
    vectors from a vector space V to a vector space W through a linear transformation 
    by a weight matrix, where vector space W has a much lower dimensionality then vector space V. 

    Inputs:
        -> dim_in (int): Integer representing the dimensionality of the input vector space
        -> embed_dim (int): Integer representing the dimensionality of the embedded vector space
        -> optim (Class): Class representing an optimization algorithm 
    """ 
    def __init__(self, dim_in, embed_dim, optim):
        self.dim_in =dim_in
        self.embed_dim = embed_dim
        self.W = self._initWeights()
        self.optim = optim() 
        # cache the inputs to the layer - will change every
        # forward pass to what the current inputs are 
        # type {np.matrix} Matrix of integers of shape (M,T) 
        self.x = None 

    def _initWeights(self):
        return np.random.randn(self.dim_in, self.embed_dim)*0.01

    def _forward(self, x):
        # Shape: (M, T, embed_dim) where M is size of batch, T is number of timesteps in a sequence, and embed_dim
        # is the dimension the vectors are embedded to
        
        # Implemented efficiently with embedding lookup - expects x to be (M,T) 
        embedded_vectors = self.W[x, :]
        self.x = x 
        return embedded_vectors

    def _backward(self, dZ, learn_rate):
        # dZ should be of shape (M, d_embed)
        # then we can get dW with (d_vocab,M).dot(M,d_embed)
        # since input is technically a one hot encoded matrix,
        # all we have to do is add the corresponding row in dZ to the 
        # row in dW (IE: the vocab_dim the row in dZ was produced by)
        dW = np.zeros_like(self.W)
        np.add.at(dW, self.x, dZ)
        self.W = self.optim(learn_rate, [self.W], [dW])
