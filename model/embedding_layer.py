from model.utils import Layer
import numpy as np


class EmbeddingLayer(Layer):
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
        self.dim_in = dim_in
        self.embed_dim = embed_dim
        self.W = self._init_weights()
        self.optim = optim()

    def _init_weights(self):
        return np.random.randn(self.dim_in, self.embed_dim) * 0.01

    def forward(self, x):
        # Implemented efficiently with embedding lookup - expects x to be (M,T)
        embedded_vectors = self.W[x, :]
        # Shape: (M, T, embed_dim) where M is size of batch, T is number of
        # timesteps in a sequence, and embed_dim is the dimension the
        # vectors are embedded to
        self.x_inp = x
        return embedded_vectors

    def weight_tied_softmax(self, x, bay):
        """
        Method is used when the weights of this layer are tied with the
        softmax function in a model.
        """

        # x should be of shape (M, d_embed), W transposed (d_embed, d_vocab)
        # bay shape (1, d_vocab)
        logits = x.dot(self.W.T) + bay
        return logits

    def backward(self, dW, learn_rate):
        """
        Update weights with dW
        """
        self.W = self.optim(learn_rate, [self.W], [dW])
