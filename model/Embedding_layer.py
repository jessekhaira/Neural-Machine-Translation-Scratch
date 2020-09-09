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
        # cache the inputs to the layer - this embedding layer
        # can be used to tie the weights between the input layer and 
        # output layer so cache both inputs 
        self.x_inp = None 
        self.x_softmax = None 

    def _initWeights(self):
        return np.random.randn(self.dim_in, self.embed_dim)*0.01

    def _forward(self, x):
        # Implemented efficiently with embedding lookup - expects x to be (M,T) 
        embedded_vectors = self.W[x, :]
        # Shape: (M, T, embed_dim) where M is size of batch, T is number of timesteps in a sequence, and embed_dim
        # is the dimension the vectors are embedded to
        self.x_inp = x 
        return embedded_vectors

    def _weightTied_Softmax(self, x, bay):
        """
        Method is used when the weights of this layer are tied with the
        softmax function in a model.
        """

        # x should be of shape (M, d_embed), W transposed (d_embed, d_vocab) 
        # bay shape (1, d_vocab)
        self.x_softmax = x
        logits = x.dot(self.W.T) + bay
        return logits 

    def _backward(self, dW, learn_rate): 
        """
        This method carries out the backward pass for an embedding layer.
        This layer excepts dX_embed which is equivalent to: dL/dX_embedded
        for the embedded vectors produced by the layer.

        If weight tying is used in an RNN, this layer also excepts dZ_logits,
        which is: dL/dZ_logits. The total gradient update for the weights in this
        layer then becomes dL/dW = dL/dW_inp + dL/dW_softmax.
        """ 
        # dX_embed should be of shape (M, d_embed)
        # then we can get dW with (d_vocab,M).dot(M,d_embed) since input is technically a one hot encoded matrix,
        # all we have to do is add the corresponding row in dZ to the row in dW 
        # (IE: the vocab_dim the row in dZ was produced by)
        
        dW = np.zeros_like(self.W)
        ## gradient from input embedding
        np.add.at(dW, self.x, dX_embed)

        ## gradient from softmax layer if used in softmax layer
        dW_softmax = 0 
        if self.x_softmax:
            dW_softmax = dZ_logits.T.dot(dZ_logits)
        dW += dW_softmax
        self.W = self.optim(learn_rate, [self.W], [dW])
