from model.Embedding_layer import Embedding_layer 
from model.RNN_cell import RNN_cell

class Encoder(object):
    """
    This class represents the Encoder object used within a sequence to sequence
    recurrent neural network. The job of this architecture is to take in an input
    sequence in a source language, and produce an embedded vector to feed into a 
    decoder network. 

    Inputs:
        -> vocab_size_src (int): Size of the source language vocabulary
        -> dim_embed_src (int): Size of the embeddings for the encoder 
        -> num_neurons_encoder (int): Number of neurons in the encoder
        -> optim(class): Optimizer used to train the parameters within the model
    """
    def __init__(self, vocab_size_src, dim_embed_src, num_neurons_encoder, optim):
        self.embedding_layer = Embedding_layer(dim_in = vocab_size_src, embed_dim=dim_embed_src, optim=optim)
        self.rnn_cell = RNN_cell(dim_in = dim_embed_src, num_neurons = num_neurons_encoder, optim = optim, embedding_layer = self.embedding_layer)
    
    def __call__(self, x, mask):
        # Shape: (M, num_neurons_encoder) containing activations that hopefully encode
        # all the words in every sequence in the source language well
        _ ,_ ,encoded_matrix = self.rnn_cell._forward(x, mask=mask)
        return encoded_matrix


    def _backward(self, dA_encodedVector, learn_rate):
        # Shape: (M, dim_embed_src)
        self.rnn_cell._backward(gradient_ahead=dA_encodedVector, learn_rate=learn_rate)    
        


        