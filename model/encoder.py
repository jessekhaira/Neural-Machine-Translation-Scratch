""" This module contains code for a class that represents the encoder
algorithm meant to be used within a sequence to sequence network for
machine translation """
from model.embedding_layer import Embedding_layer
from model.recurrent_neural_network import RecurrentNeuralNetwork


class Encoder(object):
    """ This class represents the Encoder object used within a sequence to
    sequence recurrent neural network. The job of this architecture is to
    take in an input sequence in a source language, and produce an embedded
    vector to feed into a decoder network.

    Attributes:
        vocab_size_src:
            Integer representing the size of the source language vocabulary

        dim_embed_src:
            Integer representing the size of the embeddings for the encoder

        num_neurons_encoder:
            Integer representing the number of neurons in the encoder

        optim:
            Object representing the optimization algorithm to use to learn the
            parameters for the algorithm
    """

    def __init__(self, vocab_size_src: int, dim_embed_src: int,
                 num_neurons_encoder: int, optim: object):
        self.embedding_layer = Embedding_layer(dim_in=vocab_size_src,
                                               embed_dim=dim_embed_src,
                                               optim=optim)
        self.rnn_cell = RecurrentNeuralNetwork(
            dim_in=dim_embed_src,
            num_neurons=num_neurons_encoder,
            optim=optim,
            embedding_layer=self.embedding_layer)

    def __call__(self, x, mask):
        # Shape: (M, num_neurons_encoder) containing activations that hopefully encode
        # all the words in every sequence in the source language well
        _, _, encoded_matrix = self.rnn_cell._forward(x, mask=mask)
        return encoded_matrix

    def _backward(self, da_encoded_vector, learn_rate):
        # Shape: (M, dim_embed_src)
        self.rnn_cell._backward(gradient_ahead=da_encoded_vector,
                                learn_rate=learn_rate)
