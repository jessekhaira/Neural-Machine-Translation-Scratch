""" This module contains code for a class representing a decoder
neural network, meant to be used inside a sequence to sequence
architecture"""
from model.embedding_layer import Embedding_layer
from model.recurrent_neural_network import RecurrentNeuralNetwork
from model.utils import crossEntropy
import numpy as np


class Decoder(object):
    """ This class represents the Decoder used in a sequence to sequence
    recurrent neural network.

    The decoders job is to recieve an encoded vector in some source
    language, and translate that to a decoded vector in a target language

    Attributes:
        vocab_size_trg:
            Integer representing the size of the target language vocabulary

        dim_embed_trg:
            Integer representing the size of the embeddings for the decoder

        optim:
            Object used to train the parameters within the model

        num_neurons_decoder:
            Integer representing the number of neurons in the decoder
    """

    def __init__(self, vocab_size_trg: int, dim_embed_trg: int,
                 num_neurons_decoder: int, optim: object):
        self.embedding_layer = Embedding_layer(vocab_size_trg, dim_embed_trg,
                                               optim)
        # for the decoder, we're going to tie the weights of the embedding
        # layer and the linear projection before softmax activation. If
        # vocab_size_src and vocab_size_trg are same as well, its possible to
        # tie all the weights but not done here for simplicity of
        # implementation. See: https://arxiv.org/abs/1608.05859
        self.rnn_cell = RecurrentNeuralNetwork(dim_embed_trg,
                                               num_neurons_decoder,
                                               optim,
                                               self.embedding_layer,
                                               predict=True,
                                               costFunction=crossEntropy)

    def __call__(self, encoded_batch: np.ndarray,
                 target_language_seqs: np.ndarray, mask: np.ndarray):
        """ Instantiating an object of this class and calling it will carry
        out the forward pass for the Decoder.

        The Decoder is fed in an encoded tensor from the decoder, which becomes
        the prev activations for the unit.

        In contrast to the encoder, in which you just slice out all the vectors
        occurring at a given time step and feed them in and process them in
        parallel with no labels, the Decoder has both x<t> and y<t>. The x<t>
        is the label at the previous time step, and the y<t> is the label for
        the next time step. Regardless of what we predict at a given time step,
        we feed the correct label in at the next time step.

        If your calling the Decoder directly, this assumes it is being used for
        training. If predictions are wanted, use the _beamSearch method instead,
        which carries out prediction with the decoder when given an encoded
        sentence.

        M - number of examples
        T - number of timesteps
        num_neurons - number of neurons in the encoder

        Args:
            encoded_batch:
                Numpy array of shape (M, num_neurons) representing the output
                from the encoder algorithm

            target_language_seqs:
                Numpy array of shape (M, T) representing the inputs and outputs
                meant to be fed into the decoder algorithm

            mask:
                Numpy array of shape (M,T) indicating the vectors that are
                padding vectors and do not contribute to the loss or gradients
                that are backpropagated

        Returns:
            Floating point value representing the training loss on this batch
        """
        # Shape (M, T-1)
        x_matrix = target_language_seqs[:, :-1]
        y_matrix = target_language_seqs[:, 1:]
        _, loss, _ = self.rnn_cell._forward(x=x_matrix,
                                            y=y_matrix,
                                            a_prev=encoded_batch,
                                            mask=mask)
        return loss

    def _backward(self, learn_rate):
        dencoded_batch = self.rnn_cell._backward(learn_rate=learn_rate)
        return dencoded_batch

    def beamSearch(self, encoded, eos_int, sos_int, length_normalization,
                   beam_width, max_seq_len):
        return self.rnn_cell._beamSearch(encoded, eos_int, sos_int,
                                         length_normalization, beam_width,
                                         max_seq_len)
