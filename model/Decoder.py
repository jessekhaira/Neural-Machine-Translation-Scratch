from Embedding_layer import Embedding_layer
from RNN_cell import RNN_cell
from Utils import crossEntropy
class Decoder(object):
    """
    This class represents the Decoder used in a sequence to sequence recurrent neural
    network.

    The decoders job is to recieve an encoded vector in some source language, and translate
    that to a decoded vector in a target language

    Inputs:
        -> vocab_size_trg (int): Size of the target language vocabulary
        -> dim_embed_trg (int): Size of the embeddings for the decoder
        -> optim (class): Optimizer used to train the parameters within the model
        -> num_neurons_decoder (int): Number of neurons in the decoder 
    """ 
    def __init__(self, vocab_size_trg, dim_embed_trg, num_neurons_decoder, optim):
        self.embedding_layer = Embedding_layer(vocab_size_trg, dim_embed_trg, optim)
        # for the decoder, we're going to tie the weights of the embedding layer and the linear projection
        # before softmax activation. If vocab_size_src and vocab_size_trg are same as well, its possible to tie all
        # the weights but not done here for simplicity of implementation . See: https://arxiv.org/abs/1608.05859
        self.rnn_cell = RNN_cell(dim_embed_trg, num_neurons_decoder, optim, self.embedding_layer, predict = True, costFunction = crossEntropy) 

    def __call__(self, encoded_batch, y, mask):
        loss = self.rnn_cell._forward(encoded_batch, y, mask)
        return loss 

    def _backward(self, mask, learn_rate):
        dencoded_batch= self.rnn_cell._backward(mask=mask, learn_rate=learn_rate)
        return dencoded_batch

    


