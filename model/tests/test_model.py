""" This module contains unit tests for the sequence to sequence
network on a toy dataset to ensure everything is wired up
appropriately """
import unittest
import numpy as np
from model.sequence_to_sequence_network import SequenceToSequenceRecurrentNetwork
from model.utils import getMask
from model.utils import smoothLoss
import matplotlib.pyplot as plt

objLoss = smoothLoss()


class TestSeq2SeqRNN(unittest.TestCase):
    """ This class contains unit tests for the sequence to sequence network
    on a toy dataset """

    def testOverallAndSoftmax(self):
        """ This test ensures sure that the forward pass and backward pass
        are all wired up correctly for both the encoder and decoder, along
        with testing out beam search.Achieving ~0 loss should be trivial if
        the algorithm is coded up currently.
        """
        source_data = np.array([[0, 3, 5, 6, 4, 3, 2, 1],
                                [0, 4, 5, 6, 7, 8, 9, 2],
                                [0, 3, 1, 1, 5, 6, 8, 2]])
        target_data = np.array([[0, 3, 10, 9, 10, 6, 7, 1],
                                [0, 1, 10, 3, 10, 8, 5, 2],
                                [0, 1, 10, 4, 3, 10, 7, 2]])

        trg_vocab = {
            0: "<sos>",
            1: "<eos>",
            2: "I",
            3: "luck",
            4: "smile",
            5: "dog",
            6: "walk",
            7: "today",
            8: "had",
            9: "happy",
            10: " "
        }

        obj2 = SequenceToSequenceRecurrentNetwork(vocab_size_src=10,
                                                  vocab_size_trg=12,
                                                  dim_embed_src=120,
                                                  dim_embed_trg=512,
                                                  num_neurons_encoder=512,
                                                  num_neurons_decoder=512,
                                                  trg_map_i2c=trg_vocab,
                                                  eos_int=1,
                                                  sos_int=0)
        mask_src = getMask(source_data, 1)
        mask_trg = getMask(target_data, 1)

        inp_seq = np.array([[0, 3, 5, 6, 4, 3, 2, 1]])
        learn_rate = None
        for test_epoch in range(1000000):
            if test_epoch < 1200:
                learn_rate = 0.01
            elif test_epoch > 1200 and test_epoch < 1700:
                learn_rate = 0.005
            lossVal = obj2.forward(source_data, target_data, mask_src, mask_trg)
            lossVal = objLoss(lossVal)
            obj2._backward(learn_rate)
            output = obj2.predict(inp_seq)
            print(lossVal)
            print(output)
            print(test_epoch)
            if test_epoch % 1000 == 0:
                show_param_norms(obj2.encoder, obj2.decoder, test_epoch)


def show_param_norms(enc, dec, e):

    def getParams(obj):
        waa, wax, ba = obj.rnn_cell.Waa, obj.rnn_cell.Wax, obj.rnn_cell.ba
        return waa, wax, ba

    def getNorms(obj):
        return [np.linalg.norm(x, 2) for x in obj]

    x = ['waa', 'wax', 'ba']
    waa_enc, wax_enc, ba_enc = getParams(enc)
    waa_dec, wax_dec, ba_dec = getParams(dec)

    norms_enc = getNorms([waa_enc, wax_enc, ba_enc])
    norms_dec = getNorms([waa_dec, wax_dec, ba_dec])

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(3)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index,
                     norms_enc,
                     bar_width,
                     alpha=opacity,
                     color='b',
                     label='Encoder')

    rects2 = plt.bar(index + bar_width,
                     norms_dec,
                     bar_width,
                     alpha=opacity,
                     color='g',
                     label='Decoder')

    plt.xlabel('Parameters')
    plt.ylabel('Norms')
    plt.title('Scores by person')
    plt.xticks(index + bar_width, x)
    plt.legend()
    plt.tight_layout()
    fig.savefig("Seq2Seq Encoder and Decoder parameter norms at epoch %s" % (e))
    plt.close()


if __name__ == "__main__":
    unittest.main()
