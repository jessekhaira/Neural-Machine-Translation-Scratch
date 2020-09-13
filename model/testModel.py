import unittest
from Seq2Seq_rnn import Seq2Seq_rnn
import numpy as np 
from Utils import getMask
from Utils import crossEntropy
import matplotlib.pyplot as plt 

class tests(unittest.TestCase):
    def testOverallAndSoftmax(self):
        source_data = np.array([[0,3,5,6,4,3,2,1], [0,4,5,6,7,8,9,2], [0,3,1,1,5,6,8,2]])
        target_data = np.array([[0,3,9,6,7,1], [0,1,3,10,5,2], [0,1,4,3,7,2]])

        trg_vocab = {
            0: "<sos>",
            1: "<pad>",
            2: "<eos>",
            3: "I",
            4: "luck",
            5: "smile",
            6: "dog",
            7: "walk",
            8: "today",
            9: "had",
            10: "happy",
            11: " "
        }

        obj2 = Seq2Seq_rnn(
            vocab_size_src=10,
            vocab_size_trg=12, 
            dim_embed_src=120,
            dim_embed_trg=512,
            num_neurons_encoder=512,
            num_neurons_decoder=512,
            trg_map_i2c=trg_vocab,
            eos_int=2,
            sos_int=0
        )
        mask_src = getMask(source_data, 1)
        mask_trg = getMask(target_data, 1)

        inp_seq = np.array([[4,5,2,3,1]])
        learn_rate= None
        # This makes sure that the forward pass and backward pass are all wired up correctly
        # since achieving ~0 loss on 3 examples should be trivial if they are
        # and it is! 
        for test_epoch in range(1000000):
            if test_epoch < 1000:
                learn_rate = 0.01
            else:
                learn_rate = 0.001
            lossVal = obj2._forward(source_data, target_data, mask_src, mask_trg, crossEntropy)
            obj2._backward(learn_rate)
            output = obj2.predict(inp_seq)
            print(lossVal)
            print(output)
            print(test_epoch)
            if test_epoch % 1000 == 0:
                show_param_norms(obj2.Encoder, obj2.Decoder, test_epoch)    

def show_param_norms(enc, dec, e):

    def getParams(obj):
        waa, wax, ba= obj.rnn_cell.Waa, obj.rnn_cell.Wax, obj.rnn_cell.ba
        return waa, wax, ba

    def getNorms(obj):
        return [np.linalg.norm(x,2) for x in obj]
    
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

    rects1 = plt.bar(index, norms_enc, bar_width,
    alpha=opacity,
    color='b',
    label='Encoder')

    rects2 = plt.bar(index + bar_width, norms_dec, bar_width,
    alpha=opacity,
    color='g',
    label='Decoder')

    plt.xlabel('Parameters')
    plt.ylabel('Norms')
    plt.title('Scores by person')
    plt.xticks(index + bar_width, x)
    plt.legend()
    plt.tight_layout()
    fig.savefig("Seq2Seq Encoder and Decoder parameter norms at epoch %s"%(e))
    plt.close()

if __name__ == "__main__":
    unittest.main() 