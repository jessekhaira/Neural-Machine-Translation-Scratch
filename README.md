# Neural Machine Translation 

## Description  

This repo contains code for a sequence to sequence model implemented from scratch in NumPy. The model implemented contains two neural networks: an encoder network and a decoder network. Within this model, a plain RNN cell was used for both of these networks in the configuration shown below. 

<img src="./images/encoder_decoder.jpg">

In addition to the primary model, I also implemented: 
* An efficient embedding layer(no one hot encoded vectors!)
* Weight tying between the embedding layer and the output layer in the decoder 
* Beam Search 

The purpose of the repo was to deepen my understanding of RNNs, learn about specific implementation details of how RNNs train in batches (padding, masks, etc), and also gain a better understanding of natural language processing in general.  

The core model is contained within one file: model/Seq2Seq_rnn.py. This file contains the overall sequence to sequence model definition, along with a useful method to train the model that expects a data loader containing batches of source sequences and target sequences (can be easily done with TorchText!). Example usage is shown within the Jupyter notebook attached. 
