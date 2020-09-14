# Neural Machine Translation 

-- In progress -- 


This is an implementation of a batched recurrent neural network(RNN) for the task of machine translation with just NumPy. An encoder network and a decoder network were used in the following configuration for the task: 

<img src="./encoder_decoder.jpg">


In addition to the RNN, I also implemented:
* An efficient embedding layer(no one hot encoded vectors!)
* Weight tying between the embedding layer and the output layer
* Shrinking beam search to maximize the probability of the sentence output 

The purpose of the repo was to deepen my understanding of RNNs in general, and some specific implementation details of how RNNs train in batches (padding, masks, etc). 

