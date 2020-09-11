# Neural Machine Translation 

-- In progress -- 

This is an implementation of a batched recurrent neural network(RNN) from scratch with NumPy. Weight tying is utilized within this model, where the input embedding layer and the linear projection before the softmax layer use the same weight matrix. 

The model was trained and tested on a dataset with the goal of translating english sequences to spanish sequences. 

The purpose of the repo was to deepen my understanding of how RNNs are used for machine translation, and how RNNs can train in batches using padding vectors and masks. 
