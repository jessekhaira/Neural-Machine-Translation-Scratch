from Utils import Layer
import numpy as np 
from Utils import softmax

class RNN_cell(Layer):
    """
    This class represents a basic recurrent neural network cell utilizing the tanh
    activation function.  

    Inputs:
        -> dim_in (int): Integer representing the number of features the input to the cell has 
        -> num_neurons (int): Integer representing the number of features to be learned at 
        this layer
        -> optim (class): Optimizer used to learn the parameters during training 
        -> embedding_layer (Embedding_layer): Object used to embed the input vectors into a lower dimensional
        space. If predicting, the weights for this layer will be used in the output layer as well. 
        -> predict (boolean): Boolean indicating whether or not this cell should feed its
        output to a softmax classifier. Needed for decoder in seq2seq model but encoder is not predicting 
        -> costFunction (Function|None): Function representing the cost when training. Needed for decoder
        in seq2seq model but encoder has no associated cost. 
    """
    def __init__(self, dim_in, num_neurons, optim, embedding_layer, predict=False, costFunction=None):
        self.embedding_layer = embedding_layer
        self.Waa, self.Wax, self.ba = self._initWeights(dim_in,num_neurons) 
        if self.predict:
            self.bay = np.zeros((1, self.embedding_layer.W.shape[0]))
        # Hashtable needed to cache values at every time step for backprop
        self.time_cache = {} 
        self.predict = predict
        self.costFunction = costFunction


    def _initWeights(self, dim_in, num_neurons):
        # rnn cell weights - produces activations of shape (M, num_neurons)
        ba = np.zeros((1, num_neurons))
        Wax = np.random.randn(dim_in, num_neurons)
        Waa = np.random.randn(num_neurons, num_neurons)
        return Waa, Wax, ba

    def _forward(self, x, y = None, mask = None):
        """
        This method carries out the forward pass through an RNN.

        Our padding idxs should not contribute to our loss, to our weight updates, and they shouldn't affect
        the activations flowing from the previous layer to the next layer. To that end, in the forward step,
        we multiply the loss vector of shape (M,) containing the losses for all the inputs at timestep T by the
        mask and zero out losses that belong to padding vectors, and divide by a batch size that ignores the 
        padding idxs. 

        We also replace the activations for vectors identified to be padding vectors by the activations obtained
        in the previous timestep. 


        M -> number of examples
        T -> number of timesteps 
        Inputs:
            -> x (NumPy Matrix): Matrix of integers of shape (M,T) where any row indicates a single sequence 
            -> y (NumPy Matrix | None): Matrix of integers of shape (M,T) containing the labels for each example
            in x or None if training an encoder. 
            -> mask (NumPy Matrix | None): Matrix of integers of shape (M,T) indicating which indices belong to padding
            vectors or None. 
            
            So mask[0] corresponds to a sequence of integers for the first example, and if a value is False
            within this row, that means the vector this idx corresponds to is a padding vector. 

            During training, we will be going sequentially from t = 0 to t = T, for which we will slice out from y and
            the mask an array of shape (M,) of the labels for the current step (if predicting) and whether each vector 
            in the current sequence is a padding vector or not. 
        
        Outputs:
            -> Integer representing the loss, or an encoded tensor. Depends on whether rnn cell is used to predict or
            used to encode. 
        """
        # We can't process all the vectors at every time step in parallel which is a drawback to this architecture
        # and a big reason why transformers are more efficient 

        # shape (M, num_neurons)
        a_prev = np.zeros((x.shape[0], self.Waa.shape[0]))
        loss = 0 
        probabilities = None 
        for t in range(x.shape[1]):
            # Shape (M, T, dim_in)
            embedded_tensor = self.embedding_layer._forward(x)
            # Slice out all the vectors at the same time step, and process them in parallel
            # want all the features for every single example, but only a specific timestep 

            # Shape (M, dim_in)
            curr_timestep_x = embedded_tensor[:, t, :]

            curr_timestep_labels = None 
            if y:
                # Shape (M,)
                curr_timestep_labels = y[:, t]

            curr_mask = None 
            if mask:
                # Shape (M,)
                curr_mask = mask[:, t]

            # Shape (M, num_neurons)
            activation_timestep = curr_timestep_x.dot(self.Wax) + a_prev.dot(self.Waa) + self.ba
            
            loss_t = None 
            probabilities_t = None 
            if self.predict:
                # Shape (M, d_vocab)
                logits = self.embedding_layer._weightTied_Softmax(activation_timestep, self.bay)
                # Shape (M, d_vocab)
                probabilities_t = softmax(logits)
                loss_t = self.costFunction(curr_timestep_labels, probabilities_t, mask = mask)
                # total loss is accumulated over every timestep
                loss += loss_t 
            
            # Mask: If we fed in a padding vector at this time step, we want to replace that dummy activation
            # value with the activation from the previous timestep. 

            # In our get_mask function, we have vector != padding_idx, so if a value is False, that means 
            # its a padding vector 
            activation_timestep[curr_mask == False, :] = a_prev[curr_mask == False, :]

            self.time_cache[t] = {
                "activation_timestep": activation_timestep,
                "loss_t": loss_t ,
                "probabilities_t": probabilities_t,
                "mask_t": curr_mask,
                "curr_timestep_labels": curr_timestep_labels
            }
        
        return loss if self.predict else self.time_cache[x.shape[1]-1]["activation_timestep"]


    def _backwardPredict(self, learn_rate, mask = None):
        """
        This method carries out the backward pass for a RNN cell trained in batches. As the RNN cell
        is unrolled for T timesteps in a batch, the gradients for the traininable parameters
        have to be summed up over all the different timesteps. 

        Inputs:
            -> learn_rate (int): Integer representing the learning rate for the parameters in this layer
            -> mask (NumPy Matrix | None): Matrix of integers of shape (M,T) indicating which indices belong to padding
            vectors or None.       
            -> gradient_ahead (NumPy Matrix | None): 
        
        Outputs:
            -> None | 
        """
        #
        dW_embed = np.zeros_like(self.embedding_layer.W)
        dWaa = np.zeros_like()

        pass 
    
    def _backwardNoPredict(self, learn_rate, mask = None):
        """
        """
        
        

