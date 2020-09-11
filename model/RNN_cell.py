from Utils import Layer
import numpy as np 
from Utils import softmax
from collections import OrderedDict
from collections import deque 

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
        space. If predicting, the weights for this layer will be used for the softmax projection layer as well. 
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
        # Ordered dict needed to cache values at every time step for backprop in the order the timesteps occur 
        self.time_cache = OrderedDict() 
        self.predict = predict
        self.costFunction = costFunction
        self.optim = optim() 


    def _initWeights(self, dim_in, num_neurons):
        # rnn cell weights - produces activations of shape (M, num_neurons)
        ba = np.zeros((1, num_neurons))
        Wax = np.random.randn(dim_in, num_neurons)
        Waa = np.random.randn(num_neurons, num_neurons)
        return Waa, Wax, ba

    def _forward(self, x, y = None, a_prev = None, mask = None):
        """
        This method carries out the forward pass through an RNN.

        Our padding idxs should not contribute to our loss, to our weight updates, and they shouldn't affect
        the activations flowing from the previous layer to the next layer. To that end, in the forward step,
        we multiply the loss vector of shape (M,) containing the losses for all the inputs at timestep T by the
        mask and zero out losses that belong to padding vectors, and divide by a batch size that ignores the 
        padding idxs. 

        We also replace the activations for vectors identified to be padding vectors by the activations obtained
        in the previous timestep. 


        M: number of examples
        T: number of timesteps 
        Inputs:
            -> x (NumPy Matrix): Matrix of integers of shape (M,T)
            -> y (NumPy Matrix | None): Matrix of integers of shape (M,T) containing the labels for each example
            in x or None
            -> a_prev (NumPy Matrix | None): Matrix of activations from the previous timestep 
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
        if not a_prev: 
            a_prev = np.zeros((x.shape[0], self.Waa.shape[0]))
        loss = 0 
        probabilities = None 
        for t in range(x.shape[1]):
            # get the sequence of vectors that occur specifically at this time step, then process them
            # all at one time 
            pre_embedded_inp_t = x[:,t]
            # Shape (M, dim_in)
            curr_timestep_x = self.embedding_layer._forward(pre_embedded_inp_t)

            curr_timestep_labels = None 
            if y:
                # Shape (M,)
                curr_timestep_labels = y[:, t]

            curr_mask = None 
            if mask:
                # Shape (M,)
                curr_mask = mask[:, t]

            # Shape (M, num_neurons)
            activation_timestep = np.tanh(curr_timestep_x.dot(self.Wax) + a_prev.dot(self.Waa) + self.ba)
            
            loss_t = None 
            probabilities_t = None 
            if self.predict:
                # Shape (M, d_vocab)
                logits = self.embedding_layer._weightTied_Softmax(activation_timestep, self.bay)
                # Shape (M, d_vocab)
                probabilities_t = softmax(logits)
                if curr_timestep_labels:
                    # Scalar 
                    loss_t = self.costFunction(curr_timestep_labels, probabilities_t, mask = mask)
                    # total loss is accumulated over every timestep
                    loss += loss_t 
            
            # Mask: If we fed in a padding vector at this time step, we want to replace that dummy activation
            # value with the activation from the previous timestep. 

            # In our get_mask function, we have vector != padding_idx, so if a value is False, that means 
            # its a padding vector 
            if curr_mask:
                activation_timestep[curr_mask == False, :] = a_prev[curr_mask == False, :]

            self.time_cache[t] = {
                "activation_timestep": activation_timestep,
                "loss_t": loss_t ,
                "probabilities_t": probabilities_t,
                "inputs_t": curr_timestep_x,
                "pre_embedded_inp_t": pre_embedded_inp_t,
                "mask_t": curr_mask,
                "curr_timestep_labels": curr_timestep_labels
            }
        
        return probabilities_t, loss, self.time_cache[x.shape[1]-1]["activation_timestep"]

    def _backward(self, learn_rate, gradient_ahead = None):
        if self.predict:
            return self._backwardPredict(learn_rate)
        else:
            self._backwardNoPredict(learn_rate, gradient_ahead)

    def _backwardPredict(self, learn_rate):
        """
        This method carries out the backward pass for a batched RNN cell that is predicting at every time step.
        As the RNN cell is unrolled for T timesteps in a batch, the gradients for the traininable parameters
        have to be summed up over all the different timesteps. 

        Used for the decoder in a seq2seq architecture. 

        Inputs:
            -> learn_rate (int): Integer representing the learning rate for the parameters in this layer  
        Outputs:
            -> NumPy matrix of shape (M,num_neurons) containing gradients of the loss function wrt to the activations at
            time step 0 
        """
        dW_embed, dWaa, dWax, dba = self._initGradients()
        dbay = np.zeros_like(self.bay)
        dActivations_ahead = None
        # backprop through time! 
        dA_prev = None
        for t, v in reversed(self.time_cache.items()):
            predictions_t = v["probabilities_t"]
            activations_t = v["activation_timestep"]
            labels_t = v["curr_timestep_labels"]
            x_t = v["inputs_t"]
            pre_embedded_inp_t = v["pre_embedded_inp_t"]
            mask_t = v["mask_t"]
            batch_size = predictions_t.shape[0] 
            # dL/dZ for the softmax layer simply -> combines dL/dA and dA/dZ in one efficient step 
            # Shape (M, dim_vocab)
            dLogits = predictions_t
            # Shape (m, dim_vocab)
            dLogits[np.arange(batch_size),labels_t] -=1
            dLogits /= batch_size

            # apply the mask if mask
            if mask_t:
                dLogits[mask==False] = 0 
            if not dA_prev:
                dActivations_ahead = np.zeros_like(activations_t)
            # Shape (dim_vocab, dim_embed)
            dW_embed_softmax = dLogits.T.dot(activations_t)
            # Shape (1, dim_vocab)
            dbay_t = np.sum(dLogits, axis=0)

            # -- ENTERING RNN CELL-- 
            # Shape (m, num_neurons == d_embed)
            dActivations = dLogits.dot(self.embedding_layer.W)
            # activations are used in two places - the next time step and for the softmax so
            # the total total activation gradient is the sum of the two
            dActivations += dActivations_ahead

            dX_embed, dWaa_t, dWax_t, dba_t, dActivations_ahead = self._getRnnGradients(dActivations=dActivations, activations_t=activations_t, x_t = x_t, pre_embedded_inp_t=pre_embedded_inp_t)
            ## Sum up all gradients over every timestep
            np.add.at(dW_embed, pre_embedded_inp_t, dX_embed)
            dW_embed += dW_embed_softmax

            dWaa += dWaa_t
            dWax += dWax_t
            dba += dba_t
            dbay += dbay_t 

            if t == 0:
                dA_prev = dActivations_ahead

        self.Waa, self.Wax, self.ba, self.bay = self.optim(learn_rate, params=[self.Waa, self.Wax, self.ba, self.bay], dparams=[dWaa, dWax, dba, dbay], grad_clip=True)
        self.embedding_layer._backward(dW_embed, learn_rate)

        return dA_prev



    def _getRnnGradients(dActivations, activations_t, x_t, pre_embedded_inp_t, mask_t = None):
        # We have to apply the mask at every timestep to prevent padding vectors from influencing weights
        # in net
        if mask_t:
            dActivations[mask_t == False] = 0 

        # backprop through tanh activation to get dL/dZ, applied elementwise over dL/dA
        # Shape (M, num_neurons)            
        dZ = (1-activations_t*activations_t)*dActivations

        # Shape (num_neurons, num_neurons)
        dWaa_t = activations_t.T.dot(dZ)
        # Shape (dim_in, num_neurons)
        dWax_t = x_t.T.dot(dZ)

        # Shape (1, num_neurons)
        dba_t = np.sum(dZ, axis=0)

        # Shape (m, num_neurons)
        dActivations_ahead = dZ.dot(self.Waa)

        # For embedding layer
        # Shape (m, d_embed == num_neurons)
        dX_embed = dActivations.dot(self.Wax)

        return dX_embed, dWaa_t, dWax_t, dba_t, dActivations_ahead




    def _initGradients():
        dW_embed = np.zeros_like(self.embedding_layer.W)
        dWaa = np.zeros_like(self.Waa)
        dWax = np.zeros_like(self.Wax)
        dba = np.zeros_like(self.ba)
        return dW_embed, dWaa, dWax, dba 
    
    def _backwardNoPredict(self, learn_rate, dActivations):
        """
        This method carries out the backward pass for a batched RNN cell not predicting at every timestep.
        As the RNN cell is unrolled for T timesteps in a batch, the gradients for the traininable parameters 
        have to be summed up over all the different timesteps. 

        Used for the encoder in a seq2seq architecture. 

        Inputs:
            -> learn_rate (int): Integer representing the learning rate for the parameters in this layer
            -> gradient_ahead (NumPy Matrix): Matrix containing the gradients of the cost function wrt to the activations
            produced at the last time in the rnn cell    
        Outputs:
            -> None 
        """
        dW_embed, dWaa, dWax, dba = self._initGradients()
        # backprop through time! 
        for t, v in reversed(self.time_cache.items()):
            predictions_t = v["probabilities_t"]
            activations_t = v["activation_timestep"]
            x_t = v["inputs_t"]
            pre_embedded_inp_t = v["pre_embedded_inp_t"]


            dX_embed, dWaa_t, dWax_t, dba_t, dActivations_ahead = self._getRnnGradients(dActivations=dActivations, activations_t=activations_t, x_t = x_t, pre_embedded_inp_t=pre_embedded_inp_t, mask = mask)
            ## Sum up all gradients over every timestep
            np.add.at(dW_embed, pre_embedded_inp_t, dX_embed)
            dWaa += dWaa_t
            dWax += dWax_t
            dba += dba_t
        
        self.Waa, self.Wax, self.ba = self.optim(learn_rate, params=[self.Waa, self.Wax, self.ba], dparams=[dWaa, dWax, dba], grad_clip=True)
        self.embedding_layer._backward(dW_embed, learn_rate)

    def _beamSearch(self, encoded, eos_int, sos_int, length_normalization, beam_width, max_seq_len):
        """
        The rnn cell utilizes the (greedy) beam search algorithm to determine the highest probability
        sentence that can be output from the architecture as described in. The variant of 
        beam search utilized within this method is described in the paper: https://arxiv.org/pdf/1409.3215.pdf.

        Inputs:
            -> encoded (NumPy vector): Vector of shape (1, num_neurons) where num_neurons indicates the number
            of neurons in the encoder
            -> eos_int (int): Integer representing the index which the end of sequence token occupies 
            -> sos_int (int): Integer representing the index which the start of sequence token occupies
            -> beam_width (int): Integer representing the size of the beam to use during the search
            -> length_normalization (int): Integer indicating the factor by which to normalize the probabilities
            of the candidate solutions by their length
            -> max_seq_len (int): Integer representing the maximum number of time steps the decoder will unfold 
            when generating a sentence 
        Outputs:
            -> NumPy array of integers containing the output sequence from the rnn
        """
        # only works for batch sizes of 1  
        assert encoded.shape == (1, self.Waa.shape[0]), "Your encoded vector produced by the encoder has to be of shape (1,num_neurons)!"
        # Array of length beam_width where array[i] = [[sequence of integers ending with eos], log_prob] 
        candidate_solutions = []

        # keep unfolding rnn until either of condtions is false:
        # beam width has shrunk to 0
        # t == max_seq_len
        t = 0 
        input_x_t = None 
        iteration_beam = beam_width
        current_beams = [[list(), 0] for b in range(beam_width)]
        while iteration_beam > 0 and t < max_seq_len:
            if t == 0:
                # single input at t== 0 of sos token 
                input_x_t = np.array([sos_int]).reshape(1,1)
                # Shape (1, dim_vocab)
                probabilities, _, encoded= self._forward(x=input_x_t, a_prev=encoded)
                # Shape (1, b)
                # Get indices of where the b highest probabilities occur and get their corresponding probabilities
                top_b_idxs = np.argsort(-probabilities)[:,:beam_width]
                top_b_logprobs = np.log(probabilities[:, top_b_idxs[0]])
                t += 1 
                for idx, logprobs, i in zip(top_b_idxs.T, top_b_logprobs.T, range(current_beams)):
                    current_beams[i][0].append(idx)
                    current_beams[i][1]+= logprobs
                    if idx == eos_int:
                        candidate_solutions.append(current_beams[i])
                        iteration_beam -= 1
                        current_beams.pop(i)
            else:
                # Shape (1, beam_width)
                # The last predicted integer for each beam forms input matrix at this timestep
                input_x_t = np.array([x[0][-1] for x in current_beams])

                # Each input vector produces a probability distribution for expected words at the
                # next timestep
                # Shape (beam_width, dim_vocab)
                probabilities, _, encoded = self._forward(x=input_x_t, a_prev=encoded)

                # you don't just want the highest prob at this timestep - you want the highest prob words when
                # considering all the log(probs) encountered at time 0,.., t-1 as well. So log the probabilities
                # at this timestep, then add them to the log(prob) of all previous timesteps and find the maximum b probs
                # along with the indices (i,j) they occurr at 
                log_probs = np.array([x[1] for x in current_beams]).reshape(beam_width,1)
                probabilities = np.log(probabilities)
                probabilities += log_probs
                # we need to determine the probability of the max words in this matrix along with the sequence they belong to 
                # get a tuple of ndarray where the row is contained in first arg, col is contained in second arg
                # returns tuple(nd.array<int>, nd.array<int>) where first arg refers to the input sequence that produced the prob
                # and second arg refers to the index of the word in the vocab 
                
                rows_cols_maxProbabilites= np.unravel_index((-probabilities).flatten().argsort()[:beam_width], probabilities.shape)
                max_sequence_logprobs = probabilities[rows_cols_maxProbabilites]

                new_sequences_t = [] 
                for seq_logprob, seq_idx, vocab_idx in zip(max_sequence_logprobs, *rows_cols_maxProbabilites):
                    curr_item = current_beams[seq_idx]
                    curr_item[0].append(vocab_idx)
                    curr_item[1] = seq_logprob

                    # if highest prob for this sequence is eos, this sequence is removed from
                    # consideration and added to final list of solutions
                    # otherwise append to end of new sequences at this time step and move on 
                    if vocab_idx == eos_int:
                        # remove the eos token from out item - pop it off
                        # dont want that included in translation 
                        curr_item.pop() 
                        iteration_beam -= 1
                        candidate_solutions.append(curr_item)
                    else:
                        new_sequences_t.append(curr_item)
                
                current_beams = new_sequences_t 
                t += 1 

        output_soln = self._getOutputBeamSearch(candidate_solutions, current_beams, t, iteration_beam, length_normalization)
        return output_soln


    def _getOutputBeamSearch(self, candidate_solutions, current_beams, t, iteration_beam, length_normalization):
        if iteration_beam != 0:
            # if we didn't finish by fufilling beam_width different beams w/ eos tokens, then 
            # just add the sequences we did get to end of candidate solutions 
            candidate_solutions.extend(candidate_solutions)
        
        # Normalize the logprobs by the length normalization factor to prevent shorter seqs from having higher log probs by default
        # x[0] -> list of integers containg sequence
        # x[1] -> sum of the log_probs over all time steps for this sequence
        normalized_seqs = list(map(lambda x:[x[0], 1/(len(x[0])**length_normalization)*x[1]],candidate_solutions))

        # Sort according to the log probs and then take the first sentence 
        most_likely_sentence = sorted(normalized_seqs, key = lambda x:x[1], reverse=True)[0][0]
        # This should be an array of integers containing 
        return most_likely_sentence

        
            

                    



                



            

        
    


