""" This module contains code for a class that represents a vanilla
recurrent neural network """
from model.utils import Layer
from model.embedding_layer import EmbeddingLayer
from model.utils import softmax
from collections import OrderedDict
from collections import deque
from typing import Union, Callable, Tuple
import copy
import numpy as np


class RecurrentNeuralNetwork(Layer):
    """
    This class represents a basic recurrent neural network cell
    utilizing the tanh activation function.

    Attributes:
        dim_in:
            Integer representing the number of features the input to the cell
            has

        num_neurons:
            Integer representing the number of features to be learned at
            this layer

        optim:
            Object representing the optimizer used to learn the parameters
            during training

        embedding_layer:
            Object used to embed the input vectors into a lower dimensional
            space. If predicting, the weights for this layer will be used for
            the softmax projection layer as well.

        predict:
            Boolean indicating whether or not this cell should feed its
            output to a softmax classifier. Needed for decoder in
            seq2seq model but encoder is not predicting

        costFunction:
            Function representing the cost when training, or None if there
            is no cost function. Needed for decoder in seq2seq model, but
            encoder has no associated cost.
    """

    def __init__(self,
                 dim_in: int,
                 num_neurons: int,
                 optim: object,
                 embedding_layer: EmbeddingLayer,
                 predict: bool = False,
                 costFunction: Union[Callable[[np.ndarray, np.ndarray], float],
                                     None] = None):
        self.embedding_layer = embedding_layer
        self.Waa, self.Wax, self.ba = self._init_weights(dim_in, num_neurons)
        self.predict = predict
        if self.predict:
            self.bay = np.zeros((1, self.embedding_layer.W.shape[0]))
        # Ordered dict needed to cache values at every time step for backprop in
        # the order the timesteps occur
        self.time_cache = OrderedDict()
        self.predict = predict
        self.costFunction = costFunction
        self.optim = optim()

    def _init_weights(
            self, dim_in: int,
            num_neurons: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # rnn cell weights - produces activations of shape (M, num_neurons)
        ba = np.zeros((1, num_neurons))
        wax = np.random.randn(dim_in, num_neurons) * 0.01
        waa = np.random.randn(num_neurons, num_neurons) * 0.01
        return waa, wax, ba

    def forward(
        self,
        x: np.ndarray,
        y: Union[np.ndarray, None] = None,
        a_prev: Union[np.ndarray, None] = None,
        mask: Union[np.ndarray, None] = None
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """ This method carries out the forward pass through an RNN.

        Our padding idxs should not contribute to our loss, to our weight
        updates, and they shouldn't affect the activations flowing from the
        previous layer to the next layer. To that end, in the forward step,
        we multiply the loss vector of shape (M,) containing the losses for
        all the inputs at timestep T by the mask and zero out losses that belong
        to padding vectors, and divide by a batch size that ignores the padding
        idxs.

        We also replace the activations for vectors identified to be padding
        vectors by the activations obtained in the previous timestep.


        M: number of examples
        T: number of timesteps

        Args:
            x:
                Numpy array of shape (M,T) representing the text sequences to
                train on

            y:
                Numpy array of integers of shape (M,T) containing the labels
                for each example in x or None

            a_prev:
                Numpy array of activations from the previous timestep, or None

            mask:
                Numpy array of integers of shape (M,T) indicating which indices
                belong to padding vectors or None.

                So mask[0] corresponds to a sequence of integers for the first
                example, and if a value is False within this row, that means the
                vector this idx corresponds to is a padding vector.

                During training, we will be going sequentially from t = 0 to
                t = T, for which we will slice out from y and the mask an array
                of shape (M,) of the labels for the current step (if predicting)
                and whether each vector in the current sequence is a padding
                vector or not.

        Returns:
            A numpy matrix representing the activations from this layer, or a
            floating point value representing the loss, depending on whether
            rnn cell is used to predict or used to encode.
        """
        # shape (M, num_neurons)
        a_prev = a_prev if a_prev is not None else np.zeros(
            (x.shape[0], self.Waa.shape[0]))
        loss = 0
        for t in range(x.shape[1]):
            # get the sequence of vectors that occur specifically at this
            # time step, then process them all at one time
            pre_embedded_inp_t = x[:, t]
            # Shape (M, dim_in)
            curr_timestep_x = self.embedding_layer.forward(pre_embedded_inp_t)

            curr_timestep_labels = None if y is None else y[:, t]

            curr_mask = None if mask is None else mask[:, t]

            # Shape (M, num_neurons)
            activation_timestep = np.tanh(
                curr_timestep_x.dot(self.Wax) + a_prev.dot(self.Waa) + self.ba)

            loss_t = None
            probabilities_t = None
            if self.predict:
                # Shape (M, d_vocab)
                logits = self.embedding_layer._weightTied_Softmax(
                    activation_timestep, self.bay)
                # Shape (M, d_vocab)
                probabilities_t = softmax(logits)
                if curr_timestep_labels is not None:
                    loss_t = self.costFunction(curr_timestep_labels,
                                               probabilities_t,
                                               mask=curr_mask)
                    loss += loss_t

            # Mask: If we fed in a padding vector at this time step, we want
            # to replace that dummy activation value with the activation from
            # the previous timestep.

            # In our get_mask function, we have vector != padding_idx, so if a
            # value is False, that means its a padding vector
            if curr_mask is not None:
                activation_timestep[curr_mask == False, :] = a_prev[curr_mask ==
                                                                    False, :]

            self.time_cache[t] = {
                "activation_timestep": activation_timestep,
                "loss_t": loss_t,
                "probabilities_t": probabilities_t,
                "inputs_t": curr_timestep_x,
                "pre_embedded_inp_t": pre_embedded_inp_t,
                "mask_t": curr_mask,
                "curr_timestep_labels": curr_timestep_labels
            }
            # pass on current time step activations to next step
            a_prev = activation_timestep

        return probabilities_t, loss, self.time_cache[x.shape[1] -
                                                      1]["activation_timestep"]

    def _backward(self, learn_rate, gradient_ahead=None):
        if self.predict:
            return self._backward_predict(learn_rate)
        else:
            self._backwardNoPredict(learn_rate, gradient_ahead)

    def _backward_predict(self, learn_rate: float) -> np.ndarray:
        """ This method carries out the backward pass for a batched RNN
        cell that is predicting at every time step. As the RNN cell is
        unrolled for T timesteps in a batch, the gradients for the
        traininable parameters have to be summed up over all the different
        timesteps.

        Used for the decoder in a seq2seq architecture.

        Args:
            learn_rate:
                Floating point value representing the learning rate for the
                parameters in this layer

        Returns:
            NumPy array of shape (M, num_neurons) containing gradients of the
            loss function with respect to the activations at time step 0
        """
        dW_embed, dWaa, dWax, dba = self._initGradients()
        dbay = np.zeros_like(self.bay)
        dActivations_ahead = None
        toEncoder = None
        for t, v in reversed(self.time_cache.items()):
            predictions_t = v["probabilities_t"]
            activations_t = v["activation_timestep"]
            labels_t = v["curr_timestep_labels"]
            x_t = v["inputs_t"]
            pre_embedded_inp_t = v["pre_embedded_inp_t"]
            mask_t = v["mask_t"]
            # batch size shouldn't include vectors that are padding
            batch_size = predictions_t.shape[0] if mask_t is None else np.sum(
                mask_t)
            # dL/dZ for the softmax layer simply -> combines dL/dA and dA/dZ in one efficient step
            # Shape (M, dim_vocab)
            dLogits = predictions_t
            # Shape (m, dim_vocab)
            dLogits[np.arange(dLogits.shape[0]), labels_t] -= 1
            dLogits /= batch_size

            # apply the mask if mask
            # dont let gradients for padding vectors affect the weights or biases at all
            if mask_t is not None:
                dLogits[mask_t == False] = 0
            if dActivations_ahead is None:
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

            dX_embed, dWaa_t, dWax_t, dba_t, dActivations_ahead = self._getRnnGradients(
                dActivations=dActivations,
                activations_t=activations_t,
                x_t=x_t,
                pre_embedded_inp_t=pre_embedded_inp_t)
            ## Sum up all gradients over every timestep
            np.add.at(dW_embed, pre_embedded_inp_t, dX_embed)
            dW_embed += dW_embed_softmax

            dWaa += dWaa_t
            dWax += dWax_t
            dba += dba_t
            dbay += dbay_t

            if t == 0:
                toEncoder = dActivations_ahead

        self.Waa, self.Wax, self.ba, self.bay = self.optim(
            learn_rate,
            params=[self.Waa, self.Wax, self.ba, self.bay],
            dparams=[dWaa, dWax, dba, dbay],
            grad_clip=True)
        self.embedding_layer._backward(dW_embed, learn_rate)

        return toEncoder

    def _getRnnGradients(self,
                         dActivations,
                         activations_t,
                         x_t,
                         pre_embedded_inp_t,
                         mask_t=None):
        # for padding vectors, don't update parameters
        # and for dActivations_behind, let the gradients from the original dActivations
        # flow unimpeded for pad vectors
        orig_dActivations = copy.copy(dActivations)
        if mask_t is not None:
            dActivations[mask_t == False] = 0

        # backprop through tanh activation to get dL/dZ, applied elementwise over dL/dA
        # Shape (M, num_neurons)
        dZ = (1 - activations_t * activations_t) * dActivations

        # Shape (num_neurons, num_neurons)
        dWaa_t = activations_t.T.dot(dZ)
        # Shape (dim_in, num_neurons)
        dWax_t = x_t.T.dot(dZ)

        # Shape (1, num_neurons)
        dba_t = np.sum(dZ, axis=0)

        # Shape (m, num_neurons)
        dActivations_behind = dZ.dot(self.Waa)

        # For embedding layer
        # Shape (m, d_embed == num_neurons)
        dX_embed = dActivations.dot(self.Wax.T)

        # let gradients flow unimpeded for pad vectors
        dActivations_behind[mask_t == False] = orig_dActivations[mask_t ==
                                                                 False]

        return dX_embed, dWaa_t, dWax_t, dba_t, dActivations_behind

    def _initGradients(self):
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
            mask_t = v["mask_t"]

            dX_embed, dWaa_t, dWax_t, dba_t, dActivations = self._getRnnGradients(
                dActivations=dActivations,
                activations_t=activations_t,
                x_t=x_t,
                pre_embedded_inp_t=pre_embedded_inp_t,
                mask_t=mask_t)
            ## Sum up all gradients over every timestep
            np.add.at(dW_embed, pre_embedded_inp_t, dX_embed)
            dWaa += dWaa_t
            dWax += dWax_t
            dba += dba_t

        self.Waa, self.Wax, self.ba = self.optim(
            learn_rate,
            params=[self.Waa, self.Wax, self.ba],
            dparams=[dWaa, dWax, dba],
            grad_clip=True)
        self.embedding_layer._backward(dW_embed, learn_rate)

    def _beamSearch(self, encoded, eos_int, sos_int, length_normalization,
                    beam_width, max_seq_len):
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
        assert encoded.shape == (
            1, self.Waa.shape[0]
        ), "Your encoded vector produced by the encoder has to be of shape (1,num_neurons)!"
        # Array of length beam_width where array[i] = [[sequence of integers ending with eos], log_prob]
        candidate_solutions = []
        # keep unfolding rnn until either of condtions is false:
        # beam width has shrunk to 0
        # t == max_seq_len
        t = 0
        input_x_t = None
        iteration_beam = beam_width
        current_beams = {}
        while iteration_beam > 0 and t < max_seq_len:
            if t == 0:
                # single input at t== 0 of sos token
                input_x_t = np.array([sos_int]).reshape(1, 1)
                # Shape (1, dim_vocab)
                probabilities, _, encoded = self.forward(x=input_x_t,
                                                         a_prev=encoded)
                # Shape (1, b)
                # Get indices of where the b highest probabilities occur and get their corresponding probabilities
                top_b_idxs = np.argsort(-probabilities)[:, :beam_width]
                top_b_logprobs = np.log(probabilities[:, top_b_idxs[0]])
                t += 1
                curr_beam = 0
                for idx, logprobs in zip(top_b_idxs.T, top_b_logprobs.T):
                    idx = int(idx)
                    curr_item = [[], 0]
                    curr_item[0].append(int(idx))
                    curr_item[1] += logprobs
                    if idx == eos_int:
                        iteration_beam -= 1
                    # every beam holds two things: item including seq it has produced and
                    # its log prob, along with the corresponding encoded vector belonging to it
                    else:
                        current_beams[curr_beam] = [curr_item, encoded]
                        curr_beam += 1
            else:
                # Shape (1, beam_width)
                # The last predicted integer for each beam forms input matrix at this timestep
                t += 1
                input_x_t = np.array(
                    [v[0][0][-1] for v in current_beams.values()])
                input_x_t = input_x_t.reshape(-1, 1)

                encoded = np.empty(
                    (iteration_beam, current_beams[0][1].shape[1]))
                encoded[:] = [v[1] for v in current_beams.values()]
                assert input_x_t.shape[0] == (
                    encoded.shape[0]
                ) or encoded.shape[
                    0] == 1, "Shape mismatch! Your passing in an unequal number of input arguments compared to encoded arguments!"
                probabilities, _, encoded = self.forward(x=input_x_t,
                                                         a_prev=encoded)

                # you don't just want the highest prob at this timestep - you want the highest prob words when
                # considering all the log(probs) encountered at time 0,.., t-1 as well. So log the probabilities
                # at this timestep, then add them to the log(prob) of all previous timesteps and find the maximum b probs
                # along with the indices (i,j) they occurr at
                log_probs = np.array([v[0][1] for v in current_beams.values()
                                     ]).reshape(iteration_beam, 1)
                probabilities = np.log(probabilities)
                probabilities += log_probs
                # we need to determine the probability of the max words in this matrix along with the sequence they belong to
                # get a tuple of ndarray where the row is contained in first arg, col is contained in second arg
                # returns tuple(nd.array<int>, nd.array<int>) where first arg refers to the input sequence that produced the prob
                # and second arg refers to the index of the word in the vocab

                rows_cols_maxProbabilites = np.unravel_index(
                    (-probabilities).flatten().argsort()[:iteration_beam],
                    probabilities.shape)
                max_sequence_logprobs = probabilities[rows_cols_maxProbabilites]
                curr_beam = 0
                new_beams = {}
                for seq_logprob, seq_idx, vocab_idx in zip(
                        max_sequence_logprobs, *rows_cols_maxProbabilites):
                    # if highest prob for this sequence is eos, this sequence is removed from
                    # consideration and added to final list of solutions
                    # otherwise append to end of new sequences at this time step and move on
                    if vocab_idx == eos_int:
                        iteration_beam -= 1
                        # Useful:
                        # current_beams[seq_idx] = [item, encoded]
                        # item = [[seq], log_prob]
                        candidate_solutions.append(current_beams[seq_idx][0])
                    else:
                        # build the new item off the old item
                        curr_item = copy.deepcopy(current_beams[seq_idx])
                        # add the vocab idx word to the end of seq
                        curr_item[0][0].append(vocab_idx)
                        curr_item[0][1] = seq_logprob
                        curr_item[1] = encoded[seq_idx:seq_idx + 1]
                        # use curr_beam to assign new position of new index for new timestep
                        # which may or may not be equal to the original seq_idx depending of if indices
                        # have been popped off or if we are repeating same indice multiple times
                        new_beams[curr_beam] = curr_item
                        curr_beam += 1

                current_beams = new_beams
        output_soln = self._getOutputBeamSearch(candidate_solutions,
                                                current_beams, t,
                                                iteration_beam,
                                                length_normalization)
        return output_soln

    def _getOutputBeamSearch(self, candidate_solutions, current_beams, t,
                             iteration_beam, length_normalization):
        if iteration_beam != 0:
            # if beams_not_eos remains True after t time steps, then we add the seq to the remaining_seqs
            remaining_seqs = [v[0] for v in current_beams.values()]
            candidate_solutions.extend(remaining_seqs)

        # Normalize the logprobs by the length normalization factor to prevent shorter seqs from having higher log probs by default
        # x[0] -> list of integers containg sequence
        # x[1] -> sum of the log_probs over all time steps for this sequence
        normalized_seqs = list(
            map(lambda x: [x[0], 1 / (len(x[0])**length_normalization) * x[1]],
                candidate_solutions))

        # Sort according to the log probs and then take the first sentence
        most_likely_sentence = sorted(normalized_seqs,
                                      key=lambda x: x[1],
                                      reverse=True)[0][0]
        return most_likely_sentence
