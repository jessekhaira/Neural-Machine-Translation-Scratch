from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Utils import GradientDescentMomentum
from model.Utils import getMask
from model.Utils import smoothLoss
import torch
import numpy as np
from tqdm import tqdm


class SequenceToSequenceRecurrentNetwork(object):
    """
    This class represents a sequence to sequence model used for the task of machine translation, trained
    in batches. 

    The architecture is as follows: we feed in an input sequence in a source language to the encoder, and get an encoded
    vector from the encoder. We feed that encoded vector into a decoder network which will output predictions
    of the word it thinks is the most likely in the target language. 

    Inputs:
        -> eos_int (int): Integer representing the index which the end of sequence token occupies in the target vocab
        -> sos_int (int): Integer representing the index which the start of sequence token occupies in the target vocab
        -> vocab_size_src (int): Size of the source language vocabulary
        -> vocab_size_trg (int): Size of the target language vocabulary
        -> dim_embed_src (int): Size of the embeddings for the encoder 
        -> src_map_i2c (HashMap<Integer, Token> | None): Mapping from integers to tokens for source language
        -> trg_map_i2c (HashMap<Integer, Token> | None): Mapping from integers to tokens for target language 
        -> dim_embed_trg (int): Size of the embeddings for the decoder. Kep t
        -> num_neurons_encoder (int): Number of neurons in the encoder
        -> optim(class): Optimizer used to train the parameters within the model
        -> num_neurons_decoder (int): Number of neurons in the decoder 
    """

    def __init__(self,
                 vocab_size_src,
                 vocab_size_trg,
                 eos_int,
                 sos_int,
                 dim_embed_src=512,
                 src_map_i2c=None,
                 trg_map_i2c=None,
                 dim_embed_trg=512,
                 num_neurons_encoder=512,
                 num_neurons_decoder=512,
                 optim=GradientDescentMomentum):
        assert dim_embed_trg == num_neurons_decoder, "For weight tying, the number of neurons in the decoder has to be the same as the number of dimensions in the embedding"
        # These don't have to be equal. If they aren't, you will need an additional weight matrix after the encoder to project down or up to the
        # dimensionality of the decoder, adding extra complexity. Kept symmetric for simplicities sake
        assert num_neurons_decoder == num_neurons_encoder, "Currently this model only supports symmetric decoders and encoders"
        self.src_dim = vocab_size_src
        self.trg_map_i2c = trg_map_i2c
        self.src_map_i2c = src_map_i2c
        self.optim = optim
        self.Encoder = Encoder(vocab_size_src, dim_embed_src,
                               num_neurons_encoder, optim)
        self.Decoder = Decoder(vocab_size_trg, dim_embed_trg,
                               num_neurons_decoder, optim)
        self.eos_int = eos_int
        self.sos_int = sos_int

    def _forward(self, x, y, mask_src, mask_trg):
        """
        This method computes the forward pass through the sequence to sequence model, 
        producing a loss value and a predictions vector.

        Inputs:
            -> x (NumPy matrix): Matrix of shape (M,N) where M is the batch size and 
            N is the sequence length
            -> y (NumPy vector): Vector of integers containing labels for all the examples
            in the batch
            -> mask_src (NumPy Matrix): Matrix indicating which timesteps belong to padding idxs for the encoder
            -> mask_trg (NumPy Matrix): Matrix indicating which timesteps belong to padding idxs for the decoder
            -> loss_func (Function): Function to minimize during training 
        
        Outputs:
            -> loss (int): Integer representing loss on batch 
            -> predictions (NumPy matrix): Matrix containing the probabilistic predictions for 
            every vector in the batch 
        """
        encoded_batch = self.Encoder(x, mask_src)
        loss = self.Decoder(encoded_batch, y, mask_trg)
        return loss

    def _backward(self, learn_rate):
        """
        This method computes the backward pass through the network.
        """
        # vector containing the gradient dL/dA for the encoded vector produced at last time
        # step for encoder
        dA_encodedVector = self.Decoder._backward(learn_rate)
        self.Encoder._backward(dA_encodedVector, learn_rate)

    def train(self,
              data_loader,
              batch_size,
              src_name,
              trg_name,
              padding_idx,
              valid_loader=None,
              learn_rate=0.1,
              learning_schedule=None,
              num_epochs=100,
              verbose=1,
              _testing=None):
        """
        This method is used to train the seq2seq rnn model in batches. This method expects the data
        to come in as an iterator, and the batches to come in as an object like TorchText's 
        bucket iterator produces. Each batch object should have properties of batch.src_name 
        and batch.trg_name that contain the data for the source and target languages. So with 
        the data_loader, we can access the data as:

        for curr_batch, (batch) in enumerate(data_loader):
            batch.src_name -> produces tensor of shape (batch_size, seq_len) of the source language
            batch.trg_name -> produces tensor of shape (batch_size, seq_len) of the target language
        
        Same idea for the valid_loader. 

        Inputs:
            -> data_loader (Iterator): Iterator for the data used to train the model in batches
            -> batch_size (int): Size of the batches used within an epoch 
            -> valid_loader (Iterator): Iterator for the data used to validate the model in batches 
            -> src_name (Property Accessor): Used to access the batch of examples of shape (batch_size, seq_len)
            of source language
            -> trg_name (Property Accessor): Used to access the batch of examples of shape (batch_size, seq_len)
            of the target language
            -> padding_idx (int): Represents idx used for masking 
            -> learn_rate (int): Integer representing learning rate used to update parameters
            -> learning_schedule (Function | None): Schedule used to update the learning rate throughout training. If provided,
            function must take as input learn rate and the current epoch the net is on. 
            -> num_epochs (int): Number of epochs to train the model for
            -> _testing (int | None): During testing, parameter used to ensure model can do well on a small number of 
            examples. So if one epoch has say 56 batches, instead of going through all 56 batches, we only go through testing
            batches in a single epoch. 
        """
        training_losses = []
        validation_losses = []
        smoothLossTrain = smoothLoss(0.1)
        smoothLossValid = smoothLoss(0.1)
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = []
            for i, batch in enumerate(data_loader):
                if i == _testing:
                    break

                # Shape (M,T_src)
                if verbose:
                    print('Entering epoch: %s, batch number %s!' % (epoch, i))
                src_train = getattr(batch, src_name)
                # Shape (M,T_trg)
                trg_train = getattr(batch, trg_name)

                # make sure we have np arrays and shapes of arrays are (M,T)
                src_train, trg_train = self._preprocessBatch(
                    src_train, trg_train, batch_size)

                # Shape (batch_size, seq_len)
                mask_src = getMask(src_train, padding_idx)
                mask_trg = getMask(trg_train, padding_idx)

                loss = self._forward(src_train, trg_train, mask_src, mask_trg)
                self._backward(learn_rate)
                epoch_loss.append(loss)

            # smoothen the loss out when training
            training_losses.append(smoothLossTrain(np.mean(epoch_loss)))
            # Update learn rate after every epoch
            saved_lr = learn_rate
            learn_rate = learn_rate if not learning_schedule else learning_schedule(
                learn_rate, epoch)
            if learning_schedule:
                print('old learn rate: %s, new learn rate: %s' %
                      (saved_lr, learn_rate))

            # get loss w/ teacher forcing when validating
            if valid_loader:
                batch_losses = []

                for i, batch in enumerate(valid_loader):
                    if i == _testing:
                        break
                    srcV = getattr(batch, src_name)
                    trgV = getattr(batch, trg_name)
                    srcV, trgV = self._preprocessBatch(srcV, trgV, batch_size)
                    mask_srcV = getMask(srcV, padding_idx)
                    mask_trgV = getMask(trgV, padding_idx)

                    if i % 100 == 0 and verbose:
                        input_sentence = " ".join(
                            list(map(lambda x: self.src_map_i2c[x], srcV[0])))
                        predicted = self.predict(srcV[0:1])
                        print(predicted)
                        print(
                            'Batch %s, input_sentence: %s translated sentence: %s'
                            % (i, input_sentence, predicted))

                    loss = self._forward(srcV, trgV, mask_srcV, mask_trgV)
                    batch_losses.append(loss)

                validation_losses.append(smoothLossValid(np.mean(batch_losses)))

            if verbose and valid_loader:
                print('Epoch num: %s, Train Loss: %s, Validation Loss: %s' %
                      (epoch, training_losses[-1], validation_losses[-1]))
            if verbose:
                print('Epoch num: %s, Train Loss: %s' %
                      (epoch, training_losses[-1]))

        return training_losses, validation_losses

    def _preprocessBatch(self, x1, x2, batch_size):
        if type(x1) == torch.Tensor:
            x1 = x1.numpy()
            x2 = x2.numpy()
        if x1.shape[0] != batch_size:
            x1 = x1.T
            x2 = x2.T
        return x1, x2

    def predict(self,
                inp_seq,
                length_normalization=0.75,
                beam_width=3,
                max_seq_len=20):
        """
        This method carries out translation from a source language to a target language
        when given a vector of integers in the source language (that should belong to the same vocabulary
        as the network was trained on). 

        This method utilizes the (greedy) beam search algorithm to determine the highest probability
        sentence that can be output from the architecture. 

        Candidate solutions have their probabilities normalized by their length to solve the problem 
        of outputting overly short sentences as described in: https://arxiv.org/pdf/1609.08144.pdf. 

        Inputs:
            -> inp_seq (NumPy vector): Vector of shape (1, t) where t indicates time steps. 
            -> beam_width (int): Integer representing the size of the beam to use during the search
            -> length_normalization (int): Integer indicating the factor by which to normalize the probabilities
            of the candidate solutions by their length
            -> max_seq_len (int): Integer representing the maximum number of time steps the decoder will unfold 
            when generating a sentence 
        Outputs:
            -> String containing the decoded vector by the network 
        """
        assert inp_seq.max(
        ) < self.src_dim, "The sequence has to belong to the same vocabulary as the model was trained on!"
        encoded = self.Encoder(inp_seq, None)
        output_ints = self.Decoder.beamSearch(encoded, self.eos_int,
                                              self.sos_int,
                                              length_normalization, beam_width,
                                              max_seq_len)
        output_sentence = list(map(lambda x: self.trg_map_i2c[x], output_ints))
        return "".join(output_sentence)
