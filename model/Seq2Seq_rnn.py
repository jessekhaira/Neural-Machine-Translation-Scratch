from Encoder import Encoder
from Decoder import Decoder 
from Utils import GradientDescentMomentum
from Utils import crossEntropy
from Utils import getMask
from Utils import smoothLoss
import torch 
import numpy as np 
from tqdm import tqdm
from collections import deque 

class Seq2Seq_rnn(object):
    """
    This class represents a sequence to sequence model used for the task of machine translation, trained
    in batches. 

    The architecture is as follows: we feed in an input sequence in a source language to the encoder, and get an encoded
    vector from the encoder. We feed that encoded vector into a decoder network which will output predictions
    of the word it thinks is the most likely in the target language. 

    Inputs:
        -> eos_int (int): Integer representing the index which the end of sequence token occupies 
        -> sos_int (int): Integer representing the index which the start of sequence token occupies
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
                src_map_i2c = None, 
                trg_map_i2c= None, 
                dim_embed_trg=512, 
                num_neurons_encoder=512, 
                num_neurons_decoder=512,
                optim = GradientDescentMomentum):
        assert dim_embed_trg == num_neurons_decoder, "For weight tying, the number of neurons in the decoder has to be the same as the number of dimensions in the embedding"
        # These don't have to be equal. If they aren't, you will need an additional weight matrix after the encoder to project down or up to the 
        # dimensionality of the decoder, adding extra complexity. Kept symmetric for simplicities sake
        assert num_neurons_decoder == num_neurons_encoder, "Currently this model only supports symmetric decoders and encoders"
        self.src_dim = vocab_size_src
        self.trg_map_i2c = trg_map_i2c
        self.src_map_i2c = src_map_i2c
        self.optim = optim 
        self.Encoder = Encoder(vocab_size_src, dim_embed_src, num_neurons_encoder, optim)
        self.Decoder = Decoder(vocab_size_trg, dim_embed_trg, num_neurons_decoder, optim)
        self.eos_int = eos_int
        self.sos_int = sos_int


    def _forward(self, x,y, mask_src, mask_trg, loss_func):
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


    def train (self, 
            data_loader, 
            src_name, 
            trg_name, 
            padding_idx, 
            valid_loader = None, 
            learn_rate = 0.3, 
            learning_schedule = None, 
            num_epochs=100, 
            verbose =1):
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
            -> valid_loader (Iterator): Iterator for the data used to validate the model in batches 
            -> src_name (Property Accessor): Used to access the batch of examples of shape (batch_size, seq_len)
            of source language
            -> trg_name (Property Accessor): Used to access the batch of examples of shape (batch_size, seq_len)
            of the target language
            -> padding_idx (int): Represents idx used for masking 
            -> learn_rate (int): Integer representing learning rate used to update parameters
            -> learning_schedule (Function | None): Schedule used to update the learning rate throughout training
            -> num_epochs (int): Number of epochs to train the model for
        """ 
        training_losses = [] 
        validation_losses = [] 
        smoothLossTrain = smoothLoss() 
        smoothLossValid = smoothLoss() 
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = []
            for i, batch in enumerate(data_loader):
                # Shape (M,T_src)
                src_train = batch.src_name
                # Shape (M,T_trg)
                trg_train = batch.trg_name 
                
                # convert torch tensors to numpy
                if type(src_train) == torch.Tensor:
                    src_train = src_train.numpy()
                    trg_train = trg_train.numpy() 
                assert type(src_train) == np.ndarray, "Your batches have to be numpy matrices!"
                
                # get the masks for this batch of examples -- will be of shape (batch_size, seq_len)
                mask_src = getMask(src_train)
                mask_trg = getMask(trg_train)

                loss = self._forward(src_train, trg_train, mask_src, mask_trg)
                self._backward(mask_src, mask_trg, self.optim, learn_rate)
                
                if learning_schedule:
                    learn_rate = learning_schedule(learn_rate, epoch)

                epoch_loss.append(loss)
            
            # smoothen the loss out when training 
            training_losses.append(smoothLossTrain(np.mean(epoch_loss)))


            # get loss w/ teacher forcing when validating
            if valid_loader:
                batch_losses = [] 
                
                for i, batch in enumerate(valid_loader):
                    src = batch.src_name
                    trg = batch.trg_name

                    if i%100 ==0 and verbose:
                        input_sentence = "".join(list(map(lambda x: self.src_map_i2c[x], src[0])))
                        predicted = self.predict(src[0])
                        print('Epoch %s, input_sentence %s: translated sentence %s:'%(i, input_sentence, predicted))

                    _, loss = self.forward(src, trg)
                    batch_losses.append(loss)
                
                validation_losses.append(smoothLossValid(np.mean(batch_losses)))

            if verbose:
                print('Epoch num: %s, Train Loss: %s, Validation Loss: %s'%(epoch, training_losses[-1], validation_losses[-1]))
            

        return training_losses, validation_losses

            
    def predict(self, inp_seq, length_normalization = 0.75, beam_width=3, max_seq_len = 20):
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
        assert inp_seq.max() < self.src_dim, "The sequence has to belong to the same vocabulary as the model was trained on!"
        encoded = self.Encoder(inp_seq, None)
        output_ints = self.Decoder.beamSearch(encoded, self.eos_int, self.sos_int, length_normalization, beam_width, max_seq_len)
        output_sentence = list(map(lambda x: self.trg_map_i2c[x], output_ints))
        return "".join(output_sentence)
        
    




    



        










                    





