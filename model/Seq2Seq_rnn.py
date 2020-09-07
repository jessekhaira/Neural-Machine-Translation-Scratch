from Encoder import Encoder
from Decoder import Decoder 
from Utils import GradientDescentMomentum
from Utils import crossEntropy
from Utils import getMask
from Utils import smoothLoss
import torch 
import numpy as np 
from tqdm import tqdm

class Seq2Seq_rnn(object):
    """
    This class represents a sequence to sequence model used for the task of machine translation, trained
    in batches. 

    The architecture is quite simple: we feed in an input sequence in a source language, and get an encoded
    vector from the encoder. We feed that encoded vector into a decoder network which will output predictions
    of the word it thinks is the most likely.

    Inputs:
        -> vocab_size_src (int): Size of the source language vocabulary
        -> vocab_size_trg (int): Size of the target language vocabulary
        -> dim_embed_src (int): Size of the embeddings for the encoder 
        -> src_map_i2c (HashMap<Integer, Token> | None): Mapping from integers to tokens for source language
        -> trg_map_i2c (HashMap<Integer, Token> | None): Mapping from integers to tokens for target language 
        -> dim_embed_trg (int): Size of the embeddings for the decoder
        -> num_neurons_encoder (int): Number of neurons in the encoder
        -> num_neurons_decoder (int): Number of neurons in the decoder 
    """ 
    def __init__(self,  
                vocab_size_src, 
                vocab_size_trg, 
                dim_embed_src=256, 
                src_map_i2c = None, 
                trg_map_i2c= None, 
                dim_embed_trg=256, 
                num_neurons_encoder=512, 
                num_neurons_decoder=512):
        # for the decoder, we're going to tie the weights of the embedding layer and the linear projection
        # before softmax activation. If vocab_size_src and vocab_size_trg are same as well, its possible to tie all
        # the weights but not done here for simplicity of implementation . See: https://arxiv.org/abs/1608.05859
        self.src_dim = vocab_size_src
        self.trg_map_i2c = trg_map_i2c
        self.src_map_i2c = src_map_i2c
        self.Encoder = Encoder(vocab_size_src, dim_embed_src, num_neurons_encoder)
        self.Decoder = Decoder(vocab_size_trg, dim_embed_trg, num_neurons_decoder)
        self.cost_function = crossEntropy


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
        predictions, loss = self.Decoder(encoded_batch, y, mask_trg, self.cost_function)
        return predictions, loss 


    def _backward(self, predictions, mask_src, mask_trg, optim, learn_rate):
        """
        This method computes the backward pass through the network.
        """
        # vector containing the gradient dL/dA for the encoded vector produced at last time
        # step for encoder 
        dA_encodedVector = self.Decoder._backward(predictions, mask_trg, optim, learn_rate)
        self.Encoder._backward(dA_encodedVector, mask_src, optim, learn_rate)



    def train(self, 
            data_loader, 
            src_name, 
            trg_name, 
            padding_idx, 
            valid_loader = None, 
            learn_rate = 0.3, 
            learning_schedule = None, 
            num_epochs=100, 
            optim=GradientDescentMomentum, 
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
            -> optim(object): Optimizer used to train the parameters within the model
        """ 
        optim = optim()
        training_losses = [] 
        validation_losses = [] 
        smoothLossTrain = smoothLoss() 
        smoothLossValid = smoothLoss() 
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = []
            for i, batch in enumerate(data_loader):
                src_train = batch.src_name
                trg_train = batch.trg_name 
                
                # convert torch tensors to numpy
                if type(src_train) == torch.Tensor:
                    src_train = src_train.numpy()
                    trg_train = trg_train.numpy() 
                assert type(src_train) == np.ndarray, "Your batches have to be numpy matrices!"
                
                # get the masks for this batch of examples -- will be of shape (batch_size, seq_len)
                mask_src = getMask(src_train)
                mask_trg = getMask(trg_train)

                predictions, loss = self._forward(src_train, trg_train, mask_src, mask_trg)
                self._backward(predictions, mask_src, mask_trg, optim, learn_rate)
                
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

                    if i%10 ==0:
                        src_ints = src[0].reshape(1,-1)
                        trg_ints = self.predict(src_ints)
                        map_src_i2c = np.vectorize(lambda x: self.src_map_i2c[x])
                        map_trg_i2c = np.vectorize(lambda x: self.trg_map_i2c[x])
                        src_sentence = map_src_i2c(src_ints)
                        trg_sentence = map_trg_i2c(trg_ints)
                        print('Src sentece:%s, Trg sentence predicted: %s'%(src_sentence, trg_sentence))

                    _, loss = self.forward(src, trg)
                    batch_losses.append(loss)
                
                validation_losses.append(smoothLossValid(np.mean(batch_losses)))

            if verbose:
                print('Epoch num: %s, Train Loss: %s, Validation Loss: %s'%(epoch, training_losses[-1], validation_losses[-1]))
            

        return training_losses, validation_losses
            

    def predict(self, inp_seq):
        """
        This method carries out translation from a source language to a target language
        when given a vector of integers in the source language (that belong to the same vocabulary
        as the network was trained on). 

        Inputs:
            -> inp_seq (NumPy vector): Vector of shape (1, t) where t indicates time steps. 
        Outputs:
            -> String containing the decoded vector by the network 
        """
        assert max(inp_seq) < self.src_dim, "The sequence has to belong to the same vocabulary as the model was trained on!"
        encoded_vector = self.Encoder(inp_seq)
        decoded_vector = self.Decoder(encoded_vector)
        xfunc = np.vectorize(lambda x: self.trg_map_i2c[x])
        return xfunc(decoded_vector)



    



        










                    





