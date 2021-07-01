""" This module contains a class representing a sequence to sequence
network meant to be used for the task of language translation """
from model.encoder import Encoder
from model.decoder import Decoder
from model.utils import GradientDescentMomentum
from model.utils import getMask
from model.utils import smoothLoss
import torch
import numpy as np
from tqdm import tqdm
from typing import Union, Dict, Tuple, Iterator, Callable, Literal, List


class SequenceToSequenceRecurrentNetwork(object):
    """ This class represents a sequence to sequence model used
    for the task of machine translation, trained in batches.

    The architecture is as follows: we feed in an input sequence
    in a source language to the encoder, and get an encoded vector
    from the encoder. We feed that encoded vector into a decoder
    network which will output predictions of the word it thinks is
    the most likely in the target language.

    Attributes:
        eos_int:
            Integer representing the index which the end of sequence
            token occupies in the target vocab

        sos_int:
            Integer representing the index which the start of sequence
            token occupies in the target vocab

        vocab_size_src:
            Integer representing the size of the source language vocabulary

        vocab_size_trg:
            Integer representing the size of the target language vocabulary

        dim_embed_src:
            Integer representing the size of the embeddings for the encoder

        src_map_i2c:
            Dictionary containing a mapping between integers to tokens for
            the source language. Default value is None.

        trg_map_i2c:
            Dictionary containing a mapping between integers to tokens for
            for the target language. Default value is None.

        dim_embed_trg:
            Integer representing the size of the embeddings for the decoder

        num_neurons_encoder:
            Integer representing the number of neurons in the encoder

        optim:
            Object representing the optimizer used to train the parameters
            within the model

        num_neurons_decoder:
            Integer representing the number of neurons in the decoder
    """

    def __init__(self,
                 vocab_size_src: int,
                 vocab_size_trg: int,
                 eos_int: int,
                 sos_int: int,
                 dim_embed_src: int = 512,
                 src_map_i2c: Union[Dict[int, str], None] = None,
                 trg_map_i2c: Union[Dict[int, str], None] = None,
                 dim_embed_trg: int = 512,
                 num_neurons_encoder: int = 512,
                 num_neurons_decoder: int = 512,
                 optim: object = GradientDescentMomentum):
        assert dim_embed_trg == num_neurons_decoder, (
            "For weight tying, the number of neurons in the decoder has to be" +
            "the same as the number of dimensions in the embedding")
        # These don't have to be equal. If they aren't, you will need an
        # additional weight matrix after the encoder to project down or
        # up to the dimensionality of the decoder, adding extra complexity.
        # Kept symmetric for simplicities sake
        assert num_neurons_decoder == num_neurons_encoder, (
            "Currently this model only supports symmetric decoders and encoders"
        )
        self.src_dim = vocab_size_src
        self.trg_map_i2c = trg_map_i2c
        self.src_map_i2c = src_map_i2c
        self.optim = optim
        self.encoder = Encoder(vocab_size_src, dim_embed_src,
                               num_neurons_encoder, optim)
        self.decoder = Decoder(vocab_size_trg, dim_embed_trg,
                               num_neurons_decoder, optim)
        self.eos_int = eos_int
        self.sos_int = sos_int

    def forward(self, x: np.ndarray, y: np.ndarray, mask_src: np.ndarray,
                mask_trg: np.ndarray) -> Tuple[float, np.ndarray]:
        """ This method computes the forward pass through the
        sequence to sequence model, producing a loss value and
        a predictions vector.

        M - size of batch
        N - sequence length

        Args:
            x:
                Numpy matrix of shape (M,N) containing feature vectors
                for training

            y:
                Numpy vector containing integers for the labels for the
                training vectors

            mask_src:
                Numpy matrix indicating which timesteps belong to padding
                idxs for the encoder

            mask_trg:
                Numpy matrix indicating which timesteps belong to padding
                idxs for the decoder

            loss_func:
                Function to minimize during training

        Returns:
            A floating point value representing the loss on the batch of
            input examples, and a numpy matrix containing the probabilistic
            predictions for every vector in the batch
        """
        encoded_batch = self.encoder(x, mask_src)
        loss = self.decoder(encoded_batch, y, mask_trg)
        return loss

    def _backward(self, learn_rate: float):
        """ This method computes the backward pass through the network.
        """
        # vector containing the gradient dL/dA for the encoded vector
        # produced at last time step for encoder
        da_encoded_vector = self.decoder._backward(learn_rate)
        self.encoder._backward(da_encoded_vector, learn_rate)

    def train(
            self,
            data_loader: Iterator,
            batch_size: int,
            src_name: str,
            trg_name: str,
            padding_idx: int,
            valid_loader: Iterator = None,
            learn_rate: float = 0.1,
            learning_schedule: Union[Callable[[float, int], float],
                                     None] = None,
            num_epochs: int = 100,
            verbose: Literal[0, 1] = 1,
            testing: Union[int,
                           None] = None) -> Tuple[List[float], List[float]]:
        """ This method is used to train the seq2seq rnn model in batches.
        This method expects the data to come in as an iterator, and the batches
        to come in as an object like TorchText's bucket iterator produces. Each
        batch object should have properties of batch.src_name and batch.trg_name
        that contain the data for the source and target languages. So with
        the data_loader, we can access the data as:

        for curr_batch, (batch) in enumerate(data_loader):
            batch.src_name -> produces tensor of shape (batch_size, seq_len) of
                              the source language
            batch.trg_name -> produces tensor of shape (batch_size, seq_len) of
                              the target language

        Same idea for the valid_loader.

        Args:
            data_loader:
                Iterator that will loop over the training data

            batch_size:
                Integer representing the size of the batches used within
                an epoch

            valid_loader:
                Iterator for the data used to validate the model in batches

            src_name:
                String used to access the batch of examples of shape (batch_size
                , seq_len) of source language

            trg_name:
                String used to access the batch of examples of shape (batch_size
                , seq_len) of the target language

            padding_idx:
                Integer representing the index used for masking

            learn_rate:
                Floating point value that represents the learning rate used to
                update parameters with gradient descent

            learning_schedule:
                Function representing the learning rate schedule used to update
                the learning rate throughout training, or None if not wanted.
                If provided, the function must take an input learn rate and the
                current epoch the net is on.

            num_epochs:
                Integer representing the number of epochs to train the model for

            verbose:
                Integer that should be either 0 or 1 representing whether or not
                to log statements while the algorithm trains

            testing:
                During testing, parameter used to ensure model can do well on a
                small number of examples. So if one epoch has say 56 batches,
                instead of going through all 56 batches, we only go through
                testing batches in a single epoch.

        Returns:
            List of floating point values representing the training loss, and
            a list of floating point values representing the validation loss
            respectively.
        """
        training_losses = []
        validation_losses = []
        smooth_loss_train = smoothLoss(0.1)
        smooth_loss_valid = smoothLoss(0.1)
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = []
            for i, batch in enumerate(data_loader):
                if i == testing:
                    break

                # Shape (M,T_src)
                if verbose:
                    print(f"Entering epoch: {epoch}, batch number {i}!" %
                          (epoch, i))
                src_train = getattr(batch, src_name)
                # Shape (M,T_trg)
                trg_train = getattr(batch, trg_name)

                # make sure we have np arrays and shapes of arrays are (M,T)
                src_train, trg_train = self._preprocess_batch(
                    src_train, trg_train, batch_size)

                # Shape (batch_size, seq_len)
                mask_src = getMask(src_train, padding_idx)
                mask_trg = getMask(trg_train, padding_idx)

                loss = self.forward(src_train, trg_train, mask_src, mask_trg)
                self._backward(learn_rate)
                epoch_loss.append(loss)

            # smoothen the loss out when training
            training_losses.append(smooth_loss_train(np.mean(epoch_loss)))
            # Update learn rate after every epoch
            saved_lr = learn_rate
            learn_rate = (learn_rate if not learning_schedule else
                          learning_schedule(learn_rate, epoch))
            if learning_schedule:
                print(
                    f"old learn rate: {saved_lr}, new learn rate: {learn_rate}")

            # get loss w/ teacher forcing when validating
            if valid_loader:
                batch_losses = []

                for i, batch in enumerate(valid_loader):
                    if i == testing:
                        break
                    src_v = getattr(batch, src_name)
                    trg_v = getattr(batch, trg_name)
                    src_v, trg_v = self._preprocess_batch(
                        src_v, trg_v, batch_size)
                    mask_src_v = getMask(src_v, padding_idx)
                    mask_trg_v = getMask(trg_v, padding_idx)

                    if i % 100 == 0 and verbose:
                        input_sentence = " ".join(
                            list(map(lambda x: self.src_map_i2c[x], src_v[0])))
                        predicted = self.predict(src_v[0:1])
                        print(predicted)
                        print(f"Batch {i}, input_sentence: {input_sentence} " +
                              f"translated sentence: {predicted}")

                    loss = self.forward(src_v, trg_v, mask_src_v, mask_trg_v)
                    batch_losses.append(loss)

                validation_losses.append(
                    smooth_loss_valid(np.mean(batch_losses)))

            if verbose and valid_loader:
                print(
                    f"Epoch num: {epoch}, Train Loss: {training_losses[-1]}, " +
                    f"Validation Loss: {validation_losses[-1]}")
            if verbose:
                print(f"Epoch num: {epoch}, Train Loss: {training_losses[-1]}")

        return training_losses, validation_losses

    def _preprocess_batch(self, x1: np.ndarray, x2: np.ndarray,
                          batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(x1, torch.Tensor):
            x1 = x1.numpy()
            x2 = x2.numpy()
        if x1.shape[0] != batch_size:
            x1 = x1.T
            x2 = x2.T
        return x1, x2

    def predict(self,
                inp_seq: np.ndarray,
                length_normalization: float = 0.75,
                beam_width: int = 3,
                max_seq_len: int = 20) -> str:
        """ This method carries out translation from a source language to
        a target language when given a vector of integers in the source language
        (that should belong to the same vocabulary as the network was trained
        on).

        This method utilizes the (greedy) beam search algorithm to determine the
        highest probability sentence that can be output from the architecture.

        Candidate solutions have their probabilities normalized by their length
        to solve the problem of outputting overly short sentences as described:
        in: https://arxiv.org/pdf/1609.08144.pdf.

        Args:
            inp_seq:
                Numpy array of shape (1, t) where t indicates time steps

            length_normalization:
                Integer indicating the factor by which to normalize the
                probabilities of the candidate solutions by their length

            beam_width:
                Integer representing the size of the beam to use during the
                search

            max_seq_len:
                Integer representing the maximum number of time steps the
                decoder will unfold when generating a sentence

        Returns:
            String containing the decoded vector by the network
        """
        assert inp_seq.max() < self.src_dim, (
            "The sequence has to belong to the same vocabulary as the model was trained on!"
        )
        encoded = self.encoder(inp_seq, None)
        output_ints = self.decoder.beamSearch(encoded, self.eos_int,
                                              self.sos_int,
                                              length_normalization, beam_width,
                                              max_seq_len)
        output_sentence = list(map(lambda x: self.trg_map_i2c[x], output_ints))
        return "".join(output_sentence)
