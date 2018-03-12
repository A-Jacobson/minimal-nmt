import heapq

import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Beam:
    """
    maintains a heap of size(beam_width), always removes lowest scoring nodes.
    """

    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    def add(self, score, sequence, hidden_state):
        heapq.heappush(self.heap, (score, sequence, hidden_state))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)

    def __len__(self):
        return len(self.heap)

    def __getitem__(self, idx):
        return self.heap[idx]


class BeamHelper:
    """
    Model must be in eval mode
    Note: Will be passed as decoding helper,
    but does not currently conform to that api so it gets to live here.
    Does not support batching. Does not work with current eval code
    (can't compute Xentropy loss on returned indices).
    """

    # TODO return attention masks, stop when sos index is sampled

    def __init__(self, beam_size=3, maxlen=20, sos_index=2):
        self.beam_size = beam_size
        self.maxlen = maxlen
        self.sos_index = sos_index
        self.decoder = None
        self.encoder_out = None

    def get_next(self, last_word, hidden_state):
        """
        Given the last item in a sequence and the hidden state used to generate the sequence
        return the topk most likely words and their scores
        """
        output, hidden_state, _ = self.decoder(last_word, self.encoder_out, hidden_state)
        probs = F.softmax(output, dim=2)
        next_probs, next_words = probs.topk(self.beam_size)
        return next_probs.squeeze().data, next_words.view(self.beam_size, 1, 1), hidden_state

    def search(self, start_token, initial_hidden):
        beam = Beam(self.beam_size)  # starting layer in search tree
        beam.add(score=1.0, sequence=start_token, hidden_state=initial_hidden)  # initialize root
        for _ in range(self.maxlen):
            next_beam = Beam(self.beam_size)
            for score, sequence, hidden_state in beam:
                next_probs, next_words, hidden_state = self.get_next(sequence[-1:],
                                                                     hidden_state)
                for i in range(self.beam_size):
                    score = score * next_probs[i]
                    sequence = torch.cat([sequence, next_words[i]])  # add next word to sequence
                    next_beam.add(score, sequence, hidden_state)
            # move down one layer (to the next word in sequence up to maxlen )
            beam = next_beam
        best_score, best_sequence, _ = max(beam)  # get highest scoring sequence
        return best_score, best_sequence

    def __call__(self, decoder, encoder_out, encoder_hidden):
        self.decoder = decoder
        self.encoder_out = encoder_out
        decoder_hidden = encoder_hidden[-decoder.n_layers:]  # take what we need from encoder
        start_token = Variable(decoder_hidden.data.new(1, 1).fill_(self.sos_index).long())  # start token (ugly hack)
        best_score, best_sequence = self.search(start_token, decoder_hidden)
        return best_score, best_sequence
