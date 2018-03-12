import random

import torch
from torch.autograd import Variable


class Teacher:
    def __init__(self, teacher_forcing_ratio=0.5):
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.targets = None
        self.maxlen = 0

    def set_targets(self, targets):
        self.targets = targets
        self.maxlen = len(targets) - 1

    def __call__(self, decoder, encoder_out, encoder_hidden):
        seq1_len, batch_size, _ = encoder_out.size()
        target_vocab_size = decoder.target_vocab_size

        outputs = Variable(encoder_out.data.new(self.maxlen, batch_size, target_vocab_size))
        masks = torch.zeros(self.maxlen, batch_size, seq1_len)
        decoder_hidden = encoder_hidden[-decoder.n_layers:]  # take what we need from encoder
        output = self.targets[0].unsqueeze(0)  # start token
        for t in range(self.maxlen):
            output, decoder_hidden, mask = decoder(output, encoder_out, decoder_hidden)
            outputs[t] = output
            masks[t] = mask.data
            output = Variable(output.data.max(dim=2)[1])
            # teacher forcing
            if random.random() < self.teacher_forcing_ratio:
                output = self.targets[t + 1].unsqueeze(0)
        return outputs, masks.permute(1, 2, 0)  # batch, src, trg


class Greedy:
    def __init__(self, maxlen=20, sos_index=2, use_stop=False):
        self.maxlen = maxlen
        self.sos_index = sos_index
        self.use_stop = use_stop

    def set_maxlen(self, maxlen):
        self.maxlen = maxlen

    def __call__(self, decoder, encoder_out, encoder_hidden):

        seq1_len, batch_size, _ = encoder_out.size()
        target_vocab_size = decoder.target_vocab_size

        if self.use_stop:
            assert batch_size == 1, 'use_stop does not support batching, set batch size to 1'

        outputs = Variable(encoder_out.data.new(self.maxlen, batch_size, target_vocab_size))
        masks = torch.zeros(self.maxlen, batch_size, seq1_len)
        decoder_hidden = encoder_hidden[-decoder.n_layers:]  # take what we need from encoder
        output = Variable(outputs.data.new(1, batch_size).fill_(self.sos_index).long())  # start token (ugly hack)
        for t in range(self.maxlen):
            output, decoder_hidden, mask = decoder(output, encoder_out, decoder_hidden)
            outputs[t] = output
            masks[t] = mask.data
            output = Variable(output.data.max(dim=2)[1])
            if self.use_stop and int(output.data) == self.sos_index:
                break
        return outputs, masks.permute(1, 2, 0)  # batch, trg, src (i, x, y)


