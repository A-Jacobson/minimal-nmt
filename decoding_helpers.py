import random

import torch
from torch.autograd import Variable
import torch.nn.functional as F


class Teacher:
    def __init__(self, teacher_forcing_ratio=0.5):
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.targets = None
        self.maxlen = 0

    def load_targets(self, targets):
        self.targets = targets
        self.maxlen = len(targets)

    def generate(self, decoder, encoder_out, encoder_hidden):
        outputs = []
        masks = []
        decoder_hidden = encoder_hidden[-decoder.n_layers:]  # take what we need from encoder
        output = self.targets[0].unsqueeze(0)  # start token
        for t in range(1, self.maxlen):
            output, decoder_hidden, mask = decoder(output, encoder_out, decoder_hidden)
            outputs.append(output)
            masks.append(mask.data)
            output = Variable(output.data.max(dim=2)[1])
            # teacher forcing
            is_teacher = random.random() < self.teacher_forcing_ratio
            if is_teacher:
                output = self.targets[t].unsqueeze(0)
        return torch.cat(outputs), torch.cat(masks).permute(1, 2, 0)  # batch, src, trg


class Greedy:
    def __init__(self, maxlen=20, sos_index=2, use_stop=False):
        self.maxlen = maxlen
        self.sos_index = sos_index
        self.use_stop = use_stop

    def set_maxlen(self, maxlen):
        self.maxlen = maxlen

    def generate(self, decoder, encoder_out, encoder_hidden):
        stop = False
        seq, batch_size, _ = encoder_out.size()
        if self.use_stop:
            assert batch_size == 1, 'use_stop does not support batching, set batch size to 1'

        outputs = []
        masks = []  # trg, batch, source
        decoder_hidden = encoder_hidden[-decoder.n_layers:]  # take what we need from encoder
        output = torch.zeros_like(encoder_hidden[:1, :, 0]).long() + self.sos_index  # start token
        while (len(outputs) <= self.maxlen) and not stop:
            output, decoder_hidden, mask = decoder(output, encoder_out, decoder_hidden)
            outputs.append(output)
            masks.append(mask.data)
            output = Variable(output.data.max(dim=2)[1])
            if self.use_stop:
                #  generate until stop token is sampled
                _, pred = F.softmax(output, dim=-1).topk(1)
                if int(pred.data[0]) == self.sos_index:
                    stop = True
        return torch.cat(outputs), torch.cat(masks).permute(1, 2, 0)  # batch, trg, src (i, x, y)
