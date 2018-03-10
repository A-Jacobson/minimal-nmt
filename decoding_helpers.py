import random
import torch
from torch.autograd import Variable


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
    def __init__(self, maxlen=20, sos_index=2):
        self.maxlen = maxlen
        self.sos_index = sos_index

    def set_maxlen(self, maxlen):
        self.maxlen = maxlen

    def generate(self, decoder, encoder_out, encoder_hidden):
        #TODO support generation until stop token is sampled
        seq, batch, _ = encoder_out.size()
        outputs = []
        masks = []
        decoder_hidden = encoder_hidden[-decoder.n_layers:]  # take what we need from encoder
        output = Variable(torch.zeros_like(encoder_hidden[:1, :, 0]).long() + self.sos_index)  # start token
        for t in range(self.maxlen):
            output, decoder_hidden, mask = decoder(output, encoder_out, decoder_hidden)
            outputs.append(output)
            masks.append(mask.data)
            output = Variable(output.data.max(dim=2)[1])
        return torch.cat(outputs), torch.cat(masks).permute(1, 2, 0)  # batch, src, trg