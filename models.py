import random

import torch
from torch import nn
from torch.autograd import Variable

from attention import LuongAttention


def reshape_hidden(encoder_hidden):
    """
    reshape (n_layers * direction, batch_size, dim) bidirectional encoder hidden to
    (layers, batch_size, dim*n_layers) hidden.
    """
    _, batch_size, dim = encoder_hidden.size()
    return encoder_hidden.view(-1, batch_size, dim * 2)


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=1)
        self.gru = nn.GRU(embed_dim, hidden_dim // 2, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)  # (batch_size, seq_len, embed_dim)
        encoder_out, encoder_hidden = self.gru(
            embedded, hidden)  # (seq_len, batch, hidden_dim*2)
        return encoder_out, encoder_hidden


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_dim, hidden_dim,
                 n_layers=1, dropout=0.5):
        super(Decoder, self).__init__()

        self.embed = nn.Embedding(trg_vocab_size, embed_dim)
        self.attention = LuongAttention(hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, n_layers,
                          dropout=dropout)
        self.out = nn.Linear(hidden_dim * 2, trg_vocab_size)

    def _forward(self, output, encoder_out, decoder_hidden):
        """
        decodes one output frame
        """
        embedded = self.embed(output)  # (1, batch, embed_dim)
        context, attention_weights = self.attention(
            decoder_hidden[:-1], encoder_out)  # 1, 1, 50 (seq, batch, hidden_dim)
        rnn_output, decoder_hidden = self.gru(torch.cat([embedded, context], dim=2),
                                              decoder_hidden)
        output = self.out(torch.cat([rnn_output, context], 2))
        return output, decoder_hidden, attention_weights

    def forward(self, encoder_out, encoder_hidden, trg, teacher_forcing_ratio=0.5):
        outputs = []
        masks = []
        decoder_hidden = reshape_hidden(encoder_hidden)
        start_token = torch.zeros_like(trg[:1])
        output, decoder_hidden, mask = self._forward(start_token, encoder_out, decoder_hidden)
        output = Variable(output.data.max(dim=2)[1])
        for t in range(len(trg)):
            output, decoder_hidden, mask = self._forward(output, encoder_out, decoder_hidden)
            outputs.append(output)
            masks.append(mask.data)
            output = Variable(output.data.max(dim=2)[1])
            # teacher forcing
            is_teacher = random.random() < teacher_forcing_ratio
            if is_teacher:
                output = trg[t].unsqueeze(0)
        return torch.cat(outputs), torch.cat(masks).permute(1, 2, 0)  # batch, src, trg


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        """
        feed targets and teacher_forcing_ratio at training time, set max_len at test time
        """
        encoder_out, encoder_hidden = self.encoder(src)
        outputs, masks = self.decoder(encoder_out, encoder_hidden,
                                      trg, teacher_forcing_ratio)
        return outputs, masks
