import torch
from torch import nn

from attention import LuongAttention
import hyperparams as hp


class Encoder(nn.Module):
    def __init__(self, source_vocab_size, embed_dim, hidden_dim,
                 n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(source_vocab_size, embed_dim, padding_idx=hp.pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, source, hidden=None):
        embedded = self.embed(source)  # (batch_size, seq_len, embed_dim)
        encoder_out, encoder_hidden = self.gru(
            embedded, hidden)  # (seq_len, batch, hidden_dim*2)
        # sum bidirectional outputs, the other option is to retain concat features
        encoder_out = (encoder_out[:, :, :self.hidden_dim] +
                       encoder_out[:, :, self.hidden_dim:])
        return encoder_out, encoder_hidden


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, hidden_dim,
                 n_layers, dropout):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(target_vocab_size, embed_dim, padding_idx=hp.pad_idx)
        self.attention = LuongAttention(hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, n_layers,
                          dropout=dropout)
        self.out = nn.Linear(hidden_dim * 2, target_vocab_size)

    def forward(self, output, encoder_out, decoder_hidden):
        """
        decodes one output frame
        """
        embedded = self.embed(output)  # (1, batch, embed_dim)
        context, mask = self.attention(decoder_hidden[:-1], encoder_out)  # 1, 1, 50 (seq, batch, hidden_dim)
        rnn_output, decoder_hidden = self.gru(torch.cat([embedded, context], dim=2),
                                              decoder_hidden)
        output = self.out(torch.cat([rnn_output, context], 2))
        return output, decoder_hidden, mask


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, decoding_helper):
        encoder_out, encoder_hidden = self.encoder(source)
        outputs, masks = decoding_helper.generate(self.decoder, encoder_out, encoder_hidden)
        return outputs, masks
