import pytest
import torch
from torch.autograd import Variable

from attention import LuongAttention


def test_attention_sizes():
    """
    Attention should output a fixed length context vector (seq len = 1)
    and and a weight for each item in the input sequence
    """
    encoder_out = Variable(torch.randn(152, 2, 256))  # seq, batch, dim
    query_vector = Variable(torch.randn(1, 2, 256))  # seq, batch, dim

    attention = LuongAttention(attention_dim=256)
    context, mask = attention(query_vector, encoder_out)
    assert context.size() == (1, 2, 256)  # seq, batch, dim
    assert mask.size() == (1, 2, 152)  # seq2, batch, seq1


def test_attention_softmax():
    encoder_out = Variable(torch.randn(152, 2, 256))  # seq, batch, dim
    query_vector = Variable(torch.randn(1, 2, 256))  # seq, batch, dim
    attention = LuongAttention(attention_dim=256)
    context, mask = attention(query_vector, encoder_out)
    assert pytest.approx(mask[:, 0, :].sum().data, 1e-6) == 1.0  # batch, input_seq_len
