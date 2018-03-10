from torch.optim import Adam

import hyperparams as hp
from datasets import load_dataset
from models import Encoder, Decoder, Seq2Seq
from utils import train
from sgdr import SGDRScheduler

train_iter, val_iter, test_iter, DE, EN = load_dataset(batch_size=hp.batch_size, device=hp.device)

encoder = Encoder(source_vocab_size=len(DE.vocab),
                  embed_dim=hp.embed_dim, hidden_dim=hp.hidden_dim,
                  n_layers=hp.n_layers, dropout=hp.dropout)
decoder = Decoder(target_vocab_size=len(EN.vocab),
                  embed_dim=hp.embed_dim, hidden_dim=hp.hidden_dim,
                  n_layers=hp.n_layers, dropout=hp.dropout)
seq2seq = Seq2Seq(encoder, decoder)

seq2seq.cuda(device=hp.device)
optimizer = Adam(seq2seq.parameters(), lr=hp.max_lr)
scheduler = SGDRScheduler(optimizer, max_lr=hp.max_lr, cycle_length=hp.cycle_length)

train(seq2seq, optimizer, scheduler, train_iter, val_iter, num_epochs=hp.num_epochs)
