import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm

import hyperparams as hp
from decoding_helpers import Greedy, Teacher


def sequence_to_text(sequence, field):
    return " ".join([field.vocab.itos[int(i)] for i in sequence])


def text_to_sequence(text, field):
    return [field.vocab.stoi[word] for word in text]


def evaluate(model, val_iter, writer, step):
    model.eval()
    total_loss = 0
    fields = val_iter.dataset.fields
    greedy = Greedy()
    random_batch = np.random.randint(0, len(val_iter) - 1)
    for i, batch in enumerate(val_iter):
        greedy.set_maxlen(len(batch.trg[1:]))
        outputs, attention = model(batch.src, greedy)
        seq_len, batch_size, vocab_size = outputs.size()
        loss = F.cross_entropy(outputs.view(seq_len * batch_size, vocab_size),
                          batch.trg[1:].view(-1),
                          ignore_index=hp.pad_idx)
        total_loss += loss.data[0]

        # tensorboard logging
        if i == random_batch:
            preds = outputs.topk(1)[1]
            source = sequence_to_text(batch.src[:, 0].data, fields['src'])
            prediction = sequence_to_text(preds[:, 0].data, fields['trg'])
            target = sequence_to_text(batch.trg[1:, 0].data, fields['trg'])
            attention_plot = show_attention(attention[0],
                                            prediction, source, return_array=True)

            writer.add_image('Attention', attention_plot, step)
            writer.add_text('Source: ', source, step)
            writer.add_text('Prediction: ', prediction, step)
            writer.add_text('Target: ', target, step)
    writer.add_scalar('val_loss', total_loss / len(val_iter), step)


def train(model, optimizer, scheduler, train_iter, val_iter,
          num_epochs, teacher_forcing_ratio=0.5, step=0):
    model.train()
    writer = SummaryWriter()
    teacher = Teacher(teacher_forcing_ratio)
    for _ in tqdm(range(num_epochs), total=num_epochs, unit=' epochs'):
        pbar = tqdm(train_iter, total=len(train_iter), unit=' batches')
        for b, batch in enumerate(pbar):
            optimizer.zero_grad()
            teacher.set_targets(batch.trg)
            outputs, masks = model(batch.src, teacher)
            seq_len, batch_size, vocab_size = outputs.size()
            loss = F.cross_entropy(outputs.view(seq_len * batch_size, vocab_size),
                              batch.trg[1:].view(-1),
                              ignore_index=hp.pad_idx)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 10.0, norm_type=2)  # prevent exploding grads
            scheduler.step()
            optimizer.step()

            # tensorboard logging
            pbar.set_description(f'loss: {loss.data[0]:.4f}')
            writer.add_scalar('loss', loss.data[0], step)
            writer.add_scalar('lr', scheduler.lr, step)
            step += 1
        torch.save(model.state_dict(), f'checkpoints/seq2seq_{step}.pt')
        evaluate(model, val_iter, writer, step)


def show_attention(attention, prediction=None, source=None, return_array=False):
    plt.figure(figsize=(14, 6))
    sns.heatmap(attention,
                xticklabels=prediction.split(),
                yticklabels=source.split(),
                linewidths=.05,
                cmap="Blues")
    plt.ylabel('Source (German)')
    plt.xlabel('Prediction (English)')
    plt.xticks(rotation=60)
    plt.yticks(rotation=0)
    if return_array:
        plt.tight_layout()
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        buff.seek(0)
        return np.array(Image.open(buff))
