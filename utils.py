import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm
from PIL import Image
from tqdm import tqdm
from decoding_helpers import Greedy, Teacher
import hyperparams as hp


def sequence_to_text(sequence, field):
    return " ".join([field.vocab.itos[int(i)] for i in sequence])


def text_to_sequence(text, field):
    return [field.vocab.stoi[word] for word in text]


def evaluate(model, val_iter, writer, step):
    model.eval()
    total_loss = 0
    fields = val_iter.dataset.fields
    greedy_decoding = Greedy()
    random_batch = np.random.randint(0, len(val_iter) - 1)
    for i, batch in enumerate(val_iter):
        if batch.trg.size(1) != val_iter.batch_size:  # ignore the last batch
            continue
        greedy_decoding.set_maxlen(len(batch.trg))
        out, attention = model(batch.src, greedy_decoding)
        loss = F.cross_entropy(out.view(-1, out.size(2)),
                               batch.trg.view(-1), ignore_index=1)
        total_loss += loss.data[0]
        if i == random_batch:
            probs, pred = F.softmax(out, dim=-1).topk(1)
            source = sequence_to_text(batch.src[:, 0].data, fields['src'])
            prediction = sequence_to_text(pred[1:, 0].data, fields['trg'])
            target = sequence_to_text(batch.trg[1:, 0].data, fields['trg'])
            attention_plot = show_attention(attention[0].cpu().numpy().T,
                                            prediction, source, return_array=True)

            writer.add_image('Attention', attention_plot, step)
            writer.add_text('Source: ', source, step)
            writer.add_text('Prediction: ', prediction, step)
            writer.add_text('Target: ', target, step)
    writer.add_scalar('val_loss', total_loss / len(val_iter), step)


def train(model, optimizer, scheduler, train_iter, val_iter, num_epochs, teacher_forcing_ratio=0.5, step=0):
    model.train()
    writer = SummaryWriter()
    teacher_decoding = Teacher(teacher_forcing_ratio)
    for epoch in tqdm(range(num_epochs), total=num_epochs, unit=' epochs'):
        pbar = tqdm(train_iter, total=len(train_iter), unit=' batches')
        for b, batch in enumerate(pbar):
            if batch.trg.size(1) != train_iter.batch_size:  # ignore the last batch
                continue
            optimizer.zero_grad()
            teacher_decoding.load_targets(batch.trg)
            out, attention = model(batch.src, teacher_decoding)
            loss = F.cross_entropy(out.view(-1, out.size(2)),
                                   batch.trg[1:].view(-1), ignore_index=hp.pad_idx)
            loss.backward()
            clip_grad_norm(model.parameters(), 10.0, norm_type=2)  # prevent exploding grads
            scheduler.step()
            optimizer.step()
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
