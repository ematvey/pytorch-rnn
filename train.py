import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import tasks
from seq2seq import Seq2Seq


def train(model, task, max_batches=1000, batch_size=40, cuda=False, debug=False, verbose=False):
  opt = optim.Adam(model.parameters())

  criterion = nn.CrossEntropyLoss()

  loss_track = []
  ex_iter = tasks.preproc(task, batch_size=batch_size, cuda=cuda, eos_token=model.EOS_token)
  model.train()
  for i, (inputs, targets) in enumerate(ex_iter):
    opt.zero_grad()
    logits = model(inputs, targets=targets)
    length = logits.size()[2]
    loss = criterion(logits.view(-1, length), targets.view(-1))
    loss.backward()
    opt.step()

    loss_track.append(loss.data[0])

    if verbose and i % 1 == 0:
      print('loss: %.4f' % loss.data[0])

    if verbose and i % 100 == 0:
      _, v = logits.cpu().data.topk(1)
      inp = inputs.cpu().data.numpy().T.tolist()[0]
      fc = v.cpu().squeeze(2).numpy().T.tolist()[0]
      tg = targets.cpu().data.numpy().T.tolist()[0]
      print("  inputs:  %s\n  predict: %s\n  target:  %s" % (inp, fc, tg))

    if i >= max_batches:
      break

  if debug:
    import IPython
    IPython.embed(banner1='train interrupted')

  return loss_track

if __name__ == '__main__':
  task = tasks.CopyTask(seq_len=100, vocab_size=100)

  cuda = torch.cuda.is_available()

  model = Seq2Seq(input_size=task.vocab_size,
                  output_size=task.vocab_size,
                  hidden_size=256,
                  embedding_size=task.vocab_size,
                  n_layers=5,
                  use_cuda=cuda,
                  attention=True)
  if cuda:
    model = model.cuda()

  import time
  t = time.time()
  loss_track = train(model, task, batch_size=100, max_batches=10000, cuda=cuda, verbose=True)
  print('done after %s' % (time.time() - t))
