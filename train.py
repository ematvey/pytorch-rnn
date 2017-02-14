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


def train(model, task, max_batches=1000, debug=False, verbose=False):
  batch_size = 40

  opt = optim.Adam(model.parameters())

  criterion = nn.CrossEntropyLoss()

  loss_track = []
  ex_iter = tasks.preproc(task, batch_size=batch_size, eos_token=model.EOS_token)
  model.train()
  for i, (inputs, targets) in enumerate(ex_iter):
    opt.zero_grad()
    logits = model(inputs, targets=targets)
    length = logits.size()[2]
    loss = criterion(logits.view(-1, length), targets.view(-1))
    loss.backward()
    opt.step()

    loss_track.append(loss.data[0])

    if verbose and i % 100 == 0:
      _, v = logits.data.topk(1)
      inp = inputs.data.numpy().T.tolist()[0]
      fc = v.squeeze(2).numpy().T.tolist()[0]
      tg = targets.data.numpy().T.tolist()[0]
      print("%.4f\n  inputs:  %s\n  predict: %s\n  target:  %s" % (loss.data[0], inp, fc, tg))

    if i >= max_batches:
      break

  if debug:
    import IPython
    IPython.embed(banner1='train interrupted')

  return loss_track

if __name__ == '__main__':
  # -- debug hook --
  import sys, ipdb, traceback
  def info(type, value, tb):
      traceback.print_exception(type, value, tb)
      print
      ipdb.pm()
  sys.excepthook = info


  task = tasks.ArithmeticTask()

  model = Seq2Seq(input_size=task.vocab_size,
                  output_size=task.vocab_size,
                  hidden_size=100,
                  embedding_size=task.vocab_size, n_layers=1, attention=True)

  train(model, task)