import random
import torch
from torch.autograd import Variable

class ArithmeticTask():
  PLUS = 2
  MINUS = 3
  OPERATIONS = [PLUS, MINUS]

  def __init__(self, vocab_size=100):
    self.vocab_size = vocab_size

  def _generate_arithmetic_example(self):
    ls_type = None
    r = lambda: random.randint(5, 20)
    op = lambda: random.choice(self.OPERATIONS)
    seq = [r(), op(), r(), op(), r()]
    s = 0
    i = 0
    while i < len(seq):
      if i == 0:
        s += seq[i]
        i += 1
      else:
        if seq[i] == self.PLUS:
          s += seq[i+1]
          i += 2
        elif seq[i] == self.MINUS:
          s -= seq[i+1]
          i += 2
        else:
          raise ValueError()
    return seq, s

  def _generate_positive_arithmetic_examples(self):
    success = 0
    failed = 0
    while 1:
      op_sequence, target_sum = self._generate_arithmetic_example()
      if self.vocab_size-1 > target_sum > self.MINUS:
        yield op_sequence, [target_sum]
        success += 1
      else:
        failed += 1

  def __iter__(self):
    return self._generate_positive_arithmetic_examples()

class CopyTask():
  def __init__(self, seq_len=10, vocab_size=100):
    self.seq_len = seq_len
    self.vocab_size = vocab_size

  def __iter__(self):
    while 1:
      x = [random.randint(3, self.vocab_size-1) for _ in range(self.seq_len)]
      y = x
      yield x, y


def preproc(task, batch_size, cuda=False, eos_token=1):
  xs = []
  ys = []
  for inputs, targets in task:
      xs.append(inputs)
      ys.append(targets + [eos_token])
      if len(xs) >= batch_size:
        x = Variable(torch.LongTensor(xs)).transpose(1, 0).contiguous()
        y = Variable(torch.LongTensor(ys)).transpose(1, 0).contiguous()
        if cuda:
          x = x.cuda()
          y = y.cuda()
        yield x, y
        xs = []
        ys = []
