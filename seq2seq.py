import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class Seq2Seq(nn.Module):
  EOS_token = 1

  def __init__(self,
               input_size, output_size, hidden_size,
               n_layers=1, dropout_p=0.5, attention=True, attention_length=30):
  """
  Args:
    ...
    attention_length: size of attention vector, limits precision of addressing
    ...
  """
    super(Seq2Seq, self).__init__()
    self.n_layers = n_layers
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.dropout_p = dropout_p
    self.attention = attention

    self.embedding = nn.Embedding(input_size, hidden_size)
    self.dropout = nn.Dropout(self.dropout_p)
    self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=n_layers)
    self.decoder = nn.GRU(self.hidden_size, self.hidden_size, num_layers=n_layers)
    self.projection = nn.Linear(self.hidden_size, self.output_size)

    if self.attention:
      self.attention_length = attention_length
      self.att_query = nn.Linear(self.hidden_size * 2, self.attention_length)
      self.att_keys = nn.Linear(self.hidden_size, self.attention_length)
      self.decoder_projection = nn.Linear(self.hidden_size * 2, self.hidden_size)

  def forward(self, inputs, targets=None, output_length=None, state=None, teacher_forcing_ratio=0.3):
    """Input is assumed to be LongTensor(seq_len x batch_size)"""
    batch_size = inputs.size()[1]
    input_length = inputs.size()[0]
    is_training = targets is not None
    targets_length = None
    if is_training:
      assert output_length is None
      targets_length = targets.size()[0]
      output_length = targets_length
    else:
      assert output_length is not None

    if state is None:
      state = self._init_hidden(batch_size=batch_size)

    encoder_inputs = self.dropout(self.embedding(inputs))
    # encoder_inputs [seq_len, batch_size, embedding_size]

    encoder_output, encoder_state = self.encoder(encoder_inputs, state)
    # encoder_output [seq_len, batch_size, hidden_size]
    # encoder_state [n_layers, seq_len, hidden_size]

    encoder_output_flat = encoder_output.view(input_length * batch_size, self.hidden_size)

    if self.attention:
      encoder_attention_keys = (
        self.att_keys(encoder_output_flat)
        .view(input_length, batch_size, self.attention_length)
        .transpose(1, 0)
      )

    feed_previous = True
    if is_training and teacher_forcing_ratio is not None:
      if random.random() < teacher_forcing_ratio:
        feed_previous = False

    output_logits = Variable(torch.zeros(output_length, batch_size, self.output_size)).contiguous()

    decoder_state = encoder_state
    decoder_input = Variable(torch.LongTensor([self.EOS_token] * batch_size))

    for i in range(output_length):  # 1 for EOS token
      embedded = self.embedding(decoder_input)

      if self.attention:
        decoder_state_top_layer = decoder_state[-1, :, :]
        att_inputs = torch.cat([embedded, decoder_state_top_layer], 1)
        att_query = self.att_query(att_inputs).unsqueeze(2)
        att_distribution = F.softmax(
            torch.bmm(encoder_attention_keys, att_query).transpose(0, 1)
        ).transpose(0, 1)

        att_context = torch.bmm(
            encoder_output.transpose(1, 0).transpose(1, 2),
            att_distribution,
        ).squeeze(2)


        _inp_c = torch.cat((embedded, att_context), 1)
        decoder_input_concat = self.decoder_projection(_inp_c).unsqueeze(0)

      else:
        decoder_input_concat = embedded.unsqueeze(0)

      decoder_output, decoder_state = self.decoder(decoder_input_concat, decoder_state)

      logits = self.projection(decoder_output.squeeze(0)).unsqueeze(0)

      if feed_previous:
        _, top_i = logits.data.topk(1, 2)
        decoder_input = Variable(top_i.squeeze(2).squeeze(0))
      else:
        decoder_input = targets[i]

      output_logits[i] = logits[0]

    return output_logits

  def _init_hidden(self, batch_size=1):
    return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))


vocab_size = 10
def generate_inputs(batch_size=10, seq_len=10):
  batch = []
  while True:
    while len(batch) < batch_size:
      batch.append([random.randint(3, vocab_size-1) for _ in range(seq_len)])
    inputs = Variable(torch.LongTensor(batch)).transpose(1, 0).contiguous()
    targets = inputs.clone()
    yield inputs, targets
    batch = []


if __name__ == '__main__':

  model = Seq2Seq(vocab_size, vocab_size, 100, n_layers=5, attention=False)

  # opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.01)
  opt = optim.Adam(model.parameters())
  criterion = nn.CrossEntropyLoss()

  model.train()
  for i, (inputs, targets) in enumerate(generate_inputs()):
    opt.zero_grad()
    logits = model(inputs, targets=targets)
    length = logits.size()[2]
    loss = criterion(logits.view(-1, length), targets.view(-1))
    loss.backward()
    opt.step()

    if i % 100 == 0:
      _, v = logits.data.topk(1)
      fc = v.squeeze(2).numpy().T.tolist()
      tg = targets.data.numpy().T.tolist()
      print("%.4f\n  predict: %s\n  target:  %s" % (loss.data[0], fc, tg))
