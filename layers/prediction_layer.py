import torch
import torch.nn as nn
import numpy as np
from layers.seq2seq import DecoderRNN
from utils.misc import xavier_weights_init_

class Predictor(nn.Module):
    def __init__(self, pred_length, num_node, teacher_forcing_ratio, **kwargs):
        super(Predictor, self).__init__()
        self.decoder = DecoderRNN(**kwargs)

        self.apply(xavier_weights_init_)

        self.pred_length = pred_length
        self.num_node = num_node
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def reshape_from_lstm(self, predicted):
        NV, T, C = predicted.size()
        predicted = predicted.view(-1, self.num_node, T, C)
        predicted = predicted.permute(0, 3, 2, 1).contiguous() # (N, C, T, V)
        return predicted

    def forward(self, last_loc, hidden, loc_GT=None):

        predicted = torch.zeros(last_loc.shape[0], self.pred_length, 2).to(last_loc.device)
        decoder_input = last_loc # (N*V, 1, 2)

        for t in range(self.pred_length):
            now_out, hidden = self.decoder(decoder_input, hidden)
            now_out += decoder_input
            predicted[:,t:t+1] = now_out
            teacher_force = False if loc_GT is None else np.random.random() < self.teacher_forcing_ratio
            decoder_input = loc_GT[:,t:t+1] if teacher_force else now_out

        predicted = self.reshape_from_lstm(predicted) # (N, 2, T, V)

        return predicted