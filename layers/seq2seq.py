import torch
import torch.nn as nn
import numpy as np

####################################################
# Seq2Seq LSTM AutoEncoder Model
# 	- predict locations
####################################################
class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers):
		super(EncoderRNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		# self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

	def forward(self, input):
		output, hidden = self.lstm(input)
		return output, hidden

class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, num_layers, dropout=0.5):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers

		# self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
		self.lstm = nn.GRU(output_size, hidden_size, num_layers, batch_first=True)

		#self.relu = nn.ReLU()
		# self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(p=dropout)
		self.linear = nn.Linear(hidden_size, output_size)
		self.tanh = nn.Tanh()

	def forward(self, encoded_input, hidden):
		decoded_output, hidden = self.lstm(encoded_input, hidden)
		decoded_output = self.tanh(decoded_output)
		# decoded_output = self.sigmoid(decoded_output)
		decoded_output = self.dropout(decoded_output)
		decoded_output = self.linear(decoded_output)
		# decoded_output = self.sigmoid(self.linear(decoded_output))
		return decoded_output, hidden

class Seq2Seq(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, pred_length, num_node,
              output_size=2, dropout=0.5, teacher_forcing_ratio=0.5):
		super(Seq2Seq, self).__init__()
		self.pred_length = pred_length
		self.teacher_forcing_ratio = teacher_forcing_ratio
		self.num_node = num_node

		self.encoder = EncoderRNN(input_size, hidden_size, num_layers)
		self.decoder = DecoderRNN(hidden_size, output_size, num_layers, dropout)

	def forward(self, in_data, last_location, teacher_location=None):
		encoded_output, hidden = self.encoder(in_data) # (N*V, T, C) -> (N*V, T, H), (L, N*V, H)
		outputs = torch.zeros(in_data.shape[0], self.pred_length, self.decoder.output_size).to(last_location.device) # (N*V, T, 2)
		decoder_input = last_location # (N*V,1,2)
		for t in range(self.pred_length):
			now_out, hidden = self.decoder(decoder_input, hidden) #(N*V,1,2), (L,N*V,H)
			# now_out += decoder_input
			outputs[:,t:t+1] = now_out
			teacher_force = np.random.random() < self.teacher_forcing_ratio
			decoder_input = (teacher_location[:,t:t+1] if (type(teacher_location) is not type(None)) and teacher_force else now_out)

		return outputs # (N*V, T, 2)

	def reshape_for_lstm(self, feature):
		# prepare for skeleton prediction model
		'''
		N: batch_size
		C: channel
		T: time_step
		V: nodes
		'''
		N, C, T, V = feature.size()
		now_feat = feature.permute(0, 3, 2, 1).contiguous() # to (N, V, T, C)
		now_feat = now_feat.view(N*V, T, C)
		return now_feat

	def reshape_from_lstm(self, predicted):
		NV, T, C = predicted.size()
		predicted = predicted.view(-1, self.num_node, T, C)
		predicted = predicted.permute(0, 3, 2, 1).contiguous() # (N, C, T, V)
		return predicted

	def reshape_for_context(self, feature):
		NV, H = feature.size()
		now_feat = feature.view(-1, self.num_node, H) # (N, V, H)
		return now_feat