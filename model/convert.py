import onnx
import torch
from pytorch2keras import pytorch_to_keras
from torch import nn

from model.constants import *
from onnx_tf.backend import prepare
import tensorflow as tf
from onnx2keras import onnx_to_keras

from model.lofi2lofi_model import Decoder




class Decoder2(nn.Module):
    def __init__(self, device):
        super(Decoder2, self).__init__()
        self.device = device

        self.chords_lstm = nn.LSTMCell(input_size=HIDDEN_SIZE * 1, hidden_size=HIDDEN_SIZE * 1)
        self.chord_embeddings = nn.Linear(in_features=CHORD_PREDICTION_LENGTH, out_features=HIDDEN_SIZE)
        self.chord_prediction = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE, out_features=CHORD_PREDICTION_LENGTH)
        )
        self.chord_embedding_downsample = nn.Linear(in_features=2*HIDDEN_SIZE, out_features=HIDDEN_SIZE)

        self.melody_embeddings = nn.Linear(in_features=MELODY_PREDICTION_LENGTH, out_features=HIDDEN_SIZE)
        self.melody_lstm = nn.LSTMCell(input_size=HIDDEN_SIZE * 1, hidden_size=HIDDEN_SIZE * 1)
        self.melody_prediction = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE, out_features=MELODY_PREDICTION_LENGTH)
        )
        self.melody_embedding_downsample = nn.Linear(in_features=3*HIDDEN_SIZE, out_features=HIDDEN_SIZE)

        self.key_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE2, out_features=NUMBER_OF_KEYS),
        )
        self.mode_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE2, out_features=NUMBER_OF_MODES),
        )
        self.tempo_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE2, out_features=1),
        )
        self.valence_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE2, out_features=1),
        )
        self.energy_linear = nn.Sequential(
            nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_SIZE2, out_features=1),
        )

    def forward(self, z):
        tempo_output = self.tempo_linear(z)
        key_output = self.key_linear(z)
        mode_output = self.mode_linear(z)
        valence_output = self.valence_linear(z)
        energy_output = self.energy_linear(z)

        batch_size = 1
        hx_chords = torch.zeros(batch_size, 100, device=self.device)
        cx_chords = torch.zeros(batch_size, 100, device=self.device)
        hx_melody = torch.zeros(batch_size, 100, device=self.device)
        cx_melody = torch.zeros(batch_size, 100, device=self.device)

        chord_outputs = []
        melody_outputs = []

        # the chord LSTM input at first only consists of z
        # after the first iteration, we use the chord embeddings
        chord_embeddings = z
        melody_embeddings = None  # these will be set in the very first iteration

        for i in range(50):
            hx_chords, cx_chords = self.chords_lstm(chord_embeddings, (hx_chords, cx_chords))
            chord_prediction = self.chord_prediction(hx_chords)
            chord_outputs.append(chord_prediction)

            chord_embeddings = self.chord_embeddings(chord_prediction)
            chord_embeddings = self.chord_embedding_downsample(torch.cat((chord_embeddings, z), dim=1))

            if melody_embeddings is None:
                melody_embeddings = chord_embeddings
            for j in range(8):
                hx_melody, cx_melody = self.melody_lstm(melody_embeddings, (hx_melody, cx_melody))
                melody_prediction = self.melody_prediction(hx_melody)
                melody_outputs.append(melody_prediction)

                melody_embeddings = self.melody_embeddings(melody_prediction)
                melody_embeddings = self.melody_embedding_downsample(torch.cat((melody_embeddings, chord_embeddings, z), dim=1))

        key = key_output
        mode = mode_output
        bpm = min(1, max(0, tempo_output)) * 30 + 70
        energy = min(1, max(0, energy_output))
        valence = min(1, max(0, valence_output))

        return chord_outputs, melody_outputs, key, mode, bpm, energy, valence


mu = torch.zeros(1, 100)
model = Decoder2(device="cpu")
model.load_state_dict(torch.load("../checkpoints/test.pth"))

torch.onnx.export(model, mu, "model.onnx", opset_version=11, verbose=True)
