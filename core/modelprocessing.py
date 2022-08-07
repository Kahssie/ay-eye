from core import app
from flask import request
import os
from mido import Message, MidiFile, MidiTrack
import torch
import torch.nn as nn
import time
import random

import murderminer as m

counter = 0

# ROUTE TO GENERATE MODEL 
@app.route('/generate', methods=['POST'])
def generate():
    global counter
    counter += 1
    if request.method == 'POST':
        inp = init_seq()
        gen_z = generate_seq(inp)
        name = "midi_" + counter + ".mid"
        mid_out = generate_midi(gen_z, name)
        print("midi out!", mid_out)
        

# CONSTANTS
PITCH_NAMES = ['C','D-','D','E-','E','F','F#','G','A-','A','B-','B']
PITCH_EQUALS = {"G-":"F#", "D-":"C#","A-":"G#","E-":"D#","B-":"A#"}
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

MODEL_PATH = os.path.join(os.getcwd(),"models\mse_adam_w_weightdecay.pth")
#print("MODEL PATH:", MODEL_PATH)

OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)

WINDOW_SIZE = 5 # Predict next note given first (WINDOW_SIZE - 1) notes
SEQ_LEN = 100



# MODEL DEFINITIONS
# Input: Array of note vectors
class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2, dropout=0.2):
        super(LSTMNetwork, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, 256, num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(256, 512, num_layers=num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(512, 256, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, input_dim)
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x = self.leaky_relu(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = self.leaky_relu(x)
    
        x, _ = self.lstm3(x)
        x = self.dropout(x)
        x = self.fc1(x)
        out = self.leaky_relu(x)

        out = torch.mean(out, 0)
        return out

# MIDI GENERATION
#model = LSTMNetwork(7)#.cuda()
print("LOADING MODEL")
#model.load_state_dict(torch.load(MODEL_PATH))
model = torch.load(MODEL_PATH)

# generate a track
def init_seq():
    random.seed(time.time()) # seed random
    inp = []
    for i in range (WINDOW_SIZE-1):
        starter_pitch_name = random.choice(PITCH_NAMES)
        starter_pitch_ind = m.pitch_name_to_pitch_index[starter_pitch_name]
        spiral_starter= m.pitch_index_to_position(starter_pitch_ind)
        starter = [1.0, spiral_starter[0], spiral_starter[1], spiral_starter[2], 57.0, 5.0, 5.0]
        inp.append(starter)
    return inp

# function returns a sequence of arrays that encode the information for a midi
def generate_seq(inp):
    gen_z = inp
    with torch.no_grad():
        model.eval()
    for i in range(SEQ_LEN - len(inp)):
        gen_inp = [torch.Tensor(x).cuda() for x in gen_z[-(WINDOW_SIZE-1):]]
        tensor_input = torch.stack(gen_inp, dim=0).cuda()
        note = model(tensor_input) # generate the note based on last note generated 
        gen_z.append(note)
    return gen_z

def position_to_pitch_index(pos):
    radius = 1.0
    verticalStep = 0.4
    if pos[1]==radius:
        c = 0
    if pos[0]==radius:
        c = 1
    if pos[1]==-1*radius:
        c = 2
    if pos[0]==-1*radius:
        c = 3
  # c = pitch_index - (4 * (pitch_index // 4))  #need to reverse engineer
    pitch_index = pos[2]/verticalStep
    return pitch_index

def note_to_number(note: str, octave: int) -> int:
    if note in PITCH_EQUALS.keys():
        note = PITCH_EQUALS[note]
    # assert note in NOTES, errors['notes']
    # assert octave in OCTAVES, errors['notes']

    note = NOTES.index(note)
    note += (NOTES_IN_OCTAVE * octave)

    # assert 0 <= note <= 127, errors['notes']

    return note

def note_from_pitch_ind(pitch_ind):
    val_list = list(m.pitch_name_to_pitch_index.values())
    key_list = list(m.pitch_name_to_pitch_index.keys())
    pitch_ind_n = (int(pitch_ind) % len(val_list))-6
    position = val_list.index(pitch_ind_n )
    note = key_list[position]
    return note 

dilation = 5

# code assumes that the values outputed are all valid
def generate_midi(model_output, midi_name):
    time=0
    gen_mid = MidiFile()
    gen_track = MidiTrack()
    gen_mid.tracks.append(gen_track)
    for x in model_output:
        pos = [x[1], x[2], x[3]]
        pitch_index = position_to_pitch_index(pos)
        letter_note = note_from_pitch_ind(pitch_index)
    if type(x) != list:
        x = x.tolist()
    # print(x)
    midi_number = note_to_number(letter_note, round(x[-1]))
    # midi_number = int(midi_number)
    # gen_track.append(Message(onoff, note=midi_number, velocity=x[4], time=x[5]))
    gen_track.append(Message("note_on", channel=3, note=midi_number, velocity=57, time=abs(int(time))))
    time += x[5] + dilation
    gen_track.append(Message("note_off", channel=3, note=midi_number, velocity=57, time=abs(int(time))))

    # print(len(gen_track))
    # print(x[-1])
    #gen_mid.save(midi_name)
    gen_mid.save(os.path.join(app.config['UPLOAD_FOLDER'], midi_name))
    return gen_mid