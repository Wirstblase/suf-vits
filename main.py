#matplotlib inline
import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


import soundfile as sf #to save the resulting audio instead of just playing it

'''
update: made the variable names more meaningful

    text_norm is changed to text_normalized.
    hps is changed to hparams.
    net_g is changed to synthesizer.
    stn_tst is changed to input_text.
    x_tst is changed to input_text_cuda.
    x_tst_lengths is changed to input_text_lengths.
    sid is changed to speaker_id.
    audio is changed to generated_audio.
'''


def get_text(text, hparams):
    text_normalized = text_to_sequence(text, hparams.data.text_cleaners)
    if hparams.data.add_blank:
        text_normalized = commons.intersperse(text_normalized, 0)
    text_normalized = torch.LongTensor(text_normalized)
    return text_normalized


# Load hyperparameters
hparams = utils.get_hparams_from_file("./configs/vctk_base.json")

# Initialize the synthesizer
synthesizer = SynthesizerTrn(
    len(symbols),
    hparams.data.filter_length // 2 + 1,
    hparams.train.segment_size // hparams.data.hop_length,
    n_speakers=hparams.data.n_speakers,
    **hparams.model).cuda()
synthesizer.eval()

# Load the pretrained model
utils.load_checkpoint("pretrained/pretrained_vctk.pth", synthesizer, None)

# Prepare the input text
input_text = get_text("Here's a sound recording I generated with VITS. (using espeak and their pree trained model).. - It works well!", hparams)
with torch.no_grad():
    input_text_cuda = input_text.cuda().unsqueeze(0)
    input_text_lengths = torch.LongTensor([input_text.size(0)]).cuda()
    speaker_id = torch.LongTensor([5]).cuda()

    # Perform inference
    generated_audio = synthesizer.infer(
        input_text_cuda,
        input_text_lengths,
        sid=speaker_id,
        noise_scale=.667,
        noise_scale_w=0.8,
        length_scale=1
    )[0][0, 0].data.cpu().float().numpy()

# Save the audio to a file
output_file = "output/output_audio.wav"
sf.write(output_file, generated_audio, hparams.data.sampling_rate)

# Display a message indicating where the file has been saved
print(f"Audio saved to {output_file}")