from model import Model
import argparse
import torch
import torchaudio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--voice_path",type=str,help="Please Enter Your Voice Path")
arg = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model.load_state_dict(torch.load("weights/weights.pth"))
model.to(device)
model.eval()
labels=["maryam","zeynab","alireza","hossein","nahid","mohammadali","amir","parisa"]

signal, sample_rate = torchaudio.load(arg.voice_path)

# preprocess
signal = torch.mean(signal, dim=0, keepdim=True)
new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
signal = transform(signal)
signal = signal.unsqueeze(0).to(device)

# process
preds = model(signal)

# postprocess
preds = preds.cpu().detach().numpy()
output = np.argmax(preds)
print(labels[output])
