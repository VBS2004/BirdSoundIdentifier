from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from torchaudio.prototype.pipelines import VGGISH
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import torch
import os
import glob

from torch.utils.data import DataLoader
import warnings


warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III subtype is unknown.*")

df = pd.read_csv("sample_submission.csv")
taxonomy=pd.read_csv("taxonomy.csv")

# Forward map: column name → index
forward_map = {col: i for i, col in enumerate(df.columns[1:])}

# Reverse map: index → column name
reverse_map = {i: col for col, i in forward_map.items()}

class BirdCLEFDataset:
    def __init__(self,audio_path=None, device=None, segment_duration=5):
        input_sr = VGGISH.sample_rate
        self.input_proc = VGGISH.get_input_processor()
        self.audio_path=audio_path
        self.device = device
        self.target_sample_rate = input_sr
        self.num_samples = 50000
        self.segment_duration = segment_duration
    
        self._create_test_segments()
    
    def _create_test_segments(self):
     
        self.test_audio_files = [self.audio_path]

        expanded_indices = []
        expanded_row_ids = []
        
        for idx, audio_path in enumerate(self.test_audio_files):
            file_name = os.path.basename(self.audio_path)
            info = torchaudio.info(self.audio_path)
            audio_duration_seconds = info.num_frames / info.sample_rate
            
            num_segments = max(1, int(audio_duration_seconds // self.segment_duration))
            
            for seg_idx in range(num_segments):
                start_time = seg_idx * self.segment_duration
                end_time = min((seg_idx + 1) * self.segment_duration, audio_duration_seconds)
                
                if end_time - start_time >= 1.0:
                    segment_id = (idx, start_time, end_time)
                    row_id = f"{file_name[:-4]}_{int(end_time)}"
                    
                    expanded_indices.append(segment_id)
                    expanded_row_ids.append(row_id)
        
        self.expanded_indices = expanded_indices
        self.expanded_row_ids = expanded_row_ids
    
    def __len__(self):
        return len(self.expanded_indices)

    
    def __getitem__(self, item):
        segment_info = self.expanded_indices[item]
        row_id = self.expanded_row_ids[item]
        
        file_idx, start_time, end_time = segment_info
        audio_path = self.test_audio_files[file_idx]
        
        signal, sr = self._load_audio_segment(audio_path, start_time, end_time)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        
        signal = self.input_proc(signal.squeeze(0))
        signal=signal.repeat(1,3,1,1)
        signal = signal.to(self.device)
        return signal
    
    def _load_audio_segment(self, audio_path, start_time, end_time):
        info = torchaudio.info(audio_path)
        sample_rate = info.sample_rate
        
        start_sample = int(start_time * sample_rate)
        if start_sample == 0 and end_time * sample_rate >= info.num_frames:
            return torchaudio.load(audio_path)
        
        num_frames = int((end_time - start_time) * sample_rate)
        signal, sr = torchaudio.load(audio_path, frame_offset=start_sample, num_frames=num_frames)
        return signal, sr
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        len_signal = signal.shape[1]
        if len_signal < self.num_samples:  # apply right pad
            num_missing_samples = self.num_samples - len_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device {device}")


class AudioCNN(nn.Module):
    def __init__(self, num_classes=206,apply_softmax=False):
        super(AudioCNN, self).__init__()

        self.apply_softmax=apply_softmax
        self.eff=torchvision.models.efficientnet_v2_s()

        in_features = self.eff.classifier[1].in_features  # Get in_features of the final linear layer
        self.eff.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        #x = F.interpolate(x, size=(384, 384), mode='bilinear')
        x = self.eff(x)            # Output: (B, 512, 6, 4)
        if self.apply_softmax:
            return torch.softmax(x,dim=1)
        return x

model=AudioCNN(apply_softmax=True)
model = nn.DataParallel(model)
model.load_state_dict(torch.load("best_loss.pth"))
model.to(device)
model.eval()

predictions=[]
row_ids=[]

def predict(file_name):
    bcdTest=BirdCLEFDataset(file_name,device=device)
    test_dataloader=DataLoader(bcdTest,batch_size=1,shuffle=False)

    with torch.no_grad():
        for song in test_dataloader:
            current_song_predicts=[]
            #print("Song:",song.shape)
            
            for segment in song:
                #print("Segment:",segment.shape)
                segment = segment.to(device)
                
                prediction= model(segment)
                current_song_predicts.append(prediction)
                
                current_song_prediction=prediction.mean(dim=0)
                
            predictions.append(current_song_prediction)

    primary_label=reverse_map[torch.argmax(current_song_prediction).item()]
    name=(taxonomy[taxonomy['primary_label']==primary_label]['scientific_name'].values[-1])

    print(name)
    return name,current_song_prediction[torch.argmax(current_song_prediction).item()].item()*100

if __name__=="__main__":
    predict("V:\\BirdSoundIdentifier\\uploads\\recording_1749898556906.wav")