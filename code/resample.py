import librosa
import librosa.display
import os, glob
import soundfile as sf
from tqdm import tqdm

for item in tqdm(sorted(glob.glob('../dataset/librispeech/train-clean-100/*/*/*.wav'))):
     y, sr = librosa.load(item, mono=False, sr=16000)
     y_third = librosa.resample(y, orig_sr=sr, target_sr=8000)
     new_path = "/".join(item.split('/')[:-1]).replace('train-clean-100/', 'train-clean-100-8k/')
     os.makedirs(new_path, exist_ok=True)
     sf.write(os.path.join(new_path,item.split('/')[-1].replace("flac", "wav")), y_third.T, samplerate=8000)

for item in tqdm(sorted(glob.glob('../dataset/VCTK-Corpus/wav48/*/*.wav'))):
     if os.path.split(item)[0].split('/')[-1] == 'p267': break
     new_path = "/".join(item.split('/')[:-1]).replace('wav48/', 'wav48-8k/')+'/1'
     os.makedirs(new_path, exist_ok=True)
     if os.path.exists(os.path.join(new_path,item.split('/')[-1])): continue
     if len(sorted(glob.glob(new_path+'/*.wav'))) > 50: continue
     y, sr = librosa.load(item, sr=48000)
     y_third = librosa.resample(y, orig_sr=sr, target_sr=8000)
     sf.write(os.path.join(new_path,item.split('/')[-1]), y_third.T, samplerate=8000)

for item in tqdm(sorted(glob.glob('../dataset/data_aishell/wav/test/*/*.wav'))):
     new_path = "/".join(item.split('/')[:-1]).replace('test/', 'test-8k/')+'/1'
     os.makedirs(new_path, exist_ok=True)
     if os.path.exists(os.path.join(new_path,item.split('/')[-1])): continue
     if len(sorted(glob.glob(new_path+'/*.wav'))) > 50: continue
     y, sr = librosa.load(item, sr=48000)
     y_third = librosa.resample(y, orig_sr=sr, target_sr=8000)
     sf.write(os.path.join(new_path,item.split('/')[-1]), y_third.T, samplerate=8000)


for item in tqdm(sorted(glob.glob('../dataset/voxceleb1/vox1_test_wav/*/*/*.wav'))):
     y, sr = librosa.load(item, mono=False, sr=16000)
     y_third = librosa.resample(y, orig_sr=sr, target_sr=8000)
     new_path = "/".join(item.split('/')[:-1]).replace('wav/', 'wav-8k/')
     os.makedirs(new_path, exist_ok=True)
     sf.write(os.path.join(new_path,item.split('/')[-1].replace("flac", "wav")), y_third.T, samplerate=8000)

