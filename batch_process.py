import os
import glob
import subprocess
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

try:
    path = sys.argv[1]
    proc_type = sys.argv[2]
    if len(sys.argv) > 3:
        args = sys.argv[3:]
    else:
        args = []
    
    def load_audio(file):
        audio, sr = librosa.load(file, sr=None, mono=False)
        if audio.ndim > 1:
            audio = np.asfortranarray(audio).T
        return audio, sr

    def samples_to_time(n, sr):
        time = (n*1000) // sr
        ms = time % 1000
        time = time // 1000
        sec = time % 60
        time = time // 60
        mins = time % 60
        time = time // 60
        hour = time
        if hour > 0:
            return f"{hour}:{mins:02}:{sec:02}.{ms:03}"
        else:
            return f"{mins}:{sec:02}.{ms:03}"

    def to_db(y):
        return f"{(20*np.log10(y)):.2f} dB"

    def specshow(file):
        y, sr = librosa.load(file, sr=None)
        
        fig, ax = plt.subplots(nrows=3, sharex=True)
        librosa.display.waveshow(y, sr=sr, ax=ax[0])
        ax[0].set(title=file)
        ax[0].label_outer()    
        
        mel_spect = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max, top_db=144.0)
        img = librosa.display.specshow(mel_spect, sr=sr, ax=ax[1], x_axis='time', y_axis='linear');
        img.cmap = librosa.display.cmap(mel_spect, cmap_seq='magma', robust=False)
        
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max, top_db=144.0)
        img = librosa.display.specshow(mel_spect, sr=sr, ax=ax[2], x_axis='time', y_axis='mel');
        img.cmap = librosa.display.cmap(mel_spect, cmap_seq='magma', robust=False)
        fig.colorbar(img, ax=ax, format='%+2.0f dB');
        plt.show()

    def stats(file):
        audio, sr = load_audio(file)
        if audio.ndim == 1:
            peak = 0
            peaki = 0
            rms = np.sqrt(np.mean(audio**2)*2)
            for i in range(audio.shape[0]):
                if abs(audio[i]) > peak:
                    peak = abs(audio[i])
                    peaki = i
            return samples_to_time(audio.shape[0], sr), "mono", sr, [(to_db(peak), samples_to_time(peaki, sr))], [to_db(rms)]
        else:
            peak = [0]
            peak *= audio.shape[1]
            peaki = [0]
            peaki *= audio.shape[1]
            rms = [0]
            rms *= audio.shape[1]
            for j in range(audio.shape[1]):
                rms[j] = np.sqrt(np.mean(audio[:, j]**2)*2)
                for i in range(audio.shape[0]):
                    if abs(audio[i, j]) > peak[j]:
                        peak[j] = abs(audio[i, j])
                        peaki[j] = i
            return samples_to_time(audio.shape[0], sr), f"{audio.shape[1]} ch", sr, [(to_db(peak[j]), samples_to_time(peaki[j], sr)) for j in range(audio.shape[1])], [to_db(rms[j]) for j in range(audio.shape[1])]

    def save_audio(file, audio, sr):
        sf.write(file, audio, sr, subtype="FLOAT", format='WAV')

    def slice_audio(file, first=0, last=0):
        print(f"slice_audio ({first}s, {last}s): {file}")
        audio, sr = load_audio(file)
        if first > 0:
            audio = audio[int(first*sr):]
        if last > 0:
            audio = audio[:int(-last*sr)]
        save_audio(os.path.join(os.path.dirname(file), f"{os.path.basename(file)}"), audio, sr)

    def sil_audio(file, first=0, last=0):
        print(f"sil_audio ({first}s, {last}s): {file}")
        audio, sr = load_audio(file)
        if first > 0:
            if audio.ndim == 1:
                y = np.zeros(int(first*sr))
            else:
                y = np.zeros((int(first*sr), audio.shape[1]))
            audio = np.concatenate((y, audio), axis=0)
        if last > 0:
            if audio.ndim == 1:
                y = np.zeros(int(last*sr))
            else:
                y = np.zeros((int(last*sr), audio.shape[1]))
            audio = np.concatenate((audio, y), axis=0)
        save_audio(os.path.join(os.path.dirname(file), f"{os.path.basename(file)}"), audio, sr)

    def change_sr(file, newsr):
        print(f"change_sr ({newsr}): {file}")
        audio, sr = load_audio(file)
        save_audio(os.path.join(os.path.dirname(file), f"{os.path.basename(file)}"), audio, newsr)

    def match_audio(matches):
        audios = []
        sr = 0
        maxlen = 0
        for file in matches:
            audio, sr = load_audio(file)
            if audio.shape[0] > maxlen:
                maxlen = audio.shape[0]
            audios.append(audio)
        for i in range(len(audios)):
            if audios[i].shape[0] < maxlen:
                print(f"match_audio ({i+1}/{len(matches)}): {matches[i]}")
                if audios[i].ndim == 1:
                    y = np.zeros(maxlen-audios[i].shape[0])
                else:
                    y = np.zeros((maxlen-audios[i].shape[0], audios[i].shape[1]))
                audios[i] = np.concatenate((audios[i], y), axis=0)
                save_audio(os.path.join(os.path.dirname(matches[i]), f"{os.path.basename(matches[i])}"), audios[i], sr)

    def mix_audio(matches, nth):
        audios = []
        sr = 0
        maxlen = 0
        for file in matches:
            audio, sr = load_audio(file)
            audios.append(audio)
        print(f"mix_audio: {matches[nth]}")
        y = audios[0]
        for i in range(len(audios)):
            if i > 0:
                y += audios[i]
        save_audio(os.path.join(os.path.dirname(matches[nth]), f"{chr(148206)}{os.path.basename(matches[nth])}"), y, sr)

    if os.path.isfile(path):
        if path.split('.')[-1] == "wav":
            print(f"Processing: {path}")
            if proc_type == "specshow":
                specshow(path)
            elif proc_type == "stats":
                filestat = stats(path)
                print(f"{filestat[0]}:")
                print(f"\tlen {filestat[1][0]} {filestat[1][1]} sr {filestat[1][2]}")
                print(f"\tpeak {filestat[1][3]}")
                print(f"\trms {filestat[1][4]}")
            elif proc_type == "slice":
                first = float(args[0]) if len(args) > 0 else 0
                last = float(args[1]) if len(args) > 1 else 0
                slice_audio(path, first, last)
            elif proc_type == "sil":
                first = float(args[0]) if len(args) > 0 else 0
                last = float(args[1]) if len(args) > 1 else 0
                sil_audio(path, first, last)
            elif proc_type == "csr":
                newsr = int(args[0]) if len(args) > 0 else 0
                change_sr(path, newsr)
    elif os.path.isdir(path):
        print(path)
        firstfile = ""
        firstaudio = []
        sr = 0
        nmatch = int(args[0]) if proc_type == "match" and len(args) > 0 else 0
        nmatch = int(args[0]) if proc_type == "mix" and len(args) > 0 else 0
        matches = []
        filestats = []
        statout = ""
        namelen = 0
        ii = 0

        for subdir, _, files in os.walk(path):
            for file in files:
                if file.split('.')[-1] == "flac":
                    wavpath = os.path.join(subdir, file)
                    if proc_type == "names":
                        parent = os.path.basename(os.path.dirname(subdir))
                        print('.'.join(file.split('.')[:-1])+"-"+parent+"."+file.split('.')[-1])
                        os.rename(wavpath, os.path.join(subdir, '.'.join(file.split('.')[:-1])+"-"+parent+"."+file.split('.')[-1]))
                if file.split('.')[-1] == "wav":
                    wavpath = os.path.join(subdir, file)
                    if len(file) > namelen:
                        namelen = len(file)
                    if proc_type == "specshow":
                        specshow(wavpath)
                    elif proc_type == "stats":
                        if len(filestats) == 0:
                            f = open('.'.join(os.path.join(os.path.dirname(path), file).split('.')[:-1])+".txt", 'w', encoding="utf-8")
                        filestats.append((ii, file, stats(wavpath)))
                        print(f"{filestats[-1][0]}. {filestats[-1][1]}:")
                        print(f"\tlen {filestats[-1][2][0]} {filestats[-1][2][1]} sr {filestats[-1][2][2]}")
                        print(f"\tpeak {filestats[-1][2][3]}\trms {filestats[-1][2][4]}")
                    elif proc_type == "slice":
                        first = float(args[0]) if len(args) > 0 else 0
                        last = float(args[1]) if len(args) > 1 else 0
                        slice_audio(wavpath, first, last)
                    elif proc_type == "sil":
                        first = float(args[0]) if len(args) > 0 else 0
                        last = float(args[1]) if len(args) > 1 else 0
                        sil_audio(wavpath, first, last)
                    elif proc_type == "csr":
                        newsr = int(args[0]) if len(args) > 0 else 0
                        change_sr(wavpath, newsr)
                    elif proc_type == "join":
                        print(f"join: {wavpath}")
                        if firstfile == "":
                            firstfile = file
                            firstaudio, sr = load_audio(wavpath)
                        else:
                            firstaudio = np.concatenate((firstaudio, load_audio(wavpath)[0]), axis=0)
                    elif proc_type == "match":
                        matches.append(wavpath)
                        if len(matches) >= nmatch:
                            match_audio(matches)
                            matches = []
                    elif proc_type == "mix":
                        matches.append(wavpath)
                        if len(matches) >= nmatch:
                            nth = int(args[1]) if len(args) > 1 else 0
                            mix_audio(matches, nth)
                            matches = []
                    ii += 1
        if proc_type == "stats":
            sortby = args[0] if len(args) > 0 else 0
            if sortby == "name":
                filestats = sorted(filestats, key=lambda s: s[1])
            elif sortby == "len":
                filestats = sorted(filestats, key=lambda s: s[2][0])
            elif sortby == "peak":
                filestats = sorted(filestats, key=lambda s: sum([float(y[0][:-3]) for y in s[2][3]]))
            elif sortby == "rms":
                filestats = sorted(filestats, key=lambda s: sum([float(y[:-3]) for y in s[2][4]]))
            f.write(f"sortby: {sortby}\n\n")
            ii = 0
            for s in filestats:
                blank = " "
                blank *= namelen-len(s[1])+1
                f.write(f"{ii}/{s[0]}.   \t{s[1]}{blank}")
                f.write(f"\tlen {s[2][0]} {s[2][1]} sr {s[2][2]}")
                f.write(f"\tpeak {s[2][3]}\trms {s[2][4]}\n")
                ii += 1
            f.close()
        if proc_type == "join":
            save_audio(os.path.join(os.path.dirname(path), f"{chr(148206)}{firstfile}"), firstaudio, sr)
except Exception as err:
    print(f'sdasda{err}')

quit()
