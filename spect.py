import librosa
import librosa.display

faye = "C:/Users/Faye/Downloads/got_s2e8_die_4u.wav"
sample, signal_processing = librosa.load(faye)
whale_song, _ = librosa.effects.trim(sample)
librosa.display.waveplot(whale_song, sr=signal_processing)
