"""
lalala

"""

import numpy as np
import matplotlib.pyplot as plt
class FrontEnd:

    def __init__(self, samp_rate=16000, frame_duration=0.025, frame_shift=0.010, preemphasis=0.97,
                 num_mel=40, lo_freq=0, hi_freq=None, mean_norm_feat=True, mean_norm_wav=True, compute_stats=False):
        self.samp_rate = samp_rate
        self.win_size = int(np.floor(frame_duration * samp_rate))
        self.win_shift = int(np.floor(frame_shift * samp_rate))
        self.lo_freq = lo_freq
        if (hi_freq == None):
            self.hi_freq = samp_rate//2
        else:
            self.hi_freq = hi_freq

        self.preemphasis = preemphasis
        self.num_mel = num_mel
        self.fft_size = 2
        while (self.fft_size<self.win_size):
            self.fft_size *= 2

        self.hamwin = np.hamming(self.win_size)

        self.make_mel_filterbank()
        self.mean_normalize = mean_norm_feat
        self.zero_mean_wav = mean_norm_wav
        self.global_mean = np.zeros([num_mel])
        self.global_var = np.zeros([num_mel])
        self.global_frames = 0
        self.compute_global_stats = compute_stats
    # linear-scale frequency (Hz) to mel-scale frequency
    def lin2mel(self,freq):
        return 2595*np.log10(1+freq/700)

    # mel-scale frequency to linear-scale frequency
    def mel2lin(self,mel):
        return (10**(mel/2595)-1)*700

    def make_mel_filterbank(self):

        lo_mel = self.lin2mel(self.lo_freq)
        hi_mel = self.lin2mel(self.hi_freq)

        # uniform spacing on mel scale
        mel_freqs = np.linspace(lo_mel, hi_mel,self.num_mel+2)

        # convert mel freqs to hertz and then to fft bins
        bin_width = self.samp_rate/self.fft_size # typically 31.25 Hz, bin[0]=0 Hz, bin[1]=31.25 Hz,..., bin[256]=8000 Hz
        mel_bins = np.floor(self.mel2lin(mel_freqs)/bin_width)

        num_bins = self.fft_size//2 + 1
        self.mel_filterbank = np.zeros([self.num_mel,num_bins])
        for i in range(0,self.num_mel):
            left_bin = int(mel_bins[i])
            center_bin = int(mel_bins[i+1])
            right_bin = int(mel_bins[i+2])
            up_slope = 1/(center_bin-left_bin)
            for j in range(left_bin,center_bin):
                self.mel_filterbank[i,j] = (j - left_bin)*up_slope
            down_slope = -1/(right_bin-center_bin)
            for j in range(center_bin,right_bin):
                self.mel_filterbank[i,j] = (j-right_bin)*down_slope

    def plot_mel_matrix(self):
        for i in range(0, self.num_mel):
            plt.plot(self.mel_filterbank[i,:])
        plt.show()

    def dither(self, wav):
        n = 2*np.random.rand(wav.shape[0])-1
        n *= 1/(2**15)
        return wav + n

    def pre_emphasize(self, coef:0.97):
        
        super().__init__()
        self.coef = coef
        self.preemphasis=self.wav
        
                self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 3, 'The number of dimensions of input tensor must be 3!'
        # reflect padding to match lengths of in/out
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter)


class InversePreEmphasis(torch.nn.Module):
    """
    Implement Inverse Pre-emphasis by using RNN to boost up inference speed.
    """

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.rnn = torch.nn.RNN(1, 1, 1, bias=False, batch_first=True)
       
        self.rnn.weight_ih_l0.data.fill_(1)
    
        self.rnn.weight_hh_l0.data.fill_(self.coef)

    def forward(self, input: torch.tensor) -> torch.tensor:
        x, _ = self.rnn(input.transpose(1, 2))
        return x.transpose(1, 2)
        # apply pre-emphasis filtering on waveform
        preemph_wav = [x]
        return preemph_wav

    def wav_to_frames(self, wav):
 
        num_frames = int(np.floor((wav.shape[0] - self.win_size) / self.win_shift) + 1)
        frames = np.zeros([self.win_size, num_frames])
        for t in range(0, num_frames):
            frame = wav[t * self.win_shift:t * self.win_shift + self.win_size]
            if (self.zero_mean_wav):
                frame = frame - np.mean(frame)
            frames[:, t] = self.hamwin * frame
        return frames

    def frames_to_magspec(self, frames):
        self.frame=self.frames
        spectrum = np.fft.fft(slices, axis=0)[:M // 2 + 1:-1]
        spectrum = np.abs(spectrum)
        magspec=[spectrum]
        return magspec
    print(self.frame=self.magspec)
    
   def magspec_to_fbank(self, magspec):
        self.mag=self.magspec
        logSpectrum = numpy.log(magspec)
        fbank = [logspectrum]
        return fbank
    print(self.mag=self.fbank)

    # compute the mean vector of fbank coefficients in the utterance and subtract it from all frames of fbank coefficients
    def mean_norm_fbank(self, fbank):
        self.flanky=self.flank
        low_freq_mel = 0
        high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
           f_m_minus = int(bin[m - 1])   # left
           f_m = int(bin[m])             # center
           f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
           fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
           fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
           filter_banks = numpy.dot(pow_frames, fbank.T)
           filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
           filter_banks = 20 * numpy.log10(filter_banks)
           fbank=filter_banks
           return fbank

 
    def accumulate_stats(self, fbank):
        self.global_mean += np.sum(fbank,axis=1)
        self.global_var += np.sum(fbank**2,axis=1)
        self.global_frames += fbank.shape[1]

   
    def compute_stats(self):
        self.global_mean /= self.global_frames
        self.global_var /= self.global_frames
        self.global_var -= self.global_mean**2

        return self.global_mean, 1.0/np.sqrt(self.global_var)

    def process_utterance(self, utterance):

        wav     = self.dither(utterance)
        wav     = self.pre_emphasize(wav)
        frames  = self.wav_to_frames(wav)
        magspec = self.frames_to_magspec(frames)
        fbank   = self.magspec_to_fbank(magspec)
        if (self.mean_normalize):
            fbank = self.mean_norm_fbank(fbank)

        if (self.compute_global_stats):
            self.accumulate_stats(fbank)

        return fbank


import struct
import numpy as np
import sys

def write_htk_user_feat(x, name='filename'):
    default_period = 100000 # assumes 0.010 ms frame shift
    num_dim = x.shape[0]
    num_frames = x.shape[1]
    hdr = struct.pack(
        '>iihh',  # the beginning '>' says write big-endian
        num_frames,  # nSamples
        default_period,  # samplePeriod
        4*num_dim,  # 2 floats per feature
        9)  # user features

    out_file = open(name, 'wb')
    out_file.write(hdr)

    for t in range(0, num_frames):
        frame = np.array(x[:,t],'f')
        if sys.byteorder == 'little':
            frame.byteswap(True)
        frame.tofile(out_file)

    out_file.close()

def read_htk_user_feat(name='filename'):
    f = open(name,'rb')
    hdr = f.read(12)
    num_samples, samp_period, samp_size, parm_kind = struct.unpack(">IIHH", hdr)
    if parm_kind != 9:
        raise RuntimeError('feature reading code only validated for USER feature type for this lab. There is other publicly available code for general purpose HTK feature file I/O\n')

    num_dim = samp_size//4

    feat = np.zeros([num_samples, num_dim],dtype=float)
    for t in range(num_samples):
        feat[t,:] = np.array(struct.unpack('>' + ('f' * num_dim), f.read(samp_size)),dtype=float)

    return feat


def write_ascii_stats(x,name='filename'):
    out_file = open(name,'w')
    for t in range(0, x.shape[0]):
        out_file.write("{0}\n".format(x[t]))
    out_file.close()



import htk_io as htk
import python_speech_features as sp
import argparse


data_dir = "../Downloads/dev-clean/LibriSpeech"

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Convert audio files to speech recognition features in batch mode. "
                                                 "Must specify train, dev, or test set. If train set is specified, "
                                                 "global mean and variance of features are computed for use in acoustic model training.\n")

    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
args, unknown = parser.parse_known_args()



if args.__setattr__ == "train":
    compute_stats=True
else:
    compute_stats=False
    wav_list = os.path.join(data_dir,"lists/wav_{0}.list".format(args.__setattr__))
    feat_list = os.path.join(data_dir,"lists/feat_{0}.rscp".format(args.__setattr__))
    feat_dir = os.path.join(data_dir,"feat")
    rscp_dir = "..." # note ... is CNTK notation for "relative to the location of the list of feature files
    mean_file = os.path.join(data_dir,"am/feat_mean.ascii")
    invstddev_file = os.path.join(data_dir,"am/feat_invstddev.ascii")
    wav_dir = ".."

    #if not os.path.exists(os.path.join(data_dir,'am')):
        #os.mkdir(os.path.join(data_dir,'am'))


    samp_rate = 16000
    fe = sp.FrontEnd(samp_rate=samp_rate, mean_norm_feat=True)
    # read lines

    with open(wav_list) as f:
        wav_files = f.readlines()
        wav_files = [x.strip() for x in wav_files]

    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    if not os.path.exists(os.path.dirname(feat_list)):
        os.makedirs(os.path.dirname(feat_list))
    out_list = open(feat_list,"w")
    count = 0
    for line in wav_files:

        wav_name = os.path.basename(line)
        root_name, wav_ext = os.path.splitext(wav_name)
        wav_file = os.path.join(wav_dir, line)
        feat_name = root_name + '.feat'
        feat_file = os.path.join(feat_dir , feat_name)
        x, s = sf.read(wav_file)

        if (s != samp_rate):
            raise RuntimeError("Laboratory code assumes 16 kHz audio files!")

        feat = fe.process_utterance(x)
        htk.write_htk_user_feat(feat, feat_file)
        feat_rscp_line = os.path.join(rscp_dir, '..', 'feat', feat_name)
        print("Wrote", feat.shape[1], "frames to", feat_file)
        out_list.write("%s=%s[0,%d]\n" % (feat_name, feat_rscp_line,feat.shape[1]-1))
        count += 1
    out_list.close()

    print("Processed", count, "files.")
    if (compute_stats):
        m, p = fe.compute_stats() # m=mean, p=precision (inverse standard deviation)
        htk.write_ascii_stats(m, mean_file)
        print("Wrote global mean to", mean_file)
        htk.write_ascii_stats(p, invstddev_file)
        print("Word global inv stddev to ", invstddev_file)

data_dir="../Downloads/dev-clean"
wav_file='../LibriSpeech/dev-clean/2428/83699/2428-83699-0000.flac'
feat_file=os.path.join(data_dir,'feat/2428-83699-0000.feat')
plot_output=True

if not os.path.isfile(wav_file):
    raise RuntimeError('input wav file is missing. Have you downloaded the LibriSpeech corpus?')

if not os.path.exists(os.path.join(data_dir,'feat')):
    os.mkdir(os.path.join(data_dir,'feat'))

samp_rate = 16000

s,x = sf.read(wav_file)
if (s != samp_rate):
    raise RuntimeError("LibriSpeech files are 16000 Hz, found {0}".format(s))

fe = sp.FrontEnd(samp_rate=samp_rate,mean_norm_feat=True)


feat = fe.process_utterance(x)

if (plot_output):
    if not os.path.exists('Downloads'):
        os.mkdir('Downloads')

    # plot waveform
    plt.plot(x)
    plt.title('waveform')
    plt.savefig('Downloads/waveform.png', bbox_inches='tight')
    plt.close()

    # plot mel filterbank
    for i in range(0, fe.num_mel):
        plt.plot(fe.mel_filterbank[i, :])
    plt.title('mel filterbank')
    plt.savefig('Desktop/mel_filterbank.png', bbox_inches='tight')
    plt.close()

    # plot log mel spectrum (fbank)
    plt.imshow(feat, origin='lower', aspect=4) # flip the image so that vertical frequency axis goes from low to high
    plt.title('log mel filterbank features (fbank)')
    plt.savefig('Desktop/fbank.png', bbox_inches='tight')
    plt.close()

htk.write_htk_user_feat(feat, feat_file)
print("Wrote {0} frames to {1}".format(feat.shape[1], feat_file))

