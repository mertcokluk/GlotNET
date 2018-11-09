from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio
from nnmnkwii.datasets import cmu_arctic
from nnmnkwii.io import hts
from nnmnkwii import preprocessing as P
from hparams import hparams
from os.path import exists
import librosa
import scipy.signal as dsp
from scipy.linalg import solve_toeplitz as lpcsolve

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    speakers = cmu_arctic.available_speakers

    wd = cmu_arctic.WavFileDataSource(in_dir, speakers=speakers)
    wav_paths = wd.collect_files()
    speaker_ids = wd.labels

    for index, (speaker_id, wav_path) in enumerate(
            zip(speaker_ids, wav_paths)):
        futures.append(executor.submit(
            partial(_process_utterance, out_dir, index + 1, speaker_id, wav_path, "N/A")))
    return [future.result() for future in tqdm(futures)]


def start_at(labels):
    has_silence = labels[0][-1] == "pau"
    if not has_silence:
        return labels[0][0]
    for i in range(1, len(labels)):
        if labels[i][-1] != "pau":
            return labels[i][0]
    assert False


def end_at(labels):
    has_silence = labels[-1][-1] == "pau"
    if not has_silence:
        return labels[-1][1]
    for i in range(len(labels) - 2, 0, -1):
        if labels[i][-1] != "pau":
            return labels[i][1]
    assert False


def _process_utterance(out_dir, index, speaker_id, wav_path, text):
    sr = hparams.sample_rate

    # Load the audio to a numpy array. Resampled if needed
    wav = audio.load_wav(wav_path)

    lab_path = wav_path.replace("wav/", "lab/").replace(".wav", ".lab")

    # Trim silence from hts labels if available
    # TODO
    if exists(lab_path) and False:
        labels = hts.load(lab_path)
        b = int(start_at(labels) * 1e-7 * sr)
        e = int(end_at(labels) * 1e-7 * sr)
        wav = wav[b:e]
        wav, _ = librosa.effects.trim(wav, top_db=20)
    else:
        wav, _ = librosa.effects.trim(wav, top_db=20)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    out = wav
    constant_values = 0.0
    out_dtype = np.float32

    p_vt=5
    p_gl=5
    d=0.99
    hpfilt=2
    preflt = p_vt+1

    # High-pass filter speech in order to remove possible low frequency
    # fluctuations (Linear-phase FIR, Fc = 70 Hz)
    Fstop = 40                        # Stopband Frequency
    Fpass = 70                        # Passband Frequency
    Nfir = np.round(300/16000*fs)  # FIR numerator order

    if (Nfir % 2==0):
        Nfir = Nfir + 1

    '''
    it is very very expensive to calculate the firls filter! However, as 
    long as the fs does not change, the firls filter does not change.
    Therefore, the computed filter is returned and can be passed to this
    function later on to avoid the calculated of the (same) filter.
    '''
    B = dsp.firls(Nfir, [0, Fstop/(fs/2), Fpass/(fs/2), fs/2], [0, 0, 1, 1], [1, 1])



    '''
    % Estimate the combined effect of the glottal flow and the lip radiation
    % (Hg1) and cancel it out through inverse filtering. Note that before
    % filtering, a mean-normalized pre-frame ramp is appended in order to
    % diminish ripple in the beginning of the frame. The ramp is removed after
    % filtering.
    '''
    le=np.int(len(wav)/hparams.hop_size)
    glot=np.zeros([le, 254])
    vtfilter=np.zeros([le, 5])
    
    for j in range(le):
        w = wav[hparams.hop_size*(j):hparams.hop_size*(j+1)]
        
        #print(wav[(hparams.hop_size)*(i):(hparams.hop_size)*(i+1)])
        for i in range(hpfilt):
            w = dsp.lfilter(B, 1, np.concatenate((w, np.zeros(np.int(len(B)/2)-1))))
            w = w[np.int(len(B)/2):]
        
        win = np.hanning(len(w))
        a=np.array([1])
        b=np.array([1, -d])
        signal = np.concatenate((np.linspace(-w[0],w[0],preflt),  w))
        
        windowed = w*win
        t=dsp.correlate(windowed,windowed)
        l=len(win)
        Hg1 = lpcsolve(t[l:l+1],t[l+1:l+2]);
        y = dsp.lfilter(Hg1,1,signal);
        y = y[preflt:];

        '''
        % Estimate the effect of the vocal tract (Hvt1) and cancel it out through
        % inverse filtering. The effect of the lip radiation is canceled through
        % intergration. Signal g1 is the first estimate of the glottal flow.
        '''
        
        windowedy = y*win
        r=dsp.correlate(windowedy,windowedy)
        Hvt1 = lpcsolve(r[l:l+p_vt],r[l+1:l+p_vt+1]);
        g1 = dsp.lfilter(Hvt1,1,signal);
        g1 = dsp.lfilter(a, b, g1);
        g1 = g1[preflt:];

        '''
        % Re-estimate the effect of the glottal flow (Hg2). Cancel the contribution
        % of the glottis and the lip radiation through inverse filtering and
        % integration, respectively.
        '''
        windowedg1 = w*g1
        u=dsp.correlate(windowedg1,windowedg1)
        Hg2 = lpcsolve(u[l:l+p_gl],u[l+1:l+p_gl+1]);
        y = dsp.lfilter(Hg2,1,signal);
        y = dsp.lfilter(a, b, y);
        y = y[preflt:];

        '''
        % Estimate the model for the vocal tract (Hvt2) and cancel it out through
        % inverse filtering. The final estimate of the glottal flow is obtained
        % through canceling the effect of the lip radiation.
        '''
        windowedynew = y*win
        t=dsp.correlate(windowedynew,windowedynew)
        Hvt2 = lpcsolve(t[l:l+p_vt],t[l+1:l+p_vt+1]);
        dg = dsp.lfilter(Hvt2,1,signal);
        g = dsp.lfilter(a, b, dg);
        g = g[preflt:];
        dg = dg[preflt:];

        # Set vocal tract model to 'a' and glottal source spectral model to 'ag'
        a = Hvt2;
        ag = Hg2;
        
        print(j)
        glot[j-1]=g.T
        vtfilter[j-1]=a.T
    
 


    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
    # lws pads zeros internally before performing stft
    # this is needed to adjust time resolution between audio and mel-spectrogram
    l, r = audio.lws_pad_lr(wav, hparams.fft_size, audio.get_hop_size())

    # zero pad for quantized signal
    out = np.pad(out, (l, r), mode="constant", constant_values=constant_values)
    N = mel_spectrogram.shape[0]
    assert len(out) >= N * audio.get_hop_size()

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[:N * audio.get_hop_size()]
    assert len(out) % audio.get_hop_size() == 0

    timesteps = len(out)

    # Write the spectrograms to disk:
    audio_filename = 'cmu_arctic-audio-%05d.npy' % index
    glot_filename = 'cmu_arctic-glot-%05d.npy' % index
    vt_filename = 'cmu_arctic-vt-%05d.npy' % index
    mel_filename = 'cmu_arctic-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, audio_filename),
            out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(out_dir, glot_filename),
            glot.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(out_dir, vt_filename),
            vtfilter.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return (audio_filename, mel_filename, glot_filename, vt_filename, timesteps, text, speaker_id)
