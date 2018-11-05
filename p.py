# coding: utf-8
"""
Generate waveform from trained GlotNet.

usage: postprocess.py [options] <vocal-tract> <glottal-flow> <dst_dir>

options:
    --max-abs-value=<N>               Max abs value [default: -1].
    --file-name-suffix=<s>            File name suffix [default: ].
    --speaker-id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    -h, --help               Show help message.
"""
from docopt import docopt
from os.path import dirname, join, basename, splitext
import sys
import os
import numpy as np
import librosa
import scipy.signal as dsp

def postprocess(g=None, vt=None):
    length = g.shape[0]
    b = np.array([1, -0.99])
    w = np.zeros(1) 
    for j in length:
        gj = g[j]
        vtj = vt[j]
        vtinvj = dsp.deconvolve([1,0,0,0,0,0,0,0], vtj)[0] 
        dgj = dsp.lfilter(b, 1, gj)
        z = dsp.lfilter(vtinvj, 1, dgj)
        w = np.append(w, z)
    return w 	

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    dst_dir = args["<dst_dir>"]
    vocaltract_path = args["--vocal-tract"]
    glot_path = args["--glottal-flow"]
    file_name_suffix = args["--file-name-suffix"]
    output_html = args["--output-html"]


    # Load features
    if vocaltract_path is not None and glot_path is not None :
        g = np.load(glot_path)
        vt = np.load(vocaltract_path)	
        if g.shape[1] != 254:
            np.swapaxes(g, 0, 1)
        if vt.shape[1] != 5:
            np.swapaxes(vt, 0, 1)
        
        """
        if max_abs_value > 0:
            min_, max_ = 0, max_abs_value
            if symmetric_mels:
                min_ = -max_
            print("Normalize features to desired range [0, 1] from [{}, {}]".format(min_, max_))
            c = np.interp(c, (min_, max_), (0, 1))
        """    
    else:
        g = None
        vt = None


    os.makedirs(dst_dir, exist_ok=True)
    dst_wav_path = join(dst_dir, "gen.wav")
    
    # generate
    waveform = postprocess(g, vt)

    # save
    librosa.output.write_wav(dst_wav_path, waveform, sr=22050)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
