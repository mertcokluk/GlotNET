import numpy
import scipy.signal as dsp
from scipy.linalg import solve_toeplitz as lpcsolve

wav=numpy.array([1,2,2,1,2,1,2,12,1])
fs=16000
p_vt=5
p_gl=5
d=0.99
hpfilt=2
preflt = p_vt+1

# High-pass filter speech in order to remove possible low frequency
# fluctuations (Linear-phase FIR, Fc = 70 Hz)
Fstop = 40                        # Stopband Frequency
Fpass = 70                        # Passband Frequency
Nfir = numpy.round(300/16000*fs)  # FIR numerator order

if (Nfir % 2==0):
    Nfir = Nfir + 1

'''
 it is very very expensive to calculate the firls filter! However, as 
 long as the fs does not change, the firls filter does not change.
 Therefore, the computed filter is returned and can be passed to this
 function later on to avoid the calculated of the (same) filter.
'''
B = dsp.firls(Nfir, [0, Fstop/(fs/2), Fpass/(fs/2), fs/2], [0, 0, 1, 1], [1, 1])

hpfilter_out = B

for i in range(hpfilt):
    wav = dsp.lfilter(B, 1, numpy.concatenate((wav, numpy.zeros(numpy.int(len(B)/2)-1))))
    wav = wav[numpy.int(len(B)/2):]

'''
% Estimate the combined effect of the glottal flow and the lip radiation
% (Hg1) and cancel it out through inverse filtering. Note that before
% filtering, a mean-normalized pre-frame ramp is appended in order to
% diminish ripple in the beginning of the frame. The ramp is removed after
% filtering.
'''

if len(wav) > p_vt:
    win = numpy.hanning(len(wav))
    a=numpy.array([1])
    b=numpy.array([1, -d])
    signal = numpy.concatenate((numpy.linspace(-wav[1],wav[1],preflt),  wav))
    
    windowed = wav*win
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
    windowedg1 = wav*g1
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
else: 
    g=numpy.array;
    dg=numpy.array;
    a=numpy.array;
    ag=numpy.array;
