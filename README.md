# GlotNET vocoder

An implementation GlotNET vocoder as variant of WaveNET. WaveNET is inherited from https://r9y9.github.io/wavenet_vocoder/.

The repo is quite immature, there may be defects, bugs and so on. I sincerely appreciate any feedback, suggestion or contribution. Thanks for your interest in the repo, enjoy it!

## Highlights

- Focus on local conditioning of GlotNet, which is essential for vocoder.
- Mixture of logistic distributions loss / sampling
- Various audio samples 
- Fast inference by caching intermediate states in convolutions. Similar to [arXiv:1611.09482](https://arxiv.org/abs/1611.09482)

![Alt text](gnvswn.png?raw=true "GlotNet and WaveNet")
Figure 1: WaveNet vocoder (left) uses acoustic features (AC) and past signal samples to generate the next speech sample. In contrast, GlotNet (right) operates on the more simplistic glottal excitation signal, which is filtered by a vocal tract (VT) filter already parametrized in the acoustic features.

<!--
## Pre-trained models

**Note**: This is not itself a text-to-speech (TTS) model. With a pre-trained model provided here, you can synthesize waveform given a *mel spectrogram*, not raw text. You will need mel-spectrogram prediction model (such as Tacotron2) to use the pre-trained models for TTS.

| Model URL                                                                                                                        | Data       | Hyper params URL                                                                                     | Git commit                                                                                         | Steps         |
|----------------------------------------------------------------------------------------------------------------------------------|------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|---------------|
| [link](https://www.dropbox.com/s/zdbfprugbagfp2w/20180510_mixture_lj_checkpoint_step000320000_ema.pth?dl=0)                      | LJSpeech   | [link](https://www.dropbox.com/s/0vsd7973w20eskz/20180510_mixture_lj_checkpoint_step000320000_ema.json?dl=0)                | [2092a64](https://github.com/r9y9/wavenet_vocoder/commit/2092a647e60ce002389818de1fa66d0a2c5763d8) | 1000k~  steps |
| [link](https://www.dropbox.com/s/d0qk4ow9uuh2lww/20180212_mixture_multispeaker_cmu_arctic_checkpoint_step000740000_ema.pth?dl=0) | CMU ARCTIC | [link](https://www.dropbox.com/s/i35yigj5hvmeol8/20180212_multispeaker_cmu_arctic_mixture.json?dl=0) | [b1a1076](https://github.com/r9y9/wavenet_vocoder/tree/b1a1076e8b5d9b3e275c28f2f7f4d7cd0e75dae4)   | 740k steps    |

To use pre-trained models, first checkout the specific git commit noted above. i.e.,

```
git checkout ${commit_hash}
```

And then follows "Synthesize from a checkpoint" section in the README. Note that old version of synthesis.py may not accept `--preset=<json>` parameter and you might have to change `hparams.py` according to the preset (json) file.

You could try for example:

```
# Assuming you have downloaded LJSpeech-1.1 at ~/data/LJSpeech-1.1
# pretrained model (20180510_mixture_lj_checkpoint_step000320000_ema.pth)
# hparams (20180510_mixture_lj_checkpoint_step000320000_ema.json)
git checkout 2092a64
python preprocess.py ljspeech ~/data/LJSpeech-1.1 ./data/ljspeech \
  --preset=20180510_mixture_lj_checkpoint_step000320000_ema.json
python synthesis.py --preset=20180510_mixture_lj_checkpoint_step000320000_ema.json \
  --conditional=./data/ljspeech/ljspeech-mel-00001.npy \
  20180510_mixture_lj_checkpoint_step000320000_ema.pth \
  generated
```

You can find a generated wav file in `generated` directory. Wonder how it works? then take a look at code:)
-->

## Requirements

- Python 3
- SciPy
- CUDA >= 8.0
- PyTorch >= v0.4.0
- TensorFlow >= v1.3

<!--
## Installation

The repository contains a core library (PyTorch implementation of the WaveNet) and utility scripts. All the library and its dependencies can be installed by:

```
git clone https://github.com/r9y9/wavenet_vocoder && cd wavenet_vocoder
pip install -e ".[train]"
```

If you only need the library part, then you can install it by the following command:

```
pip install wavenet_vocoder
```
-->

## Getting started
### A few notes on usage
Input data to network must be 'raw' other options 'mu-law-quantization' are for pure wavenet.
I prefer not to use upsampling of conditional features. If you are eager to do so go over scripts quickly for modifications and arrange interpolation layers' factors so that factors yields 254.
For now networks preprocess LJSpeech only, if you want to feed in other dataset corresponding scripts must be provided.

### Preset parameters

There are many hyper parameters to be turned depends on data. For typical datasets, parameters known to work good (**preset**) are provided in the repository. See `presets` directory for details. Notice that

1. `preprocess.py`
2. `train.py`
3. `synthesis.py`

accepts `--preset=<json>` *optional* parameter, which specifies where to load preset parameters. If you are going to use preset parameters, then you must use same `--preset=<json>` throughout preprocessing, training and evaluation. e.g.,

```
python preprocess.py --preset=presets/cmu_arctic_8bit.json cmu_arctic ~/data/cmu_arctic
python train.py --preset=presets/cmu_arctic_8bit.json --data-root=./data/cmu_arctic
```

instead of

```
python preprocess.py cmu_arctic ~/data/cmu_arctic
# warning! this may use different hyper parameters used at preprocessing stage
python train.py --preset=presets/cmu_arctic_8bit.json --data-root=./data/cmu_arctic
```

### 0. Download dataset

- CMU ARCTIC (en): http://festvox.org/cmu_arctic/
- LJSpeech (en): https://keithito.com/LJ-Speech-Dataset/

### 1. Preprocessing

Usage:

```
python preprocess.py ${dataset_name} ${dataset_path} ${out_dir} --preset=<json>
```

Supported `${dataset_name}`s for now are

- `ljspeech` (single speaker)

- `cmu_arctic` (multi speaker)

Assuming you use preset parameters known to work good for CMU ARCTIC dataset and have data in `~/data/cmu_arctic`, then you can preprocess data by:

```
python preprocess.py cmu_arctic ~/data/cmu_arctic ./data/cmu_arctic --preset=presets/cmu_arctic_8bit.json
```

When this is done, you will see time-aligned extracted features (audio, glottal exciation, vocal tract filter and mel-spectrogram) in `./data/cmu_arctic`.

### 2. Training


>Note: for multi gpu training, you have better ensure that batch_size % num_gpu == 0

Usage:

```
python train.py --data-root=${data-root} --preset=<json> --hparams="parameters you want to override"
```

Important options:

- `--speaker-id=<n>`: (Multi-speaker dataset only) it specifies which speaker of data we use for training. If this is not specified, all training data are used. This should only be specified when you are dealing with a multi-speaker dataset. For example, if you are trying to build a speaker-dependent WaveNet vocoder for speaker `awb` of CMU ARCTIC, then you have to specify `--speaker-id=0`. Speaker ID is automatically assigned as follows:

```py
In [1]: from nnmnkwii.datasets import cmu_arctic

In [2]: [(i, s) for (i,s) in enumerate(cmu_arctic.available_speakers)]
Out[2]:

[(0, 'awb'),
 (1, 'bdl'),
 (2, 'clb'),
 (3, 'jmk'),
 (4, 'ksp'),
 (5, 'rms'),
 (6, 'slt')]
```


#### 2. Training WaveNet conditioned on mel-spectrogram

```
python train.py --data-root=./data/cmu_arctic/ --speaker-id=0 \
    --hparams="cin_channels=80,gin_channels=-1"
```


### 3. Monitor with Tensorboard

Logs are dumped in `./log` directory by default. You can monitor logs by tensorboard:

```
tensorboard --logdir=log
```

### 4. Synthesize glottal excitation from a checkpoint

Usage:

```
python synthesis.py ${checkpoint_path} ${output_dir} --preset=<json> --hparams="parameters you want to override"
```

Important options:

- `--length=<n>`: (Un-conditional WaveNet only) Number of time steps to generate.
- `--conditional=<path>`: (Required for onditional WaveNet) Path of local conditional features (.npy). If this is specified, number of time steps to generate is determined by the size of conditional feature.

e.g.,

```
python synthesis.py --hparams="parameters you want to override" \
    checkpoints_awb/checkpoint_step000100000.pth \
    generated/test_awb \
    --conditional=./data/cmu_arctic/cmu_arctic-mel-00001.npy
```

### 5. Post Process
Combine vocal tract and glottal excitation generated by GlotNET to get the speech item.

Usage:

```
python postprocess.py ${vocal-tract_path} ${glot_path} ${output_dir} --preset=<json> --hparams="parameters you want to override"
```


## References

- [Lauri Juvela, Vassilis Tsiaras, Bajibabu Bollepalli, et al, "Speaker-independent raw waveform model for glottal excitation", arXiv:1804.09593, Apr. 2018](https://arxiv.org/pdf/1804.09593.pdf)
- [Aaron van den Oord, Sander Dieleman, Heiga Zen, et al, "WaveNet: A Generative Model for Raw Audio", 	arXiv:1609.03499, Sep 2016.](https://arxiv.org/abs/1609.03499)
- [Aaron van den Oord, Yazhe Li, Igor Babuschkin, et al, "Parallel WaveNet: Fast High-Fidelity Speech Synthesis", 	arXiv:1711.10433, Nov 2017.](https://arxiv.org/abs/1711.10433)
- [Tamamori, Akira, et al. "Speaker-dependent WaveNet vocoder." Proceedings of Interspeech. 2017.](http://www.isca-speech.org/archive/Interspeech_2017/pdfs/0314.PDF)
- [Jonathan Shen, Ruoming Pang, Ron J. Weiss, et al, "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions", arXiv:1712.05884, Dec 2017.](https://arxiv.org/abs/1712.05884)
- [Wei Ping, Kainan Peng, Andrew Gibiansky, et al, "Deep Voice 3: 2000-Speaker Neural Text-to-Speech", arXiv:1710.07654, Oct. 2017.](https://arxiv.org/abs/1710.07654)
- [Tom Le Paine, Pooya Khorrami, Shiyu Chang, et al, "Fast Wavenet Generation Algorithm", arXiv:1611.09482, Nov. 2016](https://arxiv.org/abs/1611.09482)


