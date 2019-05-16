
from pysptk.synthesis import MLSADF, Synthesizer
import seaborn
import numpy as np
import pysptk
from scipy.io import wavfile
import librosa
import librosa.display
import matplotlib
import seaborn
import time
from matplotlib import pyplot as plt
seaborn.set(style="whitegrid")

frame_length = 256
hop_length = 20

alpha = 0.2
#sr, x = wavfile.read(pysptk.util.example_audio_file())
sr, x = wavfile.read("x.wav")
assert sr == 8000
x = x.astype(np.float64)
print(x.shape)
#wavfile.write("x.wav", sr, x)

f = open("times", "w")
f.write("order,time\n")
for order in (0,4,9,14,24):
    start = time.time()
    # Note that almost all of pysptk functions assume input array is C-contiguous and np.float64 element type
    frames = librosa.util.frame(
        x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T

    # Windowing
    frames *= pysptk.blackman(frame_length)

    assert frames.shape[1] == frame_length

    pitch = pysptk.swipe(x.astype(np.float64), fs=sr,
                        hopsize=hop_length, min=60, max=240, otype="pitch")
    source_excitation = pysptk.excite(pitch, hop_length)


    # Order of mel-cepstrum


    mc = pysptk.mcep(frames, order, alpha)
    logH = pysptk.mgc2sp(mc, alpha, 0.0, frame_length).real
    print(mc.shape)
    #plt.plot(mc)
    #plotname="x_syn_coefs_" + str(order) + ".png"
    #plt.savefig(plotname)

    # Convert mel-cesptrum to MLSADF coefficients
    b = pysptk.mc2b(mc, alpha)

    synthesizer = Synthesizer(MLSADF(order=order, alpha=alpha), hop_length)

    x_synthesized = synthesizer.synthesis(source_excitation, b)

    filenam = "synthesized_sounds/"+"x_syn" + str(order+1) + ".wav"
    #wavfile.write("x.wav", sr, x)
    wavfile.write(filenam,sr,x_synthesized)
    time_total = time.time() - start
    writestring = str(order)+","+str(time_total)+"\n"
    f.write(writestring)
