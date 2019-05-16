from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
from math import log10



sr, x = wavfile.read("x.wav")
assert sr == 16000
x = x.astype(np.float64)
print(x.shape)

for i in (0, 4, 9, 14, 24, 49):
    filename_syn = "x_syn" + str(i+1) + ".wav"
    sr, x5 = wavfile.read("synthesized_sounds/"+filename_syn)
    assert sr == 16000
    x5 = x5.astype(np.float64)
    print(x5.shape)
    diff = []

    for i in range(len(x5)):
        dif = abs(x[i]-x5[i])
        if dif > 0:
            diff.append(10*log10(dif))
        else:
            diff.append(0)
    plt.figure()
    plt.plot(diff)
    plotname = "differences/"+filename_syn[:-4] + ".png"
    plt.savefig(plotname)

