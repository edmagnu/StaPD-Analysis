# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:47:38 2018

@author: labuser
"""

import numpy as np
import pandas as pd
import os


# load, just one available scan, good enough for prelim results
fname = "1_mwref.txt"
fname = os.path.join("..", "..", "2016-12-13", fname)
df = pd.read_csv(fname, delim_whitespace=True, header=None,
                 names=['f', 'V1', 'V2'], comment='#')
df.sort_values(by='f', inplace=True)
df['f'] = df['f']/1e9  # GHz

# plot to get important values
ax = df.plot(x='f', y='V1', label='Reflected')
df.plot(x='f', y='V2', ax=ax, label='Background')
ax.set(xlabel='Frequency (GHz)', ylabel='Power (arb. u.)')
f0 = 15.93245  # resonant frequency f0
ax.axvline(f0, color='k')
vres = 0.3715  # resonance peak
ax.axhline(vres, color='k')
vbfh = 0.5305  # high f baseline
ax.axhline(vbfh, color='k')
vbfl = 0.5360  # low f baseline
ax.axhline(vbfl, color='k')
vb = (vbfh + vbfl)/2  # ave baseline
hm = (vb + vres)/2  # half-max
ax.axhline(((0.5305+0.5360)/2 + 0.3715)/2, color='lightgrey')
flhm = 15.93155  # low frequency at half-max
ax.axvline(flhm, color='lightgrey')
fhhm = 15.93330  # high frequency at half-max
ax.axvline(fhhm, color='lightgrey')

# calculations
fwhm = fhhm-flhm  # Full-width at half-max
Q = f0/fwhm  # Quality factor
tau = 2*Q/(2*np.pi*f0*1e9)  # field ringdown, convert f0 from GHz -> Hz

# report
print("f0 =\t  {} GHz\tCentral Frequency".format(round(f0,5)))
print("fwhm =\t   {} GHz\tFull-Width at Half-Max".format(round(fwhm, 5)))
print("Q =\t{}\t\tQuality Factor".format(round(Q, 0)))
print("tau =\t {} ns\tField Ringdown".format(round(tau*1e9, 2)))
