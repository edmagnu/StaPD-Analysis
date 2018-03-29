# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 12:45:27 2018

@author: labuser
"""

import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy.optimize
import os
import pandas as pd


def progress(source, i, total):
    """print an updating report of 'source: i/total'"""
    # start on fresh line
    if i == 0:
        print()
    # progress
    print("\r{0}: {1} / {2}".format(source, i+1, total), end="\r")
    # newline if we've reached the end.
    if i+1 == total:
        print()
    return


def dlist_gen(path):
    """Produce a list of all "?_delay.txt" filenames in the given folder.
    Returns list of filenames"""
    flist = os.listdir(path)
    for i, fname in enumerate(flist):
        flist[i] = os.path.join(path, fname)
    return flist


def read_metadata(fname):
    """Read file metadata and return a dictionary"""
    meta = {}  # initialize metadata dict
    with open(fname) as file:
        for line in file:
            if line[0] == "#":  # metadata written in comment lines
                line = line[1:]
                para = line.split("\t")  # tabs betweeen key & value
                if len(para) == 2:  # format for actual metadata, not comment
                    meta[para[0].strip()] = para[1].strip()
    return meta


def read_tidy(fname):
    """Read file, tidy metadata and data and return the DataFrame"""
    # read data, "i" is a data column, *not* the DataFrame index
    data = pd.read_csv(fname, sep="\t", comment="#", index_col=False)
    meta = read_metadata(fname)  # read metadata
    # Combine important metadata and data.
    # Interpret each meta individually, add to all observations
    meta['Filename'] = fname
    data['Filename'] = [meta['Filename']]*data.shape[0]
    data['Date'] = [pd.to_datetime(meta['Date'])]*data.shape[0]
    # strip unit off numerical value with ".split()"
    data['DL-Pro'] = [float(meta['DL-Pro'].split(" ")[0])]*data.shape[0]
    data['DL-100'] = [float(meta['DL-100'].split(" ")[0])]*data.shape[0]
    data['Ov'] = [float(meta['Ov'].split(" ")[0])]*data.shape[0]
    data['IR'] = [float(meta['IR'].split(" ")[0])]*data.shape[0]
    data['Bot'] = [float(meta['Bot'].split(" ")[0])]*data.shape[0]
    data['MWOn'] = [meta['MW'] == "On"]*data.shape[0]
    data['MWf'] = [float(meta['mwf'].split(" ")[0])]*data.shape[0]
    data['Attn'] = [float(meta['Attn'].split(" ")[0])]*data.shape[0]
    # rename data keys
    data.rename(index=int, inplace=True,
                columns={'norm bkgnd': 'nbackground',
                         'signal bkgnd': 'sbackground'})
    # reorder DataFrame columns.
    key_order = ['Filename', 'Date', 'DL-Pro', 'DL-100', 'Ov', 'IR', 'Bot',
                 'MWOn', 'MWf', 'Attn', 'i', 'step', 'norm', 'nbackground',
                 'signal', 'sbackground']
    data = data[key_order]
    return data


def transform_data(data):
    """For analysis, I want to compare delay in wavelengths against the
    normalied rydberg signal. These are derived from the read data.
    Returns DataFrame with "wavelengths" and "nsignal" added."""
    # wavelengths = 2 * "steps" * m * "MWf" * n / c
    m = 2.539e-7  # delay stage calibration, meters/step
    n = 1.0003  # index of refraction of air
    c = 299792458.0  # Speed of Light, meters/second
    # grouping by datetime
    mask = data['Date'] < datetime.datetime(2015, 11, 29)
    data.loc[mask, 'group'] = 1  # first run on 2018-11-28
    mask = data['Date'] > datetime.datetime(2015, 11, 29)
    data.loc[mask, 'group'] = 2  # second and third runs after 2018-11-28
    mask = data['Date'] > datetime.datetime(2015, 11, 30, 3)
    data.loc[mask, 'group'] = 3  # third run after 2015-11-30 3:00am
    # Turn plate voltages into x and z fields.
    data['fx'] = (data['Ov'] - data['IR'])*0.1*0.72
    data['fz'] = data['Bot']*0.1*0.72
    # angle from x axis
    data['fa'] = np.arctan2(data['fz'], data['fx'])
    data["wavelengths"] = 2*data["step"]*m*data["MWf"]*1.0e6*n/c
    # nsignal = (signal - sbackground)/(norm - nbackground)
    data["nsignal"] = ((data["signal"] - data["sbackground"])
                       / (data["norm"] - data["nbackground"]))
    return data


def excluded_files(data):
    flist = ['Circle Static\\2015-11-28\\7_delay_botm200mV_35dB.txt',
             'Circle Static\\2015-11-29\\17_delay_hp200mV_vm020mV.txt']
    for fname in flist:
        mask = data['Filename'] != fname
        data = data[mask].copy(deep=True)
    return data


def flist_gen():
    path = os.path.join("Circle Static", "2015-11-28")
    flist = dlist_gen(path)
    path = os.path.join("Circle Static", "2015-11-29")
    flist = flist + dlist_gen(path)
    return flist

def build_rawdata():
    """Read in a list of folders. Read in data from every "#_delay.txt" file in
    each folder, and interpret the metadata. Save the data frame to
    "rawdata.txt"
    Returns read data and metadata as a DataFrame."""
    # generate file list
    files = flist_gen()  # get every delay file from each folder.
    # read all file data & metadata into a tidy DataFrame
    data = pd.DataFrame()
    for i, file in enumerate(files):
        progress("build_rawdata()", i, len(files))
        data = data.append(read_tidy(file))
    # add "wavelengths" and "nsignal" (normalized signal)
    data = transform_data(data)
    data = data.reset_index(drop=True)  # unique index
    # write so I can use it elsewhere.
    fname = os.path.join("Circle Static", "rawdata.txt")
    data.to_csv(fname)
    print(fname)
    return data


def model_func(x, y0, a, phi):
    """Sinusoidal plus offset model for delay scan phase dependence.
    "x" is the delay in wavelengths
    "y" is the normalized Rydberg signal.
    Returns model dataframe and fit parameters.
    """
    return y0 + a*np.sin(2*np.pi*x + phi)


def model_p0(x, y):
    """Guesses reasonable starting parameters p0 to pass to model_func().
    x and y are pandas.Series
    Returns p0, array of [y0, a, phi]"""
    # y0 is the mean
    y0 = y.mean()
    # phi from averaged maximum and minimum
    yroll = y.rolling(window=9, center=True).mean()
    imax = yroll.idxmax()
    imin = yroll.idxmin()
    phi = ((x[imax] % 1) + ((x[imin]-0.5) % 1)) / 2
    phi = ((phi-0.25) % 1)*np.pi
    # a is max and min
    mx = yroll.max()
    mn = yroll.min()
    a = (mx-mn)/2
    return [y0, a, phi]


def build_fits():
    """Given a dataframe with 'wavelengths' and 'nsignal', fits to the
    model_func(), and adds fit parameters y0, a and phi to the DataFrame.
    Then, add a column to the data frame of the fitted values.
    Save the new data frame to "moddata.txt"
    Produce a fits DataFrame that has one observation per file, including all
    of the metadata and fit parameters, but no individual delay scan
    measurements.
    data is DataFrame with 'y0', 'a', 'phi' and 'model' keys added
    fits is DataFrame with just metadata and fit parameters, one observation
    per file.
    Returns data, fits
    """
    fname = os.path.join("Circle Static", "rawdata.txt")
    data = pd.read_csv(fname, index_col=0)
    data['Date'] = data['Date'].astype('datetime64[ns]')
    # add columns of NaN for each new key.
    dl = data.shape[0]
    data['y0'] = np.ones(dl)*np.NaN
    data['a'] = np.ones(dl)*np.NaN
    data['phi'] = np.ones(dl)*np.NaN
    data['model'] = np.ones(dl)*np.NaN
    # build DataFrame for metadata and fit params, not each point.
    fits = pd.DataFrame()
    # get a list of unique filenames
    filelist = data['Filename'].unique()
    imax = len(filelist)
    for i, filename in enumerate(filelist):
        progress("build_fits()", i, imax)
        mask = data['Filename'] == filename  # mask all but one file
        # data is often triangle in delay, sort it
        dsort = data[mask].sort_values(by=['wavelengths'])
        # fit to sinusoidal model_func()
        p0 = model_p0(dsort['wavelengths'], dsort['nsignal'])  # best guess
        # get fit parameters
        popt, pcov = scipy.optimize.curve_fit(
                model_func, dsort['wavelengths'].astype(float),
                dsort['nsignal'].astype(float), p0)
        # fill in fit parameters and model nsignal
        data.loc[mask, 'y0'] = popt[0]
        data.loc[mask, 'a'] = popt[1]
        data.loc[mask, 'phi'] = popt[2]
        # use original data, not the sorted data
        data.loc[mask, 'model'] = model_func(
                data[mask]['wavelengths'].astype(float), *popt)
        # build fits DataFrame
        keylist = ['Filename', 'DL-Pro', 'DL-100', 'fx', 'fz', 'fa', 'MWOn',
                   'MWf', 'Attn', 'group']
        fit = dsort.iloc[0][keylist]
        fit['y0'] = popt[0]
        fit['a'] = popt[1]
        fit['phi'] = popt[2]
        fits = fits.append(fit)
    # force fits key order
    fits = fits[keylist + ['y0', 'a', 'phi']]
    # export and return result
    fname = os.path.join("Circle Static", "moddata.txt")
    data.to_csv(fname)
    fname = os.path.join("Circle Static", "fits.txt")
    fits.to_csv(fname)
    print(fname)
    return data, fits


def massage_amp_phi(fsort, gate):
    """Given a series of fit phases, fix the phases and amplitudes so that all
    phases fall between [0, pi] + gate.
    Returns a fits DataFrame with modified 'phi'
    """
    # force all amps to be positive
    mask = (fsort['a'] < 0)
    fsort.loc[mask, 'a'] = -fsort[mask]['a']
    fsort.loc[mask, 'phi'] = fsort[mask]['phi'] + np.pi
    # force phases between 0 and 2pi
    mask = (fsort['phi'] > 2*np.pi)
    fsort.loc[mask, 'phi'] = fsort[mask]['phi'] - 2*np.pi
    mask = (fsort['phi'] < 0)
    fsort.loc[mask, 'phi'] = fsort[mask]['phi'] + 2*np.pi
    # phases above gate
    mask = (fsort['phi'] > gate) & (fsort['phi'] <= 2*np.pi)
    fsort.loc[mask, 'phi'] = fsort[mask]['phi'] - np.pi
    fsort.loc[mask, 'a'] = -fsort[mask]['a']
    # phases below gate - pi
    mask = (fsort['phi'] < (gate - np.pi)) & (fsort['phi'] > 0)
    fsort.loc[mask, 'phi'] = fsort[mask]['phi'] + np.pi
    fsort.loc[mask, 'a'] = -fsort[mask]['a']
    return fsort


def main():
    # data = build_rawdata()
    # data, fits = build_fits()
    fname = os.path.join("Circle Static", "fits.txt")
    fits = pd.read_csv(fname, index_col=0)
    fits['a'] = 2*fits['a']
    # start with group 1
    gate = {1: np.pi, 2: 2*np.pi, 3: 1.5*np.pi}
    colors = {1: 'C0', 2: 'C1', 3: 'C2'}
    fig, ax = plt.subplots()
    for group in [1, 2, 3]:
        mask = fits['group'] == group
        fsort = fits[mask].sort_values(by='fa')
        # get rid of bad files
        fsort = excluded_files(fsort)
        # corece amp and phase
        fsort = massage_amp_phi(fsort, gate[group])
        # plot
        fsort.plot(x='fa', y='a', linestyle='none', marker='o', ax=ax,
                   color=colors[group])
    ax.legend().remove()
    ax.set(xlabel=('Field Angle (rad)'), ylabel='Pk-Pk Amplitude',
           xlim=(-0.35, 0.25))
    return fsort


result = main()
print(result[['fa', 'phi', 'a']])
