# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:24:19 2017

@author: edmag
"""

# Static Phase Dependence (StaPD) Experiment Data Analysis
# Eric Magnuson, University of Virginia, VA

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import os
import pandas as pd


def dlist_gen(path):
    """Produce a list of all "?_delay.txt" filenames in the given folder.
    Returns list of filenames"""
    flist = os.listdir(path)
    dlist = list()
    for i in range(len(flist)):
        if "_delay.txt" in flist[i]:
            dlist.append(path + flist[i])
    return dlist


def folderlist_gen(fname):
    """Produce a list of all the "?_delay.txt" filenames in every folder listed
    in "fname".
    Returns a list of filenames."""
    folders = []  # list of folders
    with open(fname) as file:
        for line in file:
            line = line.strip()  # get rid of "\n"
            line = "\\".join([".", "Modified Data", line])  # get a full path
            folders.append(line)
    files = []  # list of files in all folders
    for folder in folders:
        files = files + dlist_gen(folder + "\\")  # dlist_gen by folder
    return files


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
    data["Filename"] = [meta["Filename"]]*data.shape[0]
    data["Date"] = [pd.to_datetime(meta["Date"])]*data.shape[0]
    # strip unit off numerical value with ".split()"
    data["DL-Pro"] = [float(meta["DL-Pro"].split(" ")[0])]*data.shape[0]
    data["DL-100"] = [float(meta["DL-100"].split(" ")[0])]*data.shape[0]
    data["Static"] = [float(meta["Static"].split(" ")[0])]*data.shape[0]
    data["MWOn"] = [meta["MWOn"] == "On"]*data.shape[0]
    data["MWf"] = [float(meta["MWf"].split(" ")[0])]*data.shape[0]
    data["Attn"] = [float(meta["Attn"].split(" ")[0])]*data.shape[0]
    # reorder DataFrame columns.
    key_order = ["Filename", "Date", "DL-Pro", "DL-100", "Static", "MWOn",
                 "MWf", "Attn", "i", "step", "norm", "nbackground", "signal",
                 "sbackground"]
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
    data["wavelengths"] = 2*data["step"]*m*data["MWf"]*1.0e6*n/c
    # nsignal = (signal - sbackground)/(norm - nbackground)
    data["nsignal"] = ((data["signal"] - data["sbackground"])
                       / (data["norm"] - data["nbackground"]))
    return data


def build_rawdata():
    """Read in a list of folders. Read in data from every "#_delay.txt" file in
    each folder, and interpret the metadata. Save the data frame to
    "rawdata.txt"
    Returns read data and metadata as a DataFrame."""
    # generate file list from data folder list
    target = (".\\" + "data_folders.txt")
    files = folderlist_gen(target)  # get every delay file from each folder.
    # read all file data & metadata into a tidy DataFrame
    data = pd.DataFrame()
    for file in files:
        print(file)
        data = data.append(read_tidy(file))
    # add "wavelengths" and "nsignal" (normalized signal)
    data = transform_data(data)
    data = data.reset_index(drop=True)  # unique index
    print(data.keys())
    # write so I can use it elsewhere.
    data.to_csv("rawdata.txt", sep="\t")
    print("Data written to 'rawdata.txt'")
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


def plot_dscan(data, filename, save=False, close=True):
    """Given a DataFrame with multiple delay scans, select one by the
    filename. Plot the data, running average, and a model fit. Assumes data
    comes with "wavelenth", "nsignal", "y0", "a", "phi", and "model".
    Returns None"""
    # pick out one file
    data = data[data["Filename"] == filename]
    data = data.sort_values(by=["wavelengths"])
    # produce a rolling average
    rave = data[["wavelengths", "nsignal"]].rolling(window=9, center=True)
    rave = rave.mean()
    rave = rave[~np.isnan(rave["nsignal"])]
    # extract model p0 [y0, a, phi]
    popt = np.array([data["y0"].values[0], data["a"].values[0],
                     data["phi"].values[0]])
    # plot the data, model, and rolling average
    axes = data.plot(x="wavelengths", y="nsignal", kind="scatter",
                     label="data")
    data.plot(x="wavelengths", y="model", kind="line", color="black",
              label="fit", ax=axes)
    rave.plot(x="wavelengths", y="nsignal", kind="line", color="red",
              label="rolling ave.", ax=axes)
    # stamp with useful information
    fcent = ((data["DL-Pro"].unique()[0] + data["DL-100"].unique()[0])/2
             - 365869.6)
    static = data["Static"].unique()[0]*0.1*0.72
    stamp = "y0 = {:2.3f}\na = {:2.3f}\nphi = {:2.2f}".format(
            popt[0], popt[1], popt[2]/(2*np.pi))
    stamp = stamp + "\nf_cent = {:2.0f} GHz".format(fcent)
    stamp = stamp + "\nF_st = {:2.2f} mV/cm".format(static)
    plt.text(0.05, 0.05,
             stamp,
             transform=axes.transAxes, bbox=dict(facecolor="white"))
    axes.set_xlabel("Delay (MW periods)")
    axes.set_ylabel("Rydberg Signal")
    axes.set_title(filename)
    axes.legend(loc=4)
    # plt.grid(True)
    plt.tight_layout()
    if save is True:
        fname = "Modified Data\\" + filename.split(".")[0] + ".pdf"
        print(fname)
        plt.savefig(fname)
    if close is True:
        plt.close()


def build_fits(data):
    """Given a dataframe with "wavelengths" and "nsignal", fits to the
    model_func(), and adds fit parameters y0, a and phi to the DataFrame.
    Then, add a column to the data frame of the fitted values.
    Save the new data frame to "moddata.txt"
    Produce a fits DataFrame that has one observation per file, including all
    of the metadata and fit parameters, but no individual delay scan
    measurements.
    data is DataFrame with "y0", "a", "phi" and "model" keys added
    fits is DataFrame with just metadata and fit parameters, one observation
    per file.
    Returns data, fits
    """
    # add columns of NaN for each new key.
    dl = data.shape[0]
    data["y0"] = np.ones(dl)*np.NaN
    data["a"] = np.ones(dl)*np.NaN
    data["phi"] = np.ones(dl)*np.NaN
    data["model"] = np.ones(dl)*np.NaN
    # build DataFrame for metadata and fit params, not each point.
    fits = pd.DataFrame()
    # get a list of unique filenames
    filelist = data["Filename"].unique()
    imax = len(filelist)
    for i, filename in enumerate(filelist):
        print(i+1, " / ", imax)  # progress
        mask = data["Filename"] == filename  # mask all but one file
        # data is often triangle in delay, sort it
        dsort = data[mask].sort_values(by=["wavelengths"])
        # fit to sinusoidal model_func()
        p0 = model_p0(dsort["wavelengths"], dsort["nsignal"])  # bestt guess
        # get fit parameters
        popt, pcov = scipy.optimize.curve_fit(
                model_func, dsort["wavelengths"], dsort["nsignal"], p0)
        # fill in fit parameters and model nsignal
        data.loc[mask, "y0"] = popt[0]
        data.loc[mask, "a"] = popt[1]
        data.loc[mask, "phi"] = popt[2]
        # use original data, not the sorted data
        data.loc[mask, "model"] = model_func(data[mask]["wavelengths"], *popt)
        # build fits DataFrame
        keylist = ["Filename", "DL-Pro", "DL-100", "Static", "MWOn", "MWf",
                   "Attn"]
        fit = dsort.iloc[0][keylist]
        fit["y0"] = popt[0]
        fit["a"] = popt[1]
        fit["phi"] = popt[2]
        fits = fits.append(fit)
    # force fits key order
    fits = fits[keylist + ["y0", "a", "phi"]]
    # export and return result]
    data.to_csv("moddata.txt", sep="\t")
    fits.to_csv("fits.txt", sep="\t")
    return data, fits


def massage_amp_phi(fsort, gate):
    """Given a series of fit phases, fix the phases and amplitudes so that all
    phases fall between [0, pi] + gate.
    Returns a fits DataFrame with modified "phi"
    """
    # force all amps to be positive
    mask = (fsort["a"] < 0)
    fsort.loc[mask, "a"] = -fsort[mask]["a"]
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] + np.pi
    # force phases between 0 and 2pi
    mask = (fsort["phi"] > 2*np.pi)
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] - 2*np.pi
    mask = (fsort["phi"] < 0)
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] + 2*np.pi
    # collapse phases to [0, pi] + gate
    gate = np.pi
    mask = (fsort["phi"] > gate) & (fsort["phi"] <= (gate + np.pi))
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] - np.pi
    fsort.loc[mask, "a"] = -fsort[mask]["a"]
    mask = (fsort["phi"] > (gate + np.pi))
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] - 2*np.pi
    return fsort


def dil_p2():
    """
    Selects "fits.txt" data with lasers at DIL +2GHz and Attn = 38.0
    (happens to all be 2016-09-22) and plots Static vs. fit parameters "a, phi"
    Uses massage_amp_phi() before plotting to fix "a, phi".
    Returns DataFrame "fsort" that is just the plotted observations.
    """
    # read in all data wth fit info
    # data = pd.read_csv("moddata.txt", sep="\t", index_col=0)
    fits = pd.read_csv("fits.txt", sep="\t", index_col=0)
    # mask out DIL + 2 GHz and Attn = 38.0
    mask = (fits["DL-Pro"] == 365872.6) & (fits["Attn"] == 38.0)
    fsort = fits[mask].sort_values(by=["Static"])
    gate = np.pi
    fsort = massage_amp_phi(fsort, gate)
    print(fsort)
    # plot
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False,
                             figsize=(11, 8.5))
    fsort.plot(x="Static", y="a", kind="scatter", ax=axes[0])
    fsort.plot(x="Static", y="phi", kind="scatter", ax=axes[1])
    axes[1].set_yticks([-np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    axes[1].set_yticklabels([r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$",
                             r"$3\pi/2$", r"$2\pi$"])
    for i in np.arange(gate - np.pi, gate + np.pi, np.pi):
        axes[1].axhline(i, color="black")
    # make it pretty
    axes[1].set_ylabel("Phase (rad)")
    axes[1].set_xlabel("Pulsed Voltage (V)")
    axes[0].set_ylabel("Amplitude (P. of Excited)")
    axes[1].grid(True)
    axes[0].grid(True)
    plt.suptitle("DIL + 2 GHz")
    return fsort


# main program starts here
def main():
    """Read data and model fits."""
    # read in all data with fit info
    data = pd.read_csv("moddata.txt", sep="\t", index_col=0)
    fits = pd.read_csv("fits.txt", sep="\t", index_col=0)
    # mask out just DIL + 2 GHz data.
    fmask = fits["DL-Pro"] == 365872.6
    fmask = fmask & (fits["Attn"] == 38.0)
    fsort = fits[fmask].sort_values(by=["Static"])
    # force all amps to be positive
    mask = (fsort["a"] < 0)
    fsort.loc[mask, "a"] = -fsort[mask]["a"]
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] + np.pi
    # force phases between 0 and 2pi
    mask = (fsort["phi"] > 2*np.pi)
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] - 2*np.pi
    mask = (fsort["phi"] < 0)
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] + 2*np.pi
    # collapse phases to [0, pi] + gate
    gate = np.pi
    mask = (fsort["phi"] > gate) & (fsort["phi"] <= (gate + np.pi))
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] - np.pi
    fsort.loc[mask, "a"] = -fsort[mask]["a"]
    mask = (fsort["phi"] > (gate + np.pi))
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] - 2*np.pi
    # plot
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False,
                             figsize=(11, 8.5))
    fsort.plot(x="Static", y="a", kind="scatter", ax=axes[0])
    fsort.plot(x="Static", y="phi", kind="scatter", ax=axes[1])
    axes[1].set_yticks([-np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    axes[1].set_yticklabels([r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$",
                             r"$3\pi/2$", r"$2\pi$"])
    for i in np.arange(gate - np.pi, gate + np.pi, np.pi):
        axes[1].axhline(i, color="black")
    # make it pretty
    axes[1].set_ylabel("Phase (rad)")
    axes[1].set_xlabel("Pulsed Voltage (V)")
    axes[0].set_ylabel("Amplitude (P. of Excited)")
    axes[1].grid(True)
    axes[0].grid(True)
    return data, fits


fsort = dil_p2()
