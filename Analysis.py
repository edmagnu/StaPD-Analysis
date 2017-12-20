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
    # data["Date"] = [pd.to_datetime(meta["Date"])]*data.shape[0]
    # strip unit off numerical value with ".split()"
    data["DL-Pro"] = [float(meta["DL-Pro"].split(" ")[0])]*data.shape[0]
    data["DL-100"] = [float(meta["DL-100"].split(" ")[0])]*data.shape[0]
    data["Static"] = [float(meta["Static"].split(" ")[0])]*data.shape[0]
    data["MWOn"] = [meta["MWOn"] == "On"]*data.shape[0]
    data["MWf"] = [float(meta["MWf"].split(" ")[0])]*data.shape[0]
    data["Attn"] = [float(meta["Attn"].split(" ")[0])]*data.shape[0]
    # reorder DataFrame columns.
    key_order = ["Filename", "DL-Pro", "DL-100", "Static", "MWOn",
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


def plot_dscan(data, filename, save=False, close=False):
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
    p0 = np.array([float(data["y0"][[0]]), float(data["a"][[0]]),
                   float(data["phi"][[0]])])
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
    stamp = "y0 = {:2.3f}\na = {:2.3f}\nphi = {:2.2f}".format(p0[0], p0[1],
                                                              p0[2]/(2*np.pi))
    stamp = stamp + "\nf_cent = {:2.0f} GHz".format(fcent)
    stamp = stamp + "\nF_st = {:2.2f} mV/cm".format(static)
    plt.text(0.05, 0.05,
             stamp,
             transform=axes.transAxes, bbox=dict(facecolor="white"))
    axes.set_xlabel("Delay (MW periods)")
    axes.set_ylabel("Rydberg Signal")
    axes.set_title(filename)
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
    Returns data DataFrame with "y0", "a", "phi" and "model" keys added.
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
        data.loc[mask, "model"] = model_func(data[mask]["wavelengths"], *p0)
        # build fits DataFrame
        keylist = ["Filename", "DL-Pro", "DL-100", "Static", "MWOn", "MWf",
                   "Attn"]
        fit = data.iloc[0][keylist]
        fit["y0"] = p0[0]
        fit["a"] = p0[1]
        fit["phi"] = p0[2]
        fits = fits.append(fit)
    # force fits key order
    fits = fits[keylist + ["y0", "a", "phi"]]
    # export and return result]
    data.to_csv("moddata.txt", sep="\t")
    fits.to_csv("fits.txt", sep="\t")
    return data, fits


# main program starts here
def main():
    """Read in raw data Add fit parameters and model values and save."""
    # read in all data with fit info
    data = pd.read_csv("rawdata.txt", sep="\t", index_col=0)
    data, fits = build_fits(data)
    return data, fits


data, fits = main()
