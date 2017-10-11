# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:24:19 2017

@author: edmag
"""

# Static Phase Dependence (StaPD) Experiment Data Analysis
# Eric Magnuson, University of Virginia, VA

import pandas as pd

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
    data["DLPro"] = [float(meta["DLPro"].split(" ")[0])]*data.shape[0]
    data["DL100"] = [float(meta["DL100"].split(" ")[0])]*data.shape[0]
    data["Static"] = [float(meta["Static"].split(" ")[0])]*data.shape[0]
    data["MWOn"] = [meta["MWOn"] == "On"]*data.shape[0]
    data["MWf"] = [float(meta["MWf"].split(" ")[0])]*data.shape[0]
    data["Attn"] = [float(meta["Attn"].split(" ")[0])]*data.shape[0]
    # reorder DataFrame columns.
    key_order = ["Filename", "Date", "DLPro", "DL100", "Static", "MWOn", "MWf",
                 "Attn", "i", "step", "norm", "nbackground", "signal",
                 "sbackground"]
    data = data[key_order]
    return data


# main program starts here
# target
path = "C:\\Users\\edmag\\Documents\\Work\\Data\\StaPD-Analysis\\Modified Data"
folder = "2016-09-22"
file = "9_delay.txt"
fname = "\\".join([path, folder, file])
print(fname)
# read in data
data = pd.read_csv(fname, sep="\t", comment="#", index_col=False)
print(data)
# TODO(edm5gb): Figure out how to produce a list of files to load
# TODO(edm5gb): Read all target files into a tidy data set
