# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:24:19 2017

@author: edmag
"""

# Static Phase Dependence (StaPD) Experiment Data Analysis
# Eric Magnuson, University of Virginia, VA

import pandas as pd
import os


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
    data = data.reset_index()  # unique index 
    return data


# main program starts here
# generate file list from data folder list
target = (".\\" + "data_folders.txt")
files = folderlist_gen(target)  # get every delay file from each folder.
# read all file data & metadata into a tidy DataFrame
data = pd.DataFrame()
for file in files:
    print(file)
    data = data.append(read_tidy(file))
print(data.keys())
