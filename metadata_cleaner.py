# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:54:06 2017

@author: edmag
"""

# Interpret sloppy delay scan metadata and rewrite as formatted metadata
# Eric Magnuson, University of Virginia, VA

import pandas as pd


def d_print(d):
    """Print out dictionary key : value. Returns None"""
    for k,v in d.items():
        print(k, ":", v)


# File location
path = "C:\\Users\\edmag\\Documents\\Work\\Data\\StaPD-Analysis" \
       + "\\Modified Data\\MetaTest"
file = "\\8_delay.txt"
fname = path + file
metadata = {}  # initialize dictionary
# Start with date, first line
with open(fname) as file:
    for i, line in enumerate(file):
        # Only comment lines
        if line[0] != "#":
            break
        # Dates
        elif line[2:6] == "2016":
            print("It's a date!")
            date = line[2:12]  # strip to just yyyy-mm-dd
            metadate = "# Date\t" + date + "\n"
            metadata["Date"] = metadate
        # Title
        elif ("DL-Pro" in line) & ("DL-100" in line) & ("delay" in line):
            print("It's a title!")
            metatitle = "# Title\t" + line[2:]
            metadata["Title"] = metatitle
        # DLPro & DL100
        elif ("@" in line) & ("GHz" in line):
            print("It's laser frequencies!")
            line = line.split(" ")
            # DLPro
            ipro = line.index("DL-Pro")
            metaDLPro = "# DL-Pro\t" + line[ipro + 2] + " GHz\n"
            metadata["DLPro"] = metaDLPro
            # DL100
            i100 = line.index("DL-100")
            metaDL100 = "# DL-100\t" + line[i100 + 2] + " GHz\n"
            metadata["DL100"] = metaDL100
        # Static
        elif ("Static" in line) & ("mV" in line):
            print("It's Static!")
            line = line.split(" ")
            metaStatic = "# Static\t" + line[2] + " " + line[3] + "\n"
            metadata["Static"] = metaStatic
        # MWOn & MWf & Attn & FixedAttn
        elif ("MW " in line) & ("Attn" in line) & ("Hz" in line):
            print("It's MW settings!")
            line = line.split(" ")
            # MWOn
            iMW = line.index("MW")
            metaMWOn = "# MWOn\t" + line[iMW + 1] + "\n"
            metadata["MWOn"] = metaMWOn
            # MWf
            iF = line.index("f")
            metaMWf = "# MWf\t" + line[iF + 2] + " " + line[iF + 3][0:3] + "\n"
            metadata["MWf"] = metaMWf
            # Attn
            iAttn = line.index("Attn")
            metaAttn = "# Attn\t" + line[iAttn + 2] + " dB\n"
            metadata["Attn"] = metaAttn
            # FixedAttn
            fAttn = line[-1]
            fAttn = fAttn.strip()
            fAttn = fAttn.strip("(")
            fAttn = fAttn.strip(")")
            fAttn = fAttn.split("+")
            metaFixedAttn = "# FixedAttn\t" + fAttn[1] + " dB\n"
            metadata["FixedAttn"] = metaFixedAttn
        # Passes
        elif (line.strip()).split()[2] == "passes":
            line = line.strip()
            line = line.split(" ")
            print("It's passes!")
            metaPasses = "# Passes\t" + line[1] + "\n"
            metadata["Passes"] = metaPasses
        # Column labels
        elif line[:9] == "# i, step":
            print("It's column labels!")
            metaCLabels = "i, step, norm, nbackground, signal, sbackground\n"
            metadata["CLabels"] = metaCLabels
        # Bias
        elif ("Top" in line) & ("Bot" in line) & ("IR" in line):
            print("It's plate biases!")
            metaBias = "# Bias\t" + line[2:]
            metadata["Bias"] = metaBias
        # Boxcar
        elif ("Gate" in line) & ("Amp" in line):
            print("It's Boxcar settings!")
            metaBoxcar = "# Boxcar\t" + line[2:]
            metadata["Boxcar"] = metaBoxcar
        # Garbage
        else:
            print(line[:30].strip())
