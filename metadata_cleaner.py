# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:54:06 2017

@author: edmag
"""

# Interpret sloppy delay scan metadata and rewrite as formatted metadata
# Eric Magnuson, University of Virginia, VA

import os


def d_print(d):
    """Print out dictionary key : value. Returns None"""
    for k, v in d.items():
        print(k, ":", v)


def simple_path(folder, filename):
    """Tool to build file paths for the task of cleaning metadata.
    Returns path string"""
    base = "C:\\Users\\edmag\\Documents\\Work\\Data\\StaPD-Analysis"
    path = "\\".join([base, folder, filename])
    return path


def dlist_gen(path):
    """Produce a list of all "?_delay.txt" filenames in the given folder.
    Returns list of filenames"""
    flist = os.listdir(path)
    dlist = list()
    for i in range(len(flist)):
        if "_delay.txt" in flist[i]:
            dlist.append(path + flist[i])
    return dlist


def folder_clean(folder):
    """Given the target folder, cycle through every delay file and clean up
    the metadata.
    Returns None"""
    path = simple_path(folder, "")
    dlist = dlist_gen(path)
    for fname in dlist:
        metadata_cleaner(fname)


def metadata_cleaner(fname):
    """Reads metadata and column labels from the old style I handwrote and
    rewrites them in a cleaner style.
    "Filename", "Date", "Title", "DL-Pro", "DL-100", "Static", "MWOn", "MWf",
    "Attn" and the column labels are all required. Other metadata is included
    if found. Any metadata not recognized gets put under "Other".
    Returns None"""
    # initialize metadata dict with proper key order
    metadata = {"Filename": "Not Found", "Date": "Not Found",
                "Title": "Not Found", "DLPro": "Not Found",
                "DL100": "Not Found", "Static": "Not Found",
                "MWOn": "Not Found", "MWf": "Not Found", "Attn": "Not Found",
                "FixedAttn": None, "B2": None, "HP214B": None, "Oven": None,
                "Boxcar": None, "Offset": None, "Bias": None, "Passes": None,
                "Other": None, "CLabels": "Not Found"}
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
                metaStatic = "# Static\t" + line[2] + " " + line[3]
                if len(line) > 3:
                    metaStatic = metaStatic + " " + line[-1].strip()
                metaStatic = metaStatic + "\n"
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
                metaMWf = ("# MWf\t" + line[iF + 2] + " " + line[iF + 3][0:3]
                           + "\n")
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
                metaCLabels = \
                    "i, step, norm, nbackground, signal, sbackground\n"
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
            # Offset
            elif ("IR" in line) & ("Oven" in line) & ("Vis" not in line) \
                    & ("offset" in line):
                print("It's an IR/Oven offset!")
                metaOffset = "# Offset\t" + line[2:]
                metadata["Offset"] = metaOffset
            # B2 delay
            elif ("B2" in line) & ("T2" in line):
                print("It's a delay!")
                metaB2 = "# B2\t" + line[2:]
                metadata["B2"] = metaB2
            # Oven
            elif line.split(" ")[1] == "Oven":
                print("It's oven current!")
                line = line.split(" ")
                metaOven = "# Oven\t" + line[3] + " " + line[4].strip() + "\n"
                metadata["Oven"] = metaOven
            # HP214B pulse settings
            elif line.split(" ")[1] == "HP":
                print("It's HP Pulse settings!")
                metaHP214B = "# HP214B\t" + line[2:]
                metadata["HP214B"] = metaHP214B
            # other
            else:
                print(line)
                if metadata["Other"] is None:
                    metadata["Other"] = "# Other\t"
                line = line.strip()
                metadata["Other"] = metadata["Other"] + line[2:] + " ; "
    # done with file
    # put an newline to finish "Other"
    if metadata["Other"] is not None:
        metadata["Other"] = metadata["Other"] + "\n"
    # assign Filename
    metaFilename = "# Filename\t" + fname.split("\\")[-1] + "\n"
    metadata["Filename"] = metaFilename
    # print metadata in order
    print("\nMetadata output\n")
    metadata_string = ""
    for k, v in metadata.items():
        if metadata[k] is not None:
            metadata_string = metadata_string + v
    print(metadata_string)
    # writing to new file
    fnamenew = fname.split("\\")
    fnamenew[-3] = "Modified Data"
    fnamenew = "\\".join(fnamenew)
    print(fname)
    print(fnamenew)
    with open(fname, 'r') as old:
        with open(fnamenew, 'w') as new:
            new.write(metadata_string)  # write the new metadata and clabels
            for line in old:
                # don't repeat comments and column labels
                if (line[0] != "#") & (line[:9] != "# i, step"):
                    new.write(line.strip() + "\n")

# main program starts here
# File location
# path = "C:\\Users\\edmag\\Documents\\Work\\Data\\StaPD-Analysis" \
#        + "\\Modified Data\\MetaTest"
# filename = "\\8_delay.txt"
# fname = path + filename
# metadata_cleaner(fname)
