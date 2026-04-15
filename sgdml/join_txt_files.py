#!/usr/bin/env python3

"""
Rangsiman Ketkaew
April 23rd, 2020

Normal usage:
Example txt files we would like to join them as one txt file:
    file_1.txt, file_2.txt, file_3.txt, ...
    
$ python join_txt_files.py 'file_*.txt' all_in_one.txt
"""

import sys
import glob


try:
    pattern = str(sys.argv[1])
    output = str(sys.argv[2])
except:
    exit("You need to specify\n\
 1.A wildcard name (with asterick (*) as a placeholder) representing the consecutive files\n\
 2.Name of output file.")

# combine all output files into one file
txt = glob.glob(pattern)
print("Files to join:")
with open(output, 'w+') as onetxt:
    for n, t in enumerate(txt):
        print("No. {0} : {1}".format(n, t))
        with open(t, 'r') as inp:
            for line in inp:
                onetxt.write(line)

print("Done: All files have been joined and written to \'{0}\'".format(output))