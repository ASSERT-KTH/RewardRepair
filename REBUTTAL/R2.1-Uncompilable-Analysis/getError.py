#!/usr/bin/python
import sys, os, subprocess,fnmatch, shutil, csv, re, datetime
import time
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':
    with open('nohup.out','r') as result:
        lines = result.readlines()
        flag = False
        for i in range(0,len(lines)):
            l = lines[i]
            if 'BUILD FAILED' in l:
                flag = True
            if 'error: ' in l and flag:
                flag = False
                with open('test.txt','a') as analyze:
                    analyze.write(l)