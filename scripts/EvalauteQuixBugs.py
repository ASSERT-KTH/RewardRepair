#!/usr/bin/python
import sys, os, subprocess,fnmatch, shutil, csv,re, datetime
import time
import QuixBugsDiscriminator
import pandas as pd

def getBugName(bugidex):
    bug = ''
    with open ('data/Quixbugs_metadata.csv','r') as metafile:
        lines = metafile.readlines()
        for l in lines:
            bid = l.split(',')[0]
            bugname = l.split(',')[1]
            if str(bid) in str(bugIndex) and str(bugIndex) in str(bid):
                bug = bugname
                break
                
    return bug





if __name__ == '__main__':
    
    df = pd.read_csv('./results/SemanticTrain0-Quixbugs-beam200-17.csv',sep='\t')
    
    for i in range(0, df.shape[0]):
        bugIndex = str(df.iloc[:,0][i]).strip()
        generated = str(df.iloc[:,2][i]).strip()
        target = str(df.iloc[:,3][i]).strip()
        bug = str(df.iloc[:,1][i]).strip()
#         l = ['18', '22', '30', '29', '11', '27', '24', '15', '35', '12', '1', '6', '19', '20', '7', '16']
#         if bugIndex not in l:
        if generated in target and target in generated:
            result = 'identical'
        else:    
            result = QuixBugsDiscriminator.getResults(bugIndex, generated)

        with open('Compilability-Quixbug-Result-Beam200.csv', 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='\t',  escapechar=' ', 
                quoting=csv.QUOTE_NONE)
            spamwriter.writerow([bugIndex,bug,result, generated,target])