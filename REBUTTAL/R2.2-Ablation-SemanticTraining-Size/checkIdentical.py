#!/usr/bin/python


import pandas as pd
import sys

if __name__ == '__main__':
    csvfile=sys.argv[1]
    df = pd.read_csv(csvfile,sep='\t')

    print(len(df))
    idnetical = 0
    identical_bug_name=[]
    identical_bug=[]

    for i in range(0, df.shape[0]):
        bug = str(df.iloc[:,1][i]).strip()
        generated = str(df.iloc[:,2][i]).strip()
        target = str(df.iloc[:,3][i]).strip()
        generatedNoSpace = generated.replace(' ','')
        targetNoSpace = target.replace(' ','')
        if generatedNoSpace in targetNoSpace and targetNoSpace in generatedNoSpace:
            if bug not in identical_bug_name:
                identical_bug_name.append(bug)
                print(bug)
                idnetical=idnetical+1

    print(len(identical_bug_name))
