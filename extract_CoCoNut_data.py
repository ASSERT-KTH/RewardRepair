#!/usr/bin/python

if __name__ == '__main__':
    tmplines = ''
    with open('rem.txt','r') as rmfile:
        rmlines=rmfile.readlines()
    with open('add.txt','r') as addfile:
        addlines=addfile.readlines()
    with open('context.txt','r') as ctxfile:
        ctxlines=ctxfile.readlines()

    header = 'bugid \t buggy \t patch \n'
    with open('pretrain.csv','a') as tmpfile:
        tmpfile.write(header)

    for i in range(0,3241966):
        rmline = rmlines[i]
        addline = addlines[i]
        ctxline = ctxlines[i]
        rmline = rmline.replace('\t', ' ')
        rmline = rmline.strip()          
        addline = addline.replace('\t', ' ')
        addline = addline.strip()                              
        ctxline = ctxline.replace('\t', ' ')
        ctxline = ctxline.strip()
        tmplines = str(i+1) +'\t'+ 'buggy: ' +rmline +'\t'+' context: '+ ctxline +'\t'+ +addline+'\n'
        with open('pretrain.csv','a') as tmpfile:
            tmpfile.write(tmplines)
