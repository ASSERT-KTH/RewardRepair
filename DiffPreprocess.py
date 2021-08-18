#!/usr/bin/python
import sys, os, subprocess,fnmatch, shutil, csv,re, datetime
import time, json

def travFolder(dir):
    listdirs = os.listdir(dir)
    pattern="*.diff"
    for f in listdirs:
        if os.path.isfile(os.path.join(dir, f)) and fnmatch.fnmatch(f, pattern):
            path=os.path.join(dir, f)
            pfilename=f.replace('.diff','')
            js = '{'
            bugglines=''
            patchlines=''
            with open(path,'r') as difffile:
                lines = difffile.readlines()
                difflocations=0
                buggymethodstartline=0
                buggymethodendline=0
                difflines =''
                #first locate the diff lines: 
                for i in range(5,len(lines)):
                    l = lines[i]
                    if l.startswith('-') or l.startswith('+'):
                        startword = l[1:]
                        startword =  startword.strip()
                        if not l.startswith('/') and not l.startswith('*') and not l.startswith('import') and not l.startswith('System.out') and not l.startswith('Logger') and not l.startswith('log.info') and  not l.startswith('logger'):
                            difflocations=i
                            break
                #find buggy and patched lines
                for i in range(5,len(lines)):
                    l = lines[i]
                    startword = l[1:]
                    startword =  startword.strip()
                    if l.startswith('-'):
                        if not startword.startswith('/') and not startword.startswith('*') and not startword.startswith('import') and not startword.startswith('System.out') and not startword.startswith('Logger') and not startword.startswith('log.info') and  not startword.startswith('logger')  and  not startword.startswith('//'):
                            difflines = difflines + startword+' '
                    if l.startswith('+'):
                        if not startword.startswith('/') and not startword.startswith('*') and not startword.startswith('import') and not startword.startswith('System.out') and not startword.startswith('Logger') and not startword.startswith('log.info') and  not startword.startswith('logger') and  not startword.startswith('//'):
                            patchlines = patchlines + startword+' '


                buggymethodstartline=difflocations
                bugglines = "buggy: "+difflines+' context: '
                
                #find method start line:
                # for i in range(1,100):
                #     previouslineIndex = difflocations-i
                #     if previouslineIndex > 5:
                #         l = lines[previouslineIndex].strip()
                #         if (l.startswith('public') or l.startswith('private'))  and l.endswith('{'):
                #             buggymethodstartline=previouslineIndex
                #             break
                
                # #find method end line:
                # buggymethodendline=difflocations
                # for i in range(1,100):
                #     nextlineIndex = difflocations+i
                #     if nextlineIndex < len(lines):
                #         l = lines[nextlineIndex].strip()
                #         if (l.startswith('public') or l.startswith('private'))  and l.endswith('{'):
                #             buggymethodendline=nextlineIndex
                #             break
                
                buggymethodstartline = difflocations - 10
                buggymethodendline = difflocations + 10
                print pfilename, buggymethodstartline,difflocations,buggymethodendline
                
                # context lines 
                if buggymethodstartline < difflocations and buggymethodendline > difflocations:
                    for i in range(buggymethodstartline,buggymethodendline):
                        if i >0 and i < len(lines):
                            l = lines[i].strip()
                            l = l.replace('\t','')
                            if not l.startswith('/') and not l.startswith('*') and not l.startswith('import') and not l.startswith('System.out') and not l.startswith('Logger') and not l.startswith('log.info') and  not l.startswith('logger') and  not l.startswith('@'):
                                if l.startswith('-') :
                                    if l.endswith(';'):
                                        bugglines+=l[1:]+' '
                                    else:
                                        bugglines+=l[1:]+' '                         
                                elif l != '' and not l.startswith('+'):
                                    if l.endswith(';'):
                                        bugglines+=l+' '
                                    else:
                                        bugglines+=l+' '
                    bugglines = bugglines.replace('\t','').replace('  ',' ').replace('   ',' ')
                    patchlines = patchlines.replace('\t','')
                    
                    # fname = pfilename.split('_')[-1]
                    # pfilename = pfilename.split('_'+fname)[0]
                    #in case empty patch
                    if len(bugglines) > 20:
                        # patchlines = 'patch '+patchlines
                        patchlines = patchlines
                        with open('test.csv', 'a') as csvfile:
                            spamwriter = csv.writer(csvfile, delimiter='\t',  escapechar=' ', 
                                quoting=csv.QUOTE_NONE)
                            if difflines=='':
                                #spamwriter.writerow([pfilename,bugglines,patchlines,difflocations-2,'add'])
                                spamwriter.writerow([pfilename,bugglines,patchlines])

                            elif patchlines=='':
                                # spamwriter.writerow([pfilename,bugglines,patchlines,difflocations-2,'remove'])
                                spamwriter.writerow([pfilename,bugglines,patchlines])

                            else:
                                # spamwriter.writerow([pfilename,bugglines,patchlines,difflocations-2,'replace'])
                                spamwriter.writerow([pfilename,bugglines,patchlines])


                 
                

        elif not os.path.isfile(os.path.join(dir, f)):
            travFolder(dir+'/'+f)



if __name__ == '__main__':
    dirs=['./1']
    for dir in dirs:
        travFolder(dir)

