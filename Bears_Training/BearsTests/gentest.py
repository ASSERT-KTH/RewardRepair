#!/usr/bin/python
import sys, os, subprocess,fnmatch, shutil, csv,re, datetime

def compile():
    with open('./meta.csv', 'r') as file:
        lines = file.readlines()
        print(lines)
        for line in lines:
            arrs = line.split(',')
            bugid=arrs[0]
            clazz=arrs[1]
            #compile
            scriptdir = '/home/wasp/Desktop/research/bears-benchmark/scripts'
            os.chdir(scriptdir)
            print('now in the dir:'+scriptdir )
            repodir = '/home/wasp/Desktop/research/bears-bugs'
            compilestring = 'python compile_bug.py  --bugId  '+ bugid + '  --workspace '+ repodir
            print(compilestring)
            os.system(compilestring)



def generate_test():
    with open('./tmp.csv', 'r') as file:
        lines = file.readlines()
        print(lines)
        for line in lines:
            arrs = line.split(',')
            bugid=arrs[0]
            clazz=arrs[1]
	    jars=arrs[3]
            repodir = '/home/wasp/Desktop/research/bears-bugs/'+bugid+'/target/classes'
            repodirjar='/home/wasp/Desktop/research/bears-bugs/'+bugid+'/'+jars
	    print(repodirjar)
	    teststring = 'java -jar /home/wasp/Desktop/research/Evosuite/evosuite-1.0.6.jar   -projectCP  '+  repodir+':'+repodirjar + '    -class   '+clazz + '   -Dsearch_budget=100  ' 
            print(teststring)
            os.system('mkdir tests/'+bugid)
            for i in range(0,10):
                currentdir= 'tests/'+bugid+'/'+str(i)
                os.system('mkdir '+currentdir)
                os.chdir(currentdir)
                os.system(teststring)
                os.chdir('/home/wasp/Desktop/research/Evosuite')




if __name__ == '__main__':
    #compile()
    generate_test()
