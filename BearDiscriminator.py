
#!/usr/bin/python
import sys, os, subprocess,fnmatch, shutil, csv,re, datetime
import time                       
    

def getResults(bugindex,preds,root):
        #Step 1, get bug inforation from bugId
        bugId,buggyFile,lineNo,action = getInfoFromIndex(bugindex,root)
        print(bugId,buggyFile,lineNo,action)
        print('========'+root+'======')
        #compile script location
        scriptdir = root+'/bears-benchmark/myscripts'
        os.chdir(scriptdir)
        print('now in the dir:'+scriptdir )
        repodir = root+'/Bears_Training'
        checkoutstring = 'python2.7 checkout_bug.py  --bugId  '+ bugId + '  --workspace '+ repodir
        #customized script to allow test running
        compilestring = 'python2.7 mycompile_bug.py  --bugId  '+ bugId + '  --workspace '+ repodir
        print(checkoutstring)
        os.system(checkoutstring)

        #apply patch
        os.chdir(repodir)   
    
        #prepare the diff
        prepare_diff(bugId,buggyFile,preds,lineNo,action,root)
    
        os.chdir(scriptdir)
        print(compilestring)
        results = os.popen(compilestring).read()
        results = results.split('Results :')[-1]
        print('test running result focus: '+ results)        
        
        execResult = ''
    
        # Not compile
        if 'COMPILATION ERROR' in results:
            execResult = 'failcompile'
            print('COMPILATION ERROR')
         
        # Not plausible        
        elif 'Tests in error:' in results or 'Failed tests:' in results or 'BUILD FAILURE' in results:
            execResult = 'successcompile'
            print('success compile' )        

        # plausible
        elif 'BUILD SUCCESS' in results:
            print('BUILD SUCCESS')
            if 'Failures: 0' in results and 'Errors: 0' in results:
                execResult = 'passHumanTest'
                print('Plausible!!')      
    

        os.system('rm -rf '+repodir+'/'+bugId)
        os.system('rm '+repodir+'/tmp.java')
        
        
        return execResult

        

def prepare_diff(bugId,buggyFile,preds,lineNo,action,root):
    print('buggyFile:',buggyFile)
    
    project = root+'/Bears_Training/'
    projectPath = project+bugId+'/'+buggyFile
    print('projectPath:',projectPath)
    
    if not os.path.exists(projectPath):
        return 'NoExist'
    else:
        originFile = project+'tmp.java'
        
        os.system('mv '+projectPath+' '+project+'tmp.java')
        predstr = ''
        
        with open(originFile,'r') as originfile:
            lines = originfile.readlines()
            if 'add' in action:
                before = lines[0:int(lineNo)-1]
                before.append(preds+'\n')
                predstr = before+lines[int(lineNo)-1:]
            elif '-' in action:
                num = action.split('-')[1]
                before = lines[0:int(lineNo)-1]
                before.append(preds+'\n')
                predstr = before+lines[int(lineNo)-1+int(num):]
                     
    
 
        with open(projectPath,'a') as targetfile:
            targetfile.write(' '.join([s for s in predstr]))
        
        
        
        
        
        
def getInfoFromIndex(bugIndex,root):
    print("BearsDiscriminator getInfoFromIndex: ",bugIndex)
    bugname= ''
    buggyFile=''
    lineNo=''
    action=''
    with open (root+'/Bears_Training/BearsMeta.csv','r') as metafile:
        lines = metafile.readlines()
        for line in lines:
            if str(bugIndex) in line.split('\t')[0] and line.split('\t')[0] in str(bugIndex):
                bugname = line.split('\t')[1]
                buggyFile = line.split('\t')[2]
                lineNo = line.split('\t')[3]
                action = line.split('\t')[4]
                break
    
    return bugname,buggyFile,lineNo,action


def getEvosuitTestPath(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            pattern = '*ESTest.java'
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name).split(path)[1]
    return ''

   
def getLibPath(bugId,repodir,root):
    libpath=''
    with open(root+'/Bears_Training/libpath.txt') as libs:
        lines = libs.readlines()
        for l in lines:
            bid = l.split('\t')[0]
            if bid in bugId and bugId in bid:
                libpath=l.split('\t')[1]
                libpath=libpath.replace('\n','')
                libpath=libpath.replace('repodir',repodir)
    return libpath


if __name__ == '__main__':
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    getResults('10','if  (channel  !=  null  &&  channel.getPipeline().get(HttpRequestDecoder.class)  !=  null')
