import os
import subprocess
import sys


def getResults(bugIndex, preds):
    print("QuixBugsDiscriminator getResults")
    print("preds"+preds)

    
    bugId, buggyFile, lineNo = getInfoFromIndex(bugIndex) 
    lineNo=lineNo.replace('\n','').replace('\t','')
    print("bugId, buggyFile, lineNo: "+  bugId, buggyFile, lineNo)

    #The bug path:
    quixbugroot='./quixbugs-experiment'
    current_bug = quixbugroot+buggyFile
    print("current_bug: "+ current_bug)
    
    
    #Get predicts and generate diffs:
    project_path = 'quixbugs-experiment/tmp/'+bugId

    prepare_diff(project_path,bugId,current_bug,preds,lineNo)

    
    #copy and replace the patch file to buggyFile
    os.system('cp  ' +project_path+'/'+bugId+'.java  '+current_bug)
    
    
    #compile
    os.chdir('quixbugs-experiment')
    result = os.popen('mvn compile').read()
    print(result)
    os.chdir('..')
    
    lines = result.split('\n')
    
    compile_error = True 
    PassHumanTest = False
    PassRGTTest = False
    for line in lines:
        if 'BUILD SUCCESS' in line:
            compile_error =  False
            
    if compile_error:
       # The end back to the original buggy file     
        os.system('mv  '+project_path+'/'+bugId+'.java    '+project_path+'/'+bugId+'_target.java')   
        os.system('mv  '+project_path+'/'+bugId+'_origin.java  '+ project_path+'/'+bugId+'.java' ) 
        os.system('cp  ' +project_path+'/'+bugId+'.java  '+current_bug)
        os.system('rm -rf  ' +project_path)
        
        return "failcompile"
        //some code
    
            result=os.popen(exestr).read() 
            print('This is the RGT test result')
            print(result)
            
            if 'OK' in result:
                PassRGTTest =  True
                
            os.system('mv  '+project_path+'/'+bugId+'.java    '+project_path+'/'+bugId+'_target.java')   
            os.system('mv  '+project_path+'/'+bugId+'_origin.java  '+ project_path+'/'+bugId+'.java' ) 
            os.system('cp  ' +project_path+'/'+bugId+'.java  '+current_bug)
            os.system('rm -rf  ' +project_path)
            
            if PassRGTTest:
                return "passAllTest"
            else:
                return "passHumanTest"
        
            
            
    
    
def prepare_diff(project_path,bugId,bugpath,preds,lineNo):

    if not os.path.exists(project_path):
        os.system('mkdir -p  '+project_path)
    
    
    os.system('cp  ' +bugpath+' '+project_path)

    origin = project_path+'/'+bugId+'_origin.java'
    os.system('mv  ' +project_path+'/'+bugId+'.java'+'  '+origin)
    
    predstr = ''
    print('bugId',bugId)
    //some code
    
 
    with open(project_path+'/'+bugId+'.java','a') as targetfile:
        targetfile.write(predstr)
    
                
        
        

        
        
        
        
        
    
def getInfoFromIndex(bugIndex):
    print("QuixBugsDiscriminator getInfoFromIndex")
    bugIndex = bugIndex.item()
    print("bugIndex "+str(bugIndex))


    with open ('./data/Quixbugs_metadata.csv','r') as metafile:
        lines = metafile.readlines()
        for line in lines:
            if str(bugIndex) in line.split(',')[0] and line.split(',')[0] in str(bugIndex):
                return line.split(',')[1],line.split(',')[2],line.split(',')[3]

