import os
import subprocess
import sys


def getResults(bugIndex, preds):
    print("Defects4JDiscriminator getResults")
    
    
    projectId,bugId, buggyFile, lineNo, action = getInfoFromIndex(bugIndex) 
    action=action.replace('\n','').replace('\t','')
    print("projectId,bugId, buggyFile, lineNo, action: "+  projectId,bugId, buggyFile, lineNo, action)

    
    #checkout the bug
    project_path = 'D4JResults/'
    if not os.path.exists(project_path):
        os.system('mkdir '+project_path)
    checkout_bug(projectId, bugId, project_path)
    
    
    prepare_diff(projectId,bugId,buggyFile,preds,lineNo,action)
    os.chdir(project_path+'/'+projectId+'_'+bugId)
    
    os.system('defects4j compile')
    os.system('defects4j test')
    
#     cmd += "defects4j compile"
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
#     result=''
#     if 'Running ant (compile)' in result:
#         result = str(result).split("Running ant (compile)")[1]
#         result=result.split('\n')
    
#     cmd = "defects4j test"
#         result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    
#     os.system()
    
def prepare_diff(projectId, bugId,buggyFile,preds,lineNo,action):
    print('buggyFile:',buggyFile)
    
    project = 'D4JResults//'
    projectPath = project+projectId+'_'+bugId+'/'+buggyFile
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
    
    



def checkout_bug(projectId, bugId, project_path):
    print("CompilerDiscriminator checkout bugs")
    print('defects4j checkout -p '+ projectId + ' -v ' + bugId + 'b  -w ' +  project_path +'/'+projectId+'_'+bugId)

    os.system('defects4j checkout -p '+ projectId + ' -v ' + bugId + 'b  -w ' +  project_path +'/'+projectId+'_'+bugId)
       
    
    
def getInfoFromIndex(bugIndex):
    projectId = ''
    bugId=''
    buggyFile=''
    lineNo=''
    action=''
    print("bugIndex:"+str(bugIndex))

    
    with open ('D4JResults/D4JMeta.csv','r') as metafile:
        lines = metafile.readlines()
        for line in lines:
            if str(bugIndex) in line.split('\t')[0] and line.split('\t')[0] in str(bugIndex):
                projectId = line.split('\t')[1]
                buggyFile = line.split('\t')[2]
                lineNo = line.split('\t')[3]
                action = line.split('\t')[4]
                break
                
    pId = projectId.split('_')[0]         
    bugId = projectId.split('_')[1]    
    print(projectId)
    return pId, bugId,buggyFile,lineNo,action   



if __name__ == '__main__':
    
    getResults('1', '')