import os
import subprocess
import sys


def getResults(bugIndex, preds):
    print("Defects4JDiscriminator getResults")
    
    
    projectId,bugId, buggyFile, lineNo, action = getInfoFromIndex(bugIndex) 
    action=action.replace('\n','').replace('\t','')
    print("projectId,bugId, buggyFile, lineNo, action: "+  projectId,bugId, buggyFile, lineNo, action)

    
    #checkout the bug
    project_path = 'semantic_training/Defects4J_projects'
    if not os.path.exists(project_path):
        os.system('mkdir '+project_path)
    checkout_bug(projectId, bugId, project_path)
    
    
#     #postprocess the preds
#     tmp_path = 'semantic_training/tmp'
#     if not os.path.exists(tmp_path):
#         os.system('mkdir '+tmp_path)
#     postprocess_predicts(preds)

    
#     #generate patch and diff
#     patch_root_path = 'semantic_training/Defects4J_patches'
#     patch_path = patch_root_path+'/'+ projectId+'_'+bugId
#     if not os.path.exists(patch_path):
#         os.system('mkdir -p  '+patch_path)
     
#     bug_project_path='semantic_training/Defects4J_projects/'+projectId+'_'+bugId
#     buggy_file_path=bug_project_path+'/'+buggyFile
#     generate_patch(buggy_file_path,lineNo, tmp_path+'/predictions_JavaSource.txt', patch_path)
        
#     #validate patch
#     status = validate_patch(bug_project_path, buggy_file_path, patch_path)
#     print('validate_patch  status: '+status)
    
#     if os.path.exists(patch_path):
#         os.system('rm -rf  '+patch_path)
#     if os.path.exists(tmp_path):
#         os.system('rm -rf  '+tmp_path)
#     if os.path.exists(bug_project_path):
#         os.system('rm -rf  '+bug_project_path)
    
    status = ''
    return status
    


def validate_patch(bug_project_path,buggy_file_path, patch_path):
    
    buggyFileName = buggy_file_path.split('/')[-1]    
    print('cp '+ patch_path + '/1/'+buggyFileName +' '+buggy_file_path)
    os.system('cp '+ patch_path + '/1/'+buggyFileName +' '+buggy_file_path)
    
    cmd = "cd " + bug_project_path + ";"
    cmd += "defects4j compile"
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    result=''
    if 'Running ant (compile)' in result:
        result = str(result).split("Running ant (compile)")[1]
        result=result.split('\n')

    compile_error = True 
    PassHumanTest = False
    for line in result:
        print('line....'+line)
        if('OK' in line):
            compile_error = False
            break
    if(compile_error):
        return "failcompile"
    else:
        cmd = "defects4j test"
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        if 'Running ant (run.dev.tests)' in result:
            result = str(result).split("Running ant (run.dev.tests)")[1]
            result=result.split('\n')
            for line in result:
                print('line....'+line)
                if('Failing tests: 0' in line):
                    noPassHumanTest = True
                    break
        if(PassHumanTest):
            return "passHumanTest"
        else:
            return "successcompile"

        


def generate_patch(buggy_file_path,lineNo, tmp_path, patch_path):
    print('python3 semantic_training/generatePatches.py '+buggy_file_path +' ' +lineNo + ' '+  tmp_path  +' ' +  patch_path)
    os.system('python3 semantic_training/generatePatches.py '+buggy_file_path +' ' +lineNo + ' '+  tmp_path  +' ' +  patch_path)
    
    buggyFileName = buggy_file_path.split('/')[-1]    
    for patch in os.listdir(patch_path):
        print('diff -u -w '+buggy_file_path +'  '+ patch_path+ '/'+patch+'/'+buggyFileName  +' > '+ patch_path+'/'+patch+'/diff')
        os.system('diff -u -w '+buggy_file_path +'  '+ patch_path+ '/'+patch+'/'+buggyFileName  +' > '+ patch_path+'/'+patch+'/diff')

    
    
    


def postprocess_predicts(preds):

    with open ('semantic_training/tmp/predictions.txt','w') as predictions:
        predictions.write(preds)          
            
    os.system('python3 semantic_training/postPrcoessPredictions.py semantic_training/tmp/predictions.txt semantic_training/tmp')




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
    print("CompilerDiscriminator getInfoFromIndex")
    bugIndex = bugIndex.item()
    print("bugIndex:"+str(bugIndex))

    
    with open ('semantic_training/D4JMeta.csv','r') as metafile:
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

