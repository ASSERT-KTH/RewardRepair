import os,csv,subprocess
import pandas as pd

def getBugInfo(human_patch_code):
    human_patch_code=str(human_patch_code)
    human_patch_code = human_patch_code.replace(' ','').replace('\n','')
    
    with open('./data/D4JPairs.csv',encoding='utf-8') as bugs:
        lines = bugs.readlines()
        bugid = ''
        lineNo = ''
        action = ''
        buggy_code=''
        for l in lines:
            arr = l.split('\t')
            human_patch = arr[2]
            human_patch = human_patch.replace(' ','')
            if human_patch_code in human_patch and human_patch in human_patch_code:
                bugid = arr[0]
                lineNo = arr[3]
                action = arr[4]
                buggy_code = arr[1]
                break
    if bugid !='':
        if 'context' in buggy_code and 'buggy' in buggy_code:
            buggy_code=buggy_code.split('context')[0].split('buggy:')[1]
    return bugid,lineNo,action,buggy_code        
    


def getModifiedFiles(bugid):
    with open('./data/D4JMeta.csv',encoding='latin-1') as meta:
        lines = meta.readlines()
        modfiles=''
        for l in lines:
            arr = l.split('\t')
            bid = arr[0]
            if bugid in bid and bid in bugid:
                modfiles = arr[1]+'/'+arr[2]
                break
        return modfiles
                
        
        
        
def executeBug(bugid, modfiles, lineNo, action, generated_pred_code):
    try:
        exeresult = ''
        project=bugid.split('_')[0]
        bug=bugid.split('_')[1]
        tmp_folder = './tmp/'+bugid+'buggy'
        checkoutbuggystr='/home/heye/ganrepair/defects4j/framework/bin/defects4j  checkout -p '+project+' -v '+bug+'b -w '+tmp_folder
        os.system(checkoutbuggystr)

        modfiles=modfiles.replace('\n','.java')
        print(modfiles)
        os.chdir(tmp_folder)
        #cp target file
        predstr = ''


        with open(modfiles,'r') as originfile:
            lines = originfile.readlines()
            for i in range(len(lines)):
                if 'add' in action:
                    if i < (int(lineNo)-1) or i > (int(lineNo)-1):
                        predstr += lines[i]
                    else:
                        predstr += generated_pred_code+'\n'+lines[i]
                elif 'remove' in action:
                    if i < (int(lineNo)-1) or i > (int(lineNo)-1):
                        predstr += lines[i]
                elif 'replace' in action:
                    if i < (int(lineNo)-1) or i > (int(lineNo)-1):
                        predstr += lines[i]
                    else:
                        predstr += generated_pred_code+'\n'

        os.system('rm '+modfiles)   
        with open(modfiles,'a') as targetfile:
            targetfile.write(predstr)

        result = ''
        cmd = '/home/heye/ganrepair/defects4j/framework/bin/defects4j compile'   
        process = subprocess.Popen(cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        try:
            output, error = process.communicate(timeout=MAX_COMPILE_TIME)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        result = error
        result = result.decode('utf-8')
        result = result.split("\n")

        compiled = False

        for line in result:
            print(line)
            if 'OK' in line and 'Running ant' in line:
                compiled = True
                break

        if compiled: 
            print("compiled")
            exeresult = 'compiledPatch'
            cmd = '/home/heye/ganrepair/defects4j/framework/bin/defects4j test'   
            process = subprocess.Popen(cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            try:
                output, error = process.communicate(timeout=MAX_COMPILE_TIME)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            testresult = output
            testresult = testresult.decode('utf-8')
            testresults=testresult.split('\n')

            for line in testresults:
                print('test: ',line)
                if 'OK' in line:
                    exeresult='plausible'
                if 'Failing tests: 0' in line:
                    exeresult='plausible'
            print(testresults)


        else:
            exeresult = 'not-compiled'

        os.chdir('/home/heye/ganrepair/')            
        os.system('rm -rf '+tmp_folder)
        os.chdir('/home/heye/ganrepair/')
        return exeresult
    except:
        return "unknown"

        
    
if __name__ == '__main__':
    MAX_COMPILE_TIME = 60
    MAX_COMPILE_TIME = 300

    os.chdir('/home/heye/ganrepair/')       
    
    df = pd.read_csv('./data/D4J_V1_125Bugs.csv',encoding='utf-8',delimiter='\t')

    for index, row in df.iterrows():
        if index >-1:

            generated_pred_code = row['Generate Code']
            human_patch_code = row['Actual Code']

            human_patch_code=str(human_patch_code).replace('  ',' ')
            generated_pred_code=str(generated_pred_code).replace('  ',' ')


            bugid,lineNo,action,buggy_code = getBugInfo(human_patch_code)
            bresult=''


            if bugid=='':
                 bresult='unknown'
            else:   
                buggy_code = buggy_code.strip()
                generated_pred_code = generated_pred_code.strip()
                human_patch_code = human_patch_code.strip()

                no_space_buggy_code = buggy_code.replace(' ','')
                no_space_pred_code = generated_pred_code.replace(' ','')
                no_space_human_code = human_patch_code.replace(' ','')

#                 print('buggy_code',buggy_code)
#                 print('generated_pred_code',generated_pred_code)
#                 print('human_patch_code',human_patch_code)

#                 print('no_space_buggy_code',no_space_buggy_code)
#                 print('no_space_pred_code',no_space_pred_code)
#                 print('no_space_human_code',no_space_human_code)

                if buggy_code in generated_pred_code and generated_pred_code in buggy_code:
                    bresult = 'duplicate'
                elif no_space_buggy_code in no_space_pred_code and no_space_pred_code in no_space_buggy_code:
                    bresult = 'duplicate'
                elif human_patch_code in generated_pred_code and generated_pred_code in human_patch_code:
                    bresult = 'correct'
                elif no_space_human_code in no_space_pred_code and no_space_pred_code in no_space_human_code:
                    bresult = 'correct'

                else:       
                    modfiles = getModifiedFiles(bugid)
                    print(modfiles)
                    if ',' not in modfiles:
                        bresult = executeBug(bugid, modfiles,lineNo, action, generated_pred_code)
                    else:
                        bresult='2-files'

            buggy_code=buggy_code.replace('\n','').replace('  ',' ').replace('   ',' ')
            human_patch_code=str(human_patch_code).replace('\n','').replace('  ',' ').replace('   ',' ')
            generated_pred_code=generated_pred_code.replace('\n','').replace('   ',' ')


            os.chdir('/home/heye/ganrepair/')            
            with open('./data/Result_D4J_V1_125Bugs.csv','a',encoding='utf-8') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter='\t',  escapechar=' ', 
                                quoting=csv.QUOTE_NONE)
                spamwriter.writerow([bresult,bugid,buggy_code,human_patch_code,generated_pred_code])




        












