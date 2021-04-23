import numpy as np
import pandas as pd
import torch,sys
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import warnings
from torch import cuda
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import loader
import torch.autograd as autograd
import csv
import Discriminator


def train_generator_PG(generator, gen_opt, gen_tokenizer, adv_loader, device,epoch):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    generator.train()
    for _,data in enumerate(adv_loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == gen_tokenizer.pad_token_id] = -100
        
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)
        bugid = data['bugid'].to(device, dtype = torch.long)
        print(f'bugid: {bugid}')
        
        adv_train_count = 0
        
        continueAdvTraining = True
        
        
#         print(f'ids: {ids}')
        bugcode = ids[0]
        end_index=getEndIndex(bugcode,2625)         #2625 is the index for 'context'          
        bugcode = bugcode[3:end_index-1]
        buggy = [gen_tokenizer.decode(bugcode, skip_special_tokens=True, clean_up_tokenization_spaces=True)]

            
            
        while(continueAdvTraining):
            outputs = generator(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
            loss = outputs[0]
            print(f'loss: {loss}')
            lm_logits = outputs[1]
            output = F.log_softmax(lm_logits, -1)   
                
            # identity discriminator
            identity_reward = identity_discriminator(buggy[0], output)
            
            if 'same' in identity_reward:
                reward = autograd.Variable(torch.FloatTensor([2.0]))
            else:
            
                # combine cross entropy loss and compiler reward loss
                reward = validate_by_compiler(bugid, predstr)
                print(f'reward: {reward}')
            
            reward = reward.to(device)   
            loss = outputs[0]*(1-reward)
            print(f'loss: {loss}')

            if adv_train_count % 100 ==0:
                gen_opt.zero_grad()
                loss.backward()
                gen_opt.step()
            
            adv_train_count +=1
            print(f'adv_train_count: {adv_train_count}')
            
            
            recordData(epoch, bugid.item(), adv_train_count, outputs[0].item(), reward.item(), predstr )         
            
            traincount=5

            if adv_train_count > traincount:
                continueAdvTraining = False
   






    

                
def getEndIndex(g,index):
    end_index=0
    for i in g:
        end_index+=1
        # 1 for </s>
        if i == index:
            break
    return end_index
                
                
def identity_discriminator(buggy, predstr):
    print(f'buggy: {buggy}')
    print(f'predstr: {predstr}')
    if buggy in predstr and predstr in buggy:
        return 'same'
    else:
        return 'different'
    
    
    
    
    
    
    
def validate_by_compiler(bugid, preds):
    
    result = Discriminator.getResults(bugid, preds)
#     result = 'failcompile' 
    print(f'result: {result}')
    if 'failcompile' in result:
        return autograd.Variable(torch.FloatTensor([0.2]))
    elif 'successcompile' in result:
        return autograd.Variable(torch.FloatTensor([0.4]))
    elif 'passHumanTest' in result:
        return autograd.Variable(torch.FloatTensor([0.6]))
    elif 'passAllTest' in result:
        return autograd.Variable(torch.FloatTensor([0.8]))



def recordData(epoch,bugid, adv_train_count, crossEntropLoss, reward, preds):
    with open('./motivating_training_quixbugs_logs.csv', 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter='\t',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([epoch, bugid, adv_train_count, crossEntropLoss, reward, preds])
        
def valid( tokenizer, model, device, loader,epoch):
    model.eval()
    total_loss = 0 
    total_nb=0
    with torch.no_grad():
        for _,data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
            loss = outputs[0]
            total_nb += 1  
            total_loss += loss.item()    

        print(f'Total Loss:  {total_loss}/{total_nb}')
        with open('./adv_training_valid_logs.csv', 'a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter='\t',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow([epoch,(total_loss/total_nb)])
        
        
def test(tokenizer, model, device, loader,epoch):
    return_sequences = 100
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=100,
                repetition_penalty=5.0, 
                length_penalty=1.0, 
                early_stopping=True,
                num_return_sequences=return_sequences
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            print(preds[0])
            print(preds[1])
            print(preds[9])

            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            target = target[0]
            
            
            with open('./data/D4JV1_125Bugs_adv3'+'.csv', 'a') as csvfile:
                filewriter = csv.writer(csvfile, delimiter='\t',quotechar='"',quoting=csv.QUOTE_MINIMAL)
                for i in range(0,return_sequences):
                    filewriter.writerow([preds[i],target])
         
                
    return predictions, actuals


def getGeneratorDataLoader(filepatch,tokenizer,batchsize):
    df = pd.read_csv(filepatch,encoding='latin-1',delimiter='\t')
    print(df.head(1))
    
    df = df[['bugid','buggy','patch']]

    params = {
        'batch_size': batchsize,
        'shuffle': True,
        'num_workers': 0
        }

    dataset=df.sample(frac=1.0, random_state = SEED).reset_index(drop=True)
    target_set = loader.GeneratorDataset(dataset, tokenizer, MAX_LEN, PATCH_LEN)
    target_loader = DataLoader(target_set, **params)
    return target_loader


def getValidTestDataLoader(path,tokenizer,batchsize):
    eval_df = pd.read_csv(path,encoding='latin-1',delimiter='\t')
    eval_df = eval_df[['buggy','patch']]
    dataset=eval_df.sample(frac=1.0, random_state = SEED).reset_index(drop=True)
    target_set = loader.CustomDataset(dataset, tokenizer, MAX_LEN, PATCH_LEN)
    params = {
        'batch_size': batchsize,
        'shuffle': False,
        'num_workers': 2
        }
    target_loader = DataLoader(target_set, **params)
    return target_loader









def run_adv_training():

    ADV_TRAIN_PATH= './data/motivating.csv'
#     ADV_TRAIN_PATH= './data/adversial_training_Quixbugs.csv'
    VALID_PATH='./data/CodRep4.csv'
    TEST_PATH='./data/D4JPairs.csv'
    
    gen = T5ForConditionalGeneration.from_pretrained('./model/T5Coconut_adv3', output_hidden_states=True)    
    gen = gen.to(device)   
    gen_tokenizer = T5Tokenizer.from_pretrained('./model/T5Coconut_adv3',truncation=True)
    gen_optimizer = torch.optim.Adam(params = gen.parameters(), lr=LEARNING_RATE)


    adv_loader=getGeneratorDataLoader(ADV_TRAIN_PATH,gen_tokenizer,1)   
    valid_loader=getValidTestDataLoader(VALID_PATH,gen_tokenizer,VALID_BATCH_SIZE)   
    test_loader=getValidTestDataLoader(TEST_PATH,gen_tokenizer,1) 


#     valid(gen_tokenizer, gen, device, valid_loader,'before adversial training')

    for epoch in range(ADV_TRAIN_EPOCHS):
#         print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        print('\nAdversarial Training Generator : ', end='')
        train_generator_PG(gen, gen_optimizer, gen_tokenizer, adv_loader, device, epoch)         
        
    
#         gen.save_pretrained('./model/T5Coconut_adv'+str(epoch))
#         gen_tokenizer.save_pretrained('./model/T5Coconut_adv'+str(epoch))

#         print(f'Validating on valid dataset *********: {epoch}')
#         valid(gen_tokenizer, gen, device, valid_loader,epoch+1)
        
#         print(f'Validating on test dataset *********: {epoch}')
#         test(gen_tokenizer, gen, device, test_loader, epoch)       
#         print('Output Files generated for review')
        

        
        
        
        
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    SEED=42
    ADV_TRAIN_EPOCHS = 1   
    LEARNING_RATE = 1e-4
    VALID_BATCH_SIZE = 20
    MAX_LEN = 512
    PATCH_LEN = 100 
    device = 'cuda' if cuda.is_available() else 'cpu'
    run_adv_training()
