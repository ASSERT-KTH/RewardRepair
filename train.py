import numpy as np
import pandas as pd
import torch, csv
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda
import gc
import warnings
import loader
import BearDiscriminator
import torch.autograd as autograd


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.buggy = self.data.buggy
        self.patch = self.data.patch

    def __len__(self):
        return len(self.patch)

    def __getitem__(self, index):
        buggy = str(self.buggy[index])
        buggy = ' '.join(buggy.split())

        patch = str(self.patch[index])
        patch = ' '.join(patch.split())

        source = self.tokenizer.batch_encode_plus([buggy], max_length= self.source_len,pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([patch], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


def semantic_training(generator, gen_opt, gen_tokenizer, adv_loader, device,epoch):

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
        
                
        bugcode = ids[0]
        end_index=getEndIndex(bugcode,32108) #2625 is the index for 'context',32108 is the index of 'context:'       
        bugcode = bugcode[3:end_index-1] #your index may be different!
        buggy = [gen_tokenizer.decode(bugcode, skip_special_tokens=True, clean_up_tokenization_spaces=True)]
           
            
        outputs = generator(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        print(f'original loss: {loss}')
        lm_logits = outputs[1]
        output = F.log_softmax(lm_logits, -1)
        preds_seq = output.max(2)[1]

        g = preds_seq[0]
        end_index=getEndIndex(g,1)            
        g = g[:end_index]      



        preds = [gen_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)]
        predstr = preds[0] 
        print(f'predstr: {predstr}')

        # identity discriminator
        identity_reward = identity_discriminator(buggy[0], predstr)

        reward = autograd.Variable(torch.FloatTensor([1.0]))

        if 'same' in identity_reward:
            reward = autograd.Variable(torch.FloatTensor([1.4]))
        else:            
            reward = validate_by_compiler(bugid, predstr)
        
        print(f'reward: {reward}')
  
        #combine cross entropy loss and compiler reward loss
        reward = reward.to(device)   
        loss = outputs[0]*reward
        print(f'semantic loss: {loss}')

        gen_opt.zero_grad()
        loss.backward()
        gen_opt.step()        

        recordData(epoch, bugid.item(), outputs[0].item(), reward.item(), predstr )             

def recordData(epoch, bugid, crossEntropLoss, reward, preds):
    with open('./logs.csv', 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter='\t',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([epoch, bugid, crossEntropLoss, reward, preds])   

                
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
    R = 0.2
    result = BearDiscriminator.getResults(bugid.item(), preds, rootPath)
    print(f'result: {result}')
    if 'failcompile' in result:
        rewardValue=1+R
    elif 'successcompile' in result:
        rewardValue=1-R
    elif 'passHumanTest' in result:
        rewardValue=1-R*2
    elif 'passAllTest' in result:
        rewardValue=1-R*3
    else:
        rewardValue=1
        
    
    return autograd.Variable(torch.FloatTensor([rewardValue]))

    

def syntrain(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    countInt = 0
    for _,data in enumerate(loader, 0):
    
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]


        if _%1000 ==0:
            print(f'Syntatic Train Epoch: {epoch}, Loss:  {loss.item()}')

        # we also save the model here in case of an accident during training
        if _%10000 ==0:
            model.save_pretrained(SAVE_MODEL)
            tokenizer.save_pretrained(SAVE_MODEL)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        
def valid( tokenizer, model, device, loader, optimizer):
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

            outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
            total_nb += 1  
            total_loss += loss.item()    

        print(f'Total Loss:  {total_loss}/{total_nb}')
        


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
        



def syntactic(epoch,syn_train_data_path):  
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    
    # Process data
    df = pd.read_csv(syn_train_data_path,encoding='latin-1',delimiter='\t', header=0, error_bad_lines=False)
    print(df.head())
    df = df[['bugid','buggy','patch']]
    print(df.head())
    

    # tokenzier for encoding the text
    if epoch == 0 and 'CoCoNut' in syn_train_data_path:
        model = T5ForConditionalGeneration.from_pretrained('t5-base', output_hidden_states=True)    
        tokenizer = T5Tokenizer.from_pretrained('t5-base',truncation=True)
        tokenizer.add_tokens(['{', '}','<','^','>=','<=','==','buggy:','context:'])

    else:
        model = T5ForConditionalGeneration.from_pretrained(SAVE_MODEL, output_hidden_states=True)    
        tokenizer = T5Tokenizer.from_pretrained(SAVE_MODEL,truncation=True)       


    device = 'cuda' if cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Creation of Dataset and Dataloader
    train_dataset=df.sample(frac=1.0, random_state = SEED).reset_index(drop=True)     
    print("TRAIN Dataset: {}".format(train_dataset.shape))

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, PATCH_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 2
        }    

    # Creation of Dataloaders for testing and validation. 
    training_loader = DataLoader(training_set, **train_params)
  

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE) 
            
    syntrain(epoch, tokenizer, model, device, training_loader, optimizer)
    model.save_pretrained(SAVE_MODEL)
    tokenizer.save_pretrained(SAVE_MODEL)
    print(f'Syntatic Train Model Saved: {epoch}')

        
        
def semantic(epoch):   
    
    gen = T5ForConditionalGeneration.from_pretrained(SAVE_MODEL, output_hidden_states=True)    
    gen_tokenizer = T5Tokenizer.from_pretrained(SAVE_MODEL,truncation=True)
    gen = gen.to(device)   
    gen_optimizer = torch.optim.Adam(params = gen.parameters(), lr=LEARNING_RATE)
    data_loader=getGeneratorDataLoader(semantic_train_data_path,gen_tokenizer,1)   

    print('\n---Semantic Training-----\nEPOCH %d\n--------' % (epoch+1))

    # train model
    semantic_training(gen, gen_optimizer, gen_tokenizer, data_loader, device, epoch)                       
    # save trained model       
    gen.save_pretrained(SAVE_MODEL)
    gen_tokenizer.save_pretrained(SAVE_MODEL)  
    print(f'Sementic Train Model Saved: {epoch}')

                
        
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(torch.__version__)
    gc.collect()
    torch.cuda.empty_cache()
    # This is a  small dataset to try
    syn_train_data_path_1= './data/CoCoNut.csv'
    syn_train_data_path_2= './data/MegaDiff-CodRep.csv'
    semantic_train_data_path= 'Bears_Training/BearsTraining.csv'
    SAVE_MODEL='./model/RewardRepair'
    rootPath='/your/path/'
    TRAIN_BATCH_SIZE = 20   
    TRAIN_EPOCHS = 15      # number of epochs to train 
    LEARNING_RATE = 1e-4    # learning rate
    SEED = 42               # random seed (default: 42)
    MAX_LEN = 512
    PATCH_LEN = 100    
    
    #We train the CoCoNut dataset
    for epoch in range(0,TRAIN_EPOCHS):
        syntactic(epoch,syn_train_data_path_1)
    
    #we train the syntactic training and semantic training
    for epoch in range(0,TRAIN_EPOCHS):
        syntactic(epoch,syn_train_data_path_2)
        if  (epoch>5 and epoch % 3 == 0) or epoch == TRAIN_EPOCHS-1:
            semantic(epoch)
