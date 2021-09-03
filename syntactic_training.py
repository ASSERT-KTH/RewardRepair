# Importing stock libraries
import numpy as np
import pandas as pd
import torch, csv
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
# # Setting up the device for GPU usage
from torch import cuda
import gc
import warnings
import loader

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


    
def getD4JBugName(bugIndex):
    bug = ''
    with open ('data/D4JMeta.csv','r') as metafile:
        lines = metafile.readlines()
        for l in lines:
            bid = l.split('\t')[0]
            bugname = l.split('\t')[1]
            if str(bid) in str(bugIndex) and str(bugIndex) in str(bid):
                bug = bugname
                break
                
    return bug



def getQuixBugName(bugIndex):
    bug = ''
    with open ('data/Quixbugs_metadata.csv','r') as metafile:
        lines = metafile.readlines()
        for l in lines:
            bid = l.split(',')[0]
            bugname = l.split(',')[1]
            if str(bid) in str(bugIndex) and str(bugIndex) in str(bid):
                bug = bugname
                break
                
    return bug
    

def train(epoch, tokenizer, model, device, loader, optimizer):
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


        if _%500 ==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        if _%5000 ==0:
            model.save_pretrained('./model/T5Coconut-Megadiff')
            tokenizer.save_pretrained('./model/T5Coconut-Megadiff')

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
        


        
def test(epoch, tokenizer, model, device, loader):
    return_sequences = 200
    model.eval()

    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            gc.collect()
            torch.cuda.empty_cache()
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            bugid = data['bugid'].to(device, dtype = torch.long)
            
            
         
            
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=100, 
                num_beams=return_sequences,
                length_penalty=1.0, 
                early_stopping = True,
                num_return_sequences=return_sequences,
                num_beam_groups = 1,
                output_scores=True
                )
           

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]


            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            target = target[0]

            
            bugname = getD4JBugName(bugid.item())
#             bugname = getQuixBugName(bugid.item())
            
            with open('./results/D4J-beam200-'+'.csv', 'a') as csvfile:
                filewriter = csv.writer(csvfile, delimiter='\t',quotechar='"',quoting=csv.QUOTE_MINIMAL)
                for i in range(0,return_sequences):
                    filewriter.writerow([bugid.item(),bugname, preds[i],target])






def main():
 
    TRAIN_BATCH_SIZE =20    # input batch size for training (default: 64)
    VALID_BATCH_SIZE = 20   # input batch size for testing (default: 1000)
    TRAIN_EPOCHS = 2       # number of epochs to train (default: 10)
    VAL_EPOCHS = 1 
    LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
    SEED = 42               # random seed (default: 42)
    MAX_LEN = 512
    SUMMARY_LEN = 100 

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained('./model/SyntacticModel')
    tokenizer.add_tokens(['{', '}','<','^'])

    model = T5ForConditionalGeneration.from_pretrained('./model/SyntacticModel')
    device = 'cuda' if cuda.is_available() else 'cpu'

    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = model.to(device)
    

    df = pd.read_csv('./data/MegaDiff-CodRep.csv',encoding='latin-1',delimiter='\t', header=0, error_bad_lines=False)
    print(df.head())
    df = df[['bugid','buggy','patch']]
    print(df.head())
    

    eval_df = pd.read_csv('./data/MegaDiff-CodRep.csv',encoding='latin-1',delimiter='\t')
    print(eval_df.head())
    eval_df = eval_df[['buggy','patch']]
    print(eval_df.head())
    
#       Bugsjar.csv  D4JPairs.csv   Defects4JV2Training Quixbugs
    test_df = pd.read_csv('./data/ D4JPairs.csv',encoding='latin-1',delimiter='\t')
    print(test_df.head())
    test_df = test_df[['bugid','buggy','patch']]
    print(test_df.head())

    
    # Creation of Dataset and Dataloader
    train_dataset=df.sample(frac=1.0, random_state = SEED).reset_index(drop=True)     
    val_dataset=eval_df.sample(frac=1.0, random_state = SEED).reset_index(drop=True)
    test_dataset=test_df.sample(frac=1.0, random_state = SEED).reset_index(drop=True)


    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VALID Dataset: {}".format(val_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))



    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    test_set = loader.GeneratorDataset(test_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 2
        }

    val_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 2
        }
    
    test_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 2
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)
  
  

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    Training loop
    print('Initiating Fine-Tuning for the model on our dataset')
    valid(tokenizer, model, device, val_loader, optimizer)
    
    predictions, actuals = test(1, tokenizer, model, device, test_loader)
    final_df = pd.DataFrame({'Generate Code':predictions,'Actual Code':actuals})
    final_df.to_csv('./data/validation'+'.csv')
    

    for epoch in range(0,20):
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        model.save_pretrained('./model/SyntacticModel')
        tokenizer.save_pretrained('./model/SyntacticModel')
                
        test(epoch, tokenizer, model, device, test_loader)
        
        
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    device = 'cuda' if cuda.is_available() else 'cpu'
    gc.collect()
    torch.cuda.empty_cache()
    main()
