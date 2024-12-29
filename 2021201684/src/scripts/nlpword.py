import json
import math
import os
import string
import sys
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from keras.utils import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model


VocabularyPath = 'vocabulary.json'
TrainPath = 'train.txt'
WindowSize = 4
device = 'cpu'

class MyDataSet(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, item):
        input = self.inputs[item]
        target = self.targets[item]

        return input, target

    def __len__(self):
        return len(self.inputs)

def GetVocabulary(path): 
    file = open("book.txt", "r", encoding = "utf8")
    lines = []
    for i in file:
        lines.append(i) 
        
    data = ""
    for i in lines:
        data = ' '. join(lines)
     
    for i in data:
        if i in string.punctuation:
            data = data.replace(i, " ")
        elif 65 <= ord(i) <= 90:
            data = data.replace(i, chr(ord(i) + 32))
            
    WordList = []
    for i in data.split():
        WordList.append(i)       
    WordList = list(set(WordList))

    # log the VocabularyPath to save time
    WordtoindexDict = {w: i + 2 for i, w in enumerate(WordList)}
    WordtoindexDict['err'] = 0
    WordtoindexDict['<>'] = 1
    with open(VocabularyPath, 'w') as json_file:
        json_file.write(json.dumps(WordtoindexDict, indent=4))

    return WordtoindexDict

def GetDataSet(TrainPath, WordtoindexDict, WindowSize):
    def word2index(word):
        try:
            return WordtoindexDict[word]
        except:
            return 0  
    
    InputList = []
    OutputList = []
    f = open(TrainPath, 'r')
    lines = f.readlines()
    for sentence in lines:
        for i in sentence:
            if i in string.punctuation:
                sentence = sentence.replace(i, " ")
            elif 65 <= ord(i) <= 90:
                sentence = sentence.replace(i, chr(ord(i) + 32))
                
        WordList = sentence.split()
        if len(WordList) < WindowSize + 1: 
            WordList = ['<>'] * (WindowSize + 1 - len(WordList)) + WordList
        index_list = [word2index(word) for word in WordList]
            
        for i in range(len(WordList) - WindowSize):
            InputList.append(torch.tensor(index_list[i: i + WindowSize]))
            OutputList.append(torch.tensor(index_list[i + WindowSize]))

    dataset = MyDataSet(InputList, OutputList)
    return dataset

def TrainOneEpoch(model, loss_function, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    accu_loss = torch.zeros(1).to(device)  
    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file = sys.stdout)
    try:
        for step, data in enumerate(data_loader):       
            input, target = data
            pred = model(input.to(device))

            loss = loss_function(pred, target.to(device))
            loss.backward()
            accu_loss += loss.detach()

            data_loader.desc = "[train epoch {}] loss: {:.3f}, ppl: {:.3f}, lr: {:.5f}".format
            (
                epoch,
                accu_loss.item() / (step + 1),
                math.exp(accu_loss.item() / (step + 1)),
                optimizer.param_groups[0]["lr"]
            )

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()
            # update lr
            if lr_scheduler != None:
                lr_scheduler.step()
    except:
        print("End!")
    return accu_loss.item() / (step + 1), math.exp(accu_loss.item() / (step + 1))

def predict():
    model = load_model('nextword.keras')
    with open(VocabularyPath, "r") as f:
            WordtoindexDict = json.load(f)
     
    os.system('cls')
    while(True):
        text = input("Enter a line of text: ")
    
        if text == "exit":
            print("Exit now!")
            break
    
        else:
            try:
                Input = text.split(" ")
                Input = [WordtoindexDict[x] for x in Input]
                IInput = np.array([torch.tensor(Input)])
                Output = np.argsort(model.predict(IInput, verbose=0), axis=1)[:, -3:][:, ::-1]
                for word in Output[0]:
                    for key, value in WordtoindexDict.items():
                        if value == word:
                            print(key, end = ' ')
                            break
                print('')       
            except:
                print('Something wrong!')
                continue;

def main(): 
    mode = input('Enter the num 1 to predict, else to train\n')
    if mode == '1':
        predict()
        return
   
    # load the vocabulary
    if not os.path.exists(VocabularyPath):
        WordtoindexDict = GetVocabulary(VocabularyPath)
    else:
        with open(VocabularyPath, "r") as f:
            WordtoindexDict = json.load(f)
            
    VocSize = len(WordtoindexDict)
    print('The number of Vocabulary is', VocSize)

    # load the dataset
    TrainDataset = GetDataSet(TrainPath, WordtoindexDict, WindowSize)
    # train_loader = DataLoader(TrainDataset, batch_size = 64, shuffle = True, pin_memory = True, num_workers = 0)
    print('Load the dataset successfully.')

    X = []
    y = []

    for i in TrainDataset:
        X.append(i[0])
        y.append(i[1])
    
    X = np.array(X)
    y = np.array(y)

    y = to_categorical(y, num_classes = VocSize)
    
    model = Sequential()
    model.add(Embedding(VocSize, 10, input_length=1))
    model.add(LSTM(1000, return_sequences=True))
    model.add(LSTM(1000))
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(VocSize, activation="softmax"))
    
    checkpoint = ModelCheckpoint("nextword.keras", monitor='loss', verbose=1,
        save_best_only=True, mode='auto')

    reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)

    logdir='logsnextword1'
    tensorboard_Visualization = TensorBoard(log_dir=logdir)
    
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
    model.fit(X, y, epochs=150, batch_size=64, callbacks=[checkpoint, reduce, tensorboard_Visualization])

    

if __name__ == '__main__':
    main()
