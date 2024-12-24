#Needed libraries

from torch.utils.data import Dataset
import torch

import numpy as np
import json

from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Whitespace

from icecream import ic

class Tokenized_Dataset(Dataset):
    """
    Class that given a tokenizer and the provided data json is able to iterate 
    over a window of seq_lenght of all the texts in the dataset.

    params:
        json_file: Path to dataset json_file
        tokenizer_name: tokenizer_name from the transformers library
        seq_lenght: window lenght you wanna output
    """

    def __init__(self, json_file, tokenizer_name,seq_lenght=512):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.seq_lenght = seq_lenght #depending on what model we use we might wanna change it. This is for bert

        with open(json_file, encoding="utf-8") as f:
            self.data = json.load(f)
        
        #self.uniq_tags <- result here
        self.extract_unique_tags() #process dataset to extract unique tags
        #we'll add a tag for parts of the text without any tag.
        self.uniq_tags.append("NONE")

        #create tag sequence for tokens sequence.
        self.generate_tokensANDtags()

        #convert tokens to their respective index
        self.tokens_idx = self.tokenizer.convert_tokens_to_ids(self.tokens)

        #conver tags to indexs
        self.tags_idxs = self.tags2index(self.tags)

        #generate and tokenize <pad> token.
        self.pad_token = self.tokenizer.tokenize("[PAD]")[0]
        self.pad_token_idx = self.tokenizer.convert_tokens_to_ids(self.pad_token)

        

    def generate_tokensANDtags(self):
        pre_tokenizer = Whitespace()

        #pre-tokenize all text:
        self.text = ""
        self.tokens = []
        self.tags = []
        self.accumulated_idx = [] #keep track of in what text index each token ends
        for report in self.data: #iterate reports
            
            #generate text tokens
            txt = "[CLS] "+report["data"]["text"]+" [SEP]"
            pre_tkz = pre_tokenizer.pre_tokenize_str(txt)
            tokens_full = [(self.tokenizer.tokenize(tk[0]),tk[1])for tk in pre_tkz]
            tokens = []
            for tk in tokens_full:
                tokens += tk[0]

            #append text and tokens
            self.text += txt
            self.tokens += tokens

            #generate tags
            tags = np.array(["NONE"]*len(tokens))
            tk_index = 0
            for tagged_range in report["predictions"][0]["result"]:
                #Note, we add 5 because the added [CLS] token shifts the indexes in respect to the originals.
                start_idx = tagged_range["value"]["start"] + 5
                end_idx = tagged_range["value"]["end"] + 5
                
                for tkns,idxs in tokens_full:
                    tok_start = idxs[0]
                    tok_end = idxs[1]
 
                    if(tok_start >= start_idx) and (tok_end<= end_idx):
                        tags[tk_index:tk_index+len(tkns)] = tagged_range["value"]["labels"][0]

                    tk_index += len(tkns)
            self.tags += list(tags)
        
        self.accumulated_idx = self.accumulated_idx[1:] #remove auxiliary initial 0

    def tags2index(self,label_seq:list)->list:
        return [self.uniq_tags.index(lab) for lab in label_seq]


    def extract_unique_tags(self):
        self.uniq_tags = set()
        for report in self.data:
            for tagged_range in report["predictions"][0]["result"]:
                #we checked and this array for all reports only contains one element one label per tagged range
                self.uniq_tags.add(tagged_range["value"]["labels"][0])

        self.uniq_tags = list(self.uniq_tags)
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx): 
        idx_2 = idx + self.seq_lenght     
        if idx_2 >= len(self.tokens): #in case we reach end of dataset and doesn't match seq lenght
            idx_2 = len(self.tokens)
            pad_lenght = (self.seq_lenght - (idx_2-idx))
            pad_tok = [self.pad_token]*pad_lenght
            pad_tok_idx = self.tokenizer.convert_tokens_to_ids(pad_tok)

            pad_label = ["NONE"]*pad_lenght
            pad_idxs = self.tags2index(pad_label)
            data = {"x":self.tokens_idx[idx:idx_2]+pad_tok_idx, "x_ref":self.tokens[idx:idx_2]+pad_tok, "y":self.tags_idxs[idx:idx_2]+pad_idxs}
        else:
            data = {"x":torch.tensor(self.tokens_idx[idx:idx_2]), "x_ref":self.tokens[idx:idx_2], "y":torch.tensor(self.tags_idxs[idx:idx_2])}
        
        return data


