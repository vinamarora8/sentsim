import torch
from torch.utils.data import Dataset

class MsrPCDataset(Dataset):

    def __init__(self, test=False, tokenizer=None):

        self.train_fname = 'data/msr-paraphrase-corpus/msr_paraphrase_train.txt'
        self.test_fname = 'data/msr-paraphrase-corpus/msr_paraphrase_test.txt'

        self.fname = self.test_fname if test else self.train_fname

        # Read data from file
        self.sentence1 = []
        self.sentence2 = []
        self.match     = []
        self.readfile(self.fname) # Updates the above lists

        # Tokenize if asked
        if tokenizer is not None:
            self.sentence1 = tokenizer(
                self.sentence1,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            self.sentence2 = tokenizer(
                self.sentence2,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

    def readfile(self, fname):
        '''
        File format: Like CSV, \t as separator
        MATCH, ID1, ID2, SENTENCE1, SENTENCE2
        Contains a header
        '''
        with open(fname, 'r') as f:
            f.readline() # Gets rid of header
            for line in f:
                line = line.strip()
                split_line = line.split('\t')
                m = int(split_line[0])
                s1 = split_line[3]
                s2 = split_line[4]

                self.sentence1.append(s1)
                self.sentence2.append(s2)
                self.match.append(m)

    def __len__(self):
        return len(self.match)

    def __getitem__(self, idx):
        x = [self.sentence1[idx], self.sentence2[idx]]
        y = self.match[idx]

        return x, y
