import torch
from torch.utils.data import Dataset
import random


class MsrPCDataset(Dataset):

    def __init__(self, split, test=False, tokenizer=None):

        self.train_fname = 'data/msr-paraphrase-corpus/msr_paraphrase_train.txt'
        self.test_fname = 'data/msr-paraphrase-corpus/msr_paraphrase_test.txt'

        self.fname = self.test_fname if test else self.train_fname

        # Read data from file
        self.sentences_dict = {0:[], 1:[]}
        self.sentence = []
        self.match     = []
        self.readfile(self.fname) # Updates the above lists
        self.split_data()

        # Tokenize if asked
        if tokenizer is not None:
            self.sentence = tokenizer(
                self.sentence,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

        L = len(self.match)
        if split == 'train':
            idx_start = 0
            idx_end = idx_start + int(0.7 * L)
        elif split == 'val':
            idx_start = int(0.7 * L)
            idx_end = idx_start + int(0.15 * L)
        elif split == 'test':
            idx_start = int(0.85 * L)
            idx_end = L

        self.sentence = self.sentence[idx_start:idx_end]
        self.match = self.match[idx_start:idx_end]


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

                self.sentence.append(s1 + ' ' + s2)
                self.match.append(m)


    def split_data(self):
        random.seed(0)

        required_len = min(len(self.sentences_dict[0]), len(self.sentences_dict[1]))
        # randomly pick required_len from both the sets
        combined_list = []
        combined_list.extend(random.sample(self.sentences_dict[0], required_len))
        combined_list.extend(random.sample(self.sentences_dict[1], required_len))
        random.shuffle(combined_list)
        for sent_set in combined_list:
            self.sentence.append(sent_set[0])
            self.sentence.append(sent_set[1])
            self.match.append(sent_set[2])

    def __len__(self):
        return len(self.match)

    def __getitem__(self, idx):
        x = self.sentence[idx]
        y = torch.tensor(self.match[idx]).unsqueeze(-1)

        return [x, y]
