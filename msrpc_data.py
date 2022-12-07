import torch
from torch.utils.data import Dataset
import random

class MsrPCDataset(Dataset):

    def __init__(self, split, augment=True):
        '''
        split: 'test', 'train', or 'val'
        '''

        self.train_fname = 'data/msr-paraphrase-corpus/msr_paraphrase_train.txt'
        self.test_fname = 'data/msr-paraphrase-corpus/msr_paraphrase_test.txt'

        # Read data from file
        self.sentences_dict = {0:[], 1:[]}
        self.readfile(self.train_fname) # Updates the above lists
        self.readfile(self.test_fname) # Updates the above lists

        # Clean data
        self.sentence1 = []
        self.sentence2 = []
        self.match = []

        self.split_data()

        if augment:
            self.augment_data()

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

        self.sentence1 = self.sentence1[idx_start:idx_end]
        self.sentence2 = self.sentence2[idx_start:idx_end]
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
                self.sentences_dict[m].append([s1, s2, m])

    def split_data(self):
        random.seed(0)

        required_len = min(len(self.sentences_dict[0]), len(self.sentences_dict[1]))
        # randomly pick required_len from both the sets
        combined_list = []
        combined_list.extend(random.sample(self.sentences_dict[0], required_len))
        combined_list.extend(random.sample(self.sentences_dict[1], required_len))
        random.shuffle(combined_list)
        for sent_set in combined_list:
            self.sentence1.append(sent_set[0])
            self.sentence2.append(sent_set[1])
            self.match.append(sent_set[2])


    def augment_data(self):
        L = self.__len__()

        aug1 = [self.sentence1, self.sentence1, [1]*L]
        aug2 = [self.sentence2, self.sentence2, [1]*L]
        aug3 = [self.sentence1[:-1], self.sentence1[1:], [0]*(L-1)]
        aug4 = [self.sentence2[:-1], self.sentence2[1:], [0]*(L-1)]

        self.sentence1 = self.sentence1 + aug1[0] + aug2[0] + aug3[0] + aug4[0]
        self.sentence2 = self.sentence2 + aug1[1] + aug2[1] + aug3[1] + aug4[1]
        self.match = self.match + aug1[2] + aug2[2] + aug3[2] + aug4[2]

        # Shuffle
        z = list(zip(self.sentence1, self.sentence2, self.match))
        random.shuffle(z)
        self.sentence1, self.sentence2, self.match = zip(*z)

        self.sentence1 = list(self.sentence1)
        self.sentence2 = list(self.sentence2)
        self.match = list(self.match)



    def __len__(self):
        return len(self.match)

    def __getitem__(self, idx):
        x = [self.sentence1[idx], self.sentence2[idx]]
        y = torch.tensor(self.match[idx]).unsqueeze(-1)

        return x, y
