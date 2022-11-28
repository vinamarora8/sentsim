import torch
from torch.utils.data import Dataset
import random

class ParaphraseDataset(Dataset):

    def __init__(self, split, mode, N=None):
        '''
        split: 'test', 'train', or 'val'
        mode: 'msrpc, tapaco, all'
        '''

        self.sentences_dict = {0:[], 1:[]}

        # Read data from file
        if mode == "msrpc" or mode == "all":
            fname = 'data/msr-paraphrase-corpus/msr_paraphrase_train.txt'
            self.readfile(fname, 0, 3, 4)
            fname = 'data/msr-paraphrase-corpus/msr_paraphrase_test.txt'
            self.readfile(fname, 0, 3, 4)

        if mode == "tapaco" or mode == "all":
            fname = 'data/tapaco/out_file.tsv'
            self.readfile(fname, 2, 0, 1)

        # Clean data
        self.sentence1 = []
        self.sentence2 = []
        self.match = []
        self.split_data()

        if N is not None:
            L = N
        else:
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


    def readfile(self, fname, m_idx, s1_idx, s2_idx):
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
                m = int(split_line[m_idx])
                s1 = split_line[s1_idx]
                s2 = split_line[s2_idx]
                self.sentences_dict[m].append([s1, s2, m])

    # Splits data to have equal number of paraphrases and non paraphrases
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

    def __len__(self):
        return len(self.match)

    def __getitem__(self, idx):
        x = [self.sentence1[idx], self.sentence2[idx]]
        y = torch.tensor(self.match[idx]).unsqueeze(-1)

        return x, y
