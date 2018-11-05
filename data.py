# import os
# import torch

# from collections import Counter


# class Dictionary(object):
#     def __init__(self):
#         self.word2idx = {}
#         self.idx2word = []
#         self.counter = Counter()
#         self.total = 0

#     def add_word(self, word):
#         if word not in self.word2idx:
#             self.idx2word.append(word)
#             self.word2idx[word] = len(self.idx2word) - 1
#         token_id = self.word2idx[word]
#         self.counter[token_id] += 1
#         self.total += 1
#         return self.word2idx[word]

#     def __len__(self):
#         return len(self.idx2word)


# class Corpus(object):
#     def __init__(self, path):
#         self.dictionary = Dictionary()
#         self.train = self.tokenize(os.path.join(path, 'train.txt'))
#         self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
#         self.test = self.tokenize(os.path.join(path, 'test.txt'))

#     def tokenize(self, path):
#         """Tokenizes a text file."""
#         assert os.path.exists(path)
#         # Add words to the dictionary
#         with open(path, 'r') as f:
#             tokens = 0
#             for line in f:
#                 words = line.split() + ['<eos>']
#                 tokens += len(words)
#                 for word in words:
#                     self.dictionary.add_word(word)

#         # Tokenize file content
#         with open(path, 'r') as f:
#             ids = torch.LongTensor(tokens)
#             token = 0
#             for line in f:
#                 words = line.split() + ['<eos>']
#                 for word in words:
#                     ids[token] = self.dictionary.word2idx[word]
#                     token += 1

#         return ids

from collections import Counter
import os

import torch


class Dictionary(object):
    def __init__(self): # do we need limit?
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.UNK = '<unk>'
        self.EOS = '<eos>'
        self.EOB = '<eob>'
        # if vocab_link and os.path.isfile(vocab_link):
        #     self.load_vocab(vocab_link)

        for word in [self.UNK, self.EOS, self.EOB]:
            if not word in self.word2idx:
                self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        return self.word2idx[word]

    def load_vocab(self, vocab_link):
        vocab_file = open(vocab_link, 'r')
        lines = vocab_file.readlines()
        n = int(lines[-1].strip())
        self.idx2word = [0 for _ in range(n)]
        for line in lines[:-1]:
            parts = line.strip().split('\t')
            token_id, word, count = int(parts[0]), parts[1], int(parts[2]) 
            self.word2idx[word] = token_id
            self.idx2word[token_id] = word
            self.counter[token_id] = count

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, raw_path, proc_path='processed'):
        os.makedirs(proc_path, exist_ok=True)
        self.dictionary = Dictionary()
        self.DIVIDER = '==================================================================================='
        self.vocab_link = 'vocab.txt'

        exists = self.check_exist(proc_path)

        if not exists:
            print('Creating corpus from raw data ...')
            if not raw_path:
                raise ValueError("data_root [directory to the original data] must be specified")
            self.create_dictionary(proc_path, os.path.join(raw_path, 'train.txt'))
            self.train = self.tokenize(raw_path, proc_path, 'train.txt')
            self.valid = self.tokenize(raw_path, proc_path, 'valid.txt')
            self.test = self.tokenize(raw_path, proc_path, 'test.txt')
        else:
            self.load_corpus(proc_path)

    def check_exist(self, proc_path):
        paths = [proc_path, proc_path + '/vocab.txt', proc_path + '/train.ids', 
                proc_path + '/valid.ids', proc_path + '/test.ids']
        for name in paths:
            if not os.path.exists(name):
                return False
        return True

    def create_dictionary(self, proc_path, filename):
        '''
        Add words to the dictionary only if it's train file
        '''
        with open(filename, 'r') as f:
            f.readline()
            for line in f:
                line = line.strip()
                if not line or line == self.DIVIDER:
                    continue
                words = line.split() + [self.dictionary.EOS]
                for word in words:
                    self.dictionary.add_word(word)

        with open(os.path.join(proc_path, self.vocab_link), 'w') as f:
            f.write(str(len(self.dictionary)) + '\n')
            for token_id, count in self.dictionary.counter.most_common():
                f.write('\t'.join([str(token_id), 
                            self.dictionary.idx2word[token_id], 
                            str(count)]) + '\n')


    def tokenize(self, raw_path, proc_path, filename):
        unk_id = self.dictionary.word2idx[self.dictionary.UNK]
        out = open(os.path.join(proc_path, filename[:-3] + 'ids'), 'w')
        with open(os.path.join(raw_path, filename), 'r') as f:
            ids = []
            for line in f:
                line = line.strip()
                if line.strip() == self.DIVIDER:
                    words = [self.dictionary.EOB]
                else:
                    words = line.split() + [self.dictionary.EOS]
                for word in words:
                    ids.append(self.dictionary.word2idx.get(word, unk_id))
        out.write(self.list2str(ids))
        out.close()
        return torch.LongTensor(ids)
        # return np.asarray(ids)

    def load_ids(self, filename):
        ids = open(filename, 'r').read().strip().split('\t')
        return torch.LongTensor([int(i) for i in ids])
        # return np.asarray([int(i) for i in ids])

    def list2str(self, list):
        return '\t'.join([str(num) for num in list])

    def load_corpus(self, proc_path):
        print('Loading corpus from processed data ...')
        vocab_file = open(os.path.join(proc_path, self.vocab_link), 'r')
        n = int(vocab_file.readline().strip())
        self.dictionary.idx2word = [0 for _ in range(n)]
        for line in vocab_file:
            parts = line.strip().split('\t')
            token_id, word, count = int(parts[0]), parts[1], int(parts[2]) 
            self.dictionary.word2idx[word] = token_id
            self.dictionary.idx2word[token_id] = word
            self.dictionary.counter[token_id] = count
        self.train = self.load_ids(os.path.join(proc_path, 'train.ids'))
        self.valid = self.load_ids(os.path.join(proc_path, 'valid.ids'))
        self.test = self.load_ids(os.path.join(proc_path, 'test.ids'))