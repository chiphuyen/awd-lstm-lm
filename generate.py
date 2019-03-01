###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import os

import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='/home/chipn/data/childbooks',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model_400_400_512_2715.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='10000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=0.5,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=200,
                    help='reporting interval')
args = parser.parse_args()

output_folder = 'generated'
os.makedirs(output_folder, exist_ok=True)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)[0].to(device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
# input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

beginners = "bunny brown and his sister sue and their shetland pony <eos> by <eos>".split()
print(beginners)
outname = output_folder + '/' + '_'.join(beginners) + args.checkpoint[2:] + '.txt'

# input = torch.tensor([[2660]]).to(device)

# print('input', corpus.dictionary.idx2word[input[0][0]])

with open(outname, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i, word in enumerate(beginners):
            outf.write('\n' if word == '<eos>' else word + ' ')
            if not word in corpus.dictionary.word2idx:
                word = '<unk>'
            input = torch.tensor([[corpus.dictionary.word2idx[word]]]).to(device)
            
            if i != len(beginners) - 1:
                _, hidden = model(input, hidden)

        for i in range(args.words):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]
            if word == '<eob>':
                break

            outf.write('\n' if word == '<eos>' else word + ' ')

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
