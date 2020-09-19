import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext

import spacy
import numpy as np
import random
import math
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

''' 1. Prepare Data '''

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# src:en
spacy_en = spacy.load('en')
# trg:zh
spacy_zh = spacy.load('zh')

# tokenize
def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_zh(text):
    """
    Tokenizes Chinese text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_zh.tokenizer(text)]


SRC = torchtext.data.Field(tokenize = tokenize_en,
                           init_token = '<sos>',
                           eos_token = '<eos>',
                           lower = True)

TRG = torchtext.data.Field(tokenize = tokenize_zh,
                           init_token = '<sos>',
                           eos_token = '<eos>',
                           lower = True)

train_data, valid_data, test_data = torchtext.data.TabularDataset.splits(
            path='./data/',
            train='train.tsv',
            validation='valid.tsv',
            test='test.tsv',
            format='tsv',
            fields=[('src', SRC), ('trg', TRG)])

print("Number of training examples: " + str(len(train_data.examples)))
print("Number of validation examples: " + str(len(valid_data.examples)))
print("Number of testing examples: " + str(len(test_data.examples)))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

print("Unique tokens in source (en) vocabulary: " + str(len(SRC.vocab)))
print("Unique tokens in target (zh) vocabulary: " + str(len(TRG.vocab)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 16

train_iterator, valid_iterator, test_iterator = torchtext.data.BucketIterator.splits(
                                                (train_data, valid_data, test_data),
                                                batch_size = BATCH_SIZE,
                                                device = device,
                                                sort = False)


''' 2. Building the Seq2Seq Model '''

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, device, max_length = 100):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd!"

        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim,
                                              out_channels = 2 * hid_dim,
                                              kernel_size = kernel_size,
                                              padding = (kernel_size - 1) // 2)
                                              for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        #src = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        #create position tensor
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [0, 1, 2, 3, ..., src len - 1]

        #pos = [batch size, src len]

        #embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)

        #tok_embedded = pos_embedded = [batch size, src len, emb dim]

        #combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)

        #embedded = [batch size, src len, emb dim]

        #pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)

        #conv_input = [batch size, src len, hid dim]

        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)

        #conv_input = [batch size, hid dim, src len]

        #begin convolutional blocks...

        for i, conv in enumerate(self.convs):

            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            #conved = [batch size, 2 * hid dim, src len]

            #pass through GLU activation function
            conved = F.glu(conved, dim = 1)

            #conved = [batch size, hid dim, src len]

            #apply residual connection
            conved = (conved + conv_input) * self.scale

            #conved = [batch size, hid dim, src len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        #...end convolutional blocks

        #permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))

        #conved = [batch size, src len, emb dim]

        #elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale

        #combined = [batch size, src len, emb dim]

        return conved, combined


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, trg_pad_idx, device, max_length = 100):
        super().__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim, output_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim,
                                              out_channels = 2 * hid_dim,
                                              kernel_size = kernel_size)
                                              for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):

        #embedded = [batch size, trg len, emb dim]
        #conved = [batch size, hid dim, trg len]
        #encoder_conved = encoder_combined = [batch size, src len, emb dim]

        #permute and convert back to emb dim
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))

        #conved_emb = [batch size, trg len, emb dim]

        combined = (conved_emb + embedded) * self.scale

        #combined = [batch size, trg len, emb dim]

        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))

        #energy = [batch size, trg len, src len]

        attention = F.softmax(energy, dim=2)

        #attention = [batch size, trg len, src len]

        attended_encoding = torch.matmul(attention, encoder_combined)

        #attended_encoding = [batch size, trg len, emd dim]

        #convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)

        #attended_encoding = [batch size, trg len, hid dim]

        #apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale

        #attended_combined = [batch size, hid dim, trg len]

        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):

        #trg = [batch size, trg len]
        #encoder_conved = encoder_combined = [batch size, src len, emb dim]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        #create position tensor
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [batch size, trg len]

        #embed tokens and positions
        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)

        #tok_embedded = [batch size, trg len, emb dim]
        #pos_embedded = [batch size, trg len, emb dim]

        #combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)

        #embedded = [batch size, trg len, emb dim]

        #pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)

        #conv_input = [batch size, trg len, hid dim]

        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)

        #conv_input = [batch size, hid dim, trg len]

        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]

        for i, conv in enumerate(self.convs):

            #apply dropout
            conv_input = self.dropout(conv_input)

            #need to pad so decoder can't "cheat"
            padding = torch.zeros(batch_size,
                                  hid_dim,
                                  self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)

            padded_conv_input = torch.cat((padding, conv_input), dim = 2)

            #padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]

            #pass through convolutional layer
            conved = conv(padded_conv_input)

            #conved = [batch size, 2 * hid dim, trg len]

            #pass through GLU activation function
            conved = F.glu(conved, dim = 1)

            #conved = [batch size, hid dim, trg len]

            #calculate attention
            attention, conved = self.calculate_attention(embedded,
                                                         conved,
                                                         encoder_conved,
                                                         encoder_combined)

            #attention = [batch size, trg len, src len]

            #apply residual connection
            conved = (conved + conv_input) * self.scale

            #conved = [batch size, hid dim, trg len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))

        #conved = [batch size, trg len, emb dim]

        output = self.fc_out(self.dropout(conved))

        #output = [batch size, trg len, output dim]

        return output, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):

        #src = [batch size, src len]
        #trg = [batch size, trg len - 1] (<eos> token sliced off the end)

        #calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        #encoder_conved is output from final encoder conv. block
        #encoder_combined is encoder_conved plus (elementwise) src embedding plus
        #  positional embeddings
        encoder_conved, encoder_combined = self.encoder(src)

        #encoder_conved = [batch size, src len, emb dim]
        #encoder_combined = [batch size, src len, emb dim]

        #calculate predictions of next words
        #output is a batch of predictions for each word in the trg sentence
        #attention a batch of attention scores across the src sentence for
        #  each word in the trg sentence
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)

        #output = [batch size, trg len - 1, output dim]
        #attention = [batch size, trg len - 1, src len]

        return output, attention


''' 3. Training the Seq2Seq Model '''

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
EMB_DIM = 256
HID_DIM = 512 # each conv. layer has 2 * hid_dim filters
ENC_LAYERS = 10 # number of conv. blocks in encoder
DEC_LAYERS = 10 # number of conv. blocks in decoder
ENC_KERNEL_SIZE = 3 # must be odd!
DEC_KERNEL_SIZE = 3 # can be even or odd
ENC_DROPOUT = 0.25
DEC_DROPOUT = 0.25
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)

model = Seq2Seq(enc, dec)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]) # consistent with cuda

model = model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('The model has ' + str(count_parameters(model)) + ' trainable parameters')

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output, _ = model(src, trg[:,:-1])

        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)

        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:,:-1])

            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)

            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



N_EPOCHS = 10
CLIP = 0.1

for epoch in range(N_EPOCHS):
    savedFiles = os.listdir('./savedModel/4')
    best_valid_loss = float(savedFiles[0][8:-3]) if savedFiles else float('inf')
    if savedFiles:
        model.load_state_dict(torch.load('./savedModel/4/' + savedFiles[0]))

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        savedFiles = os.listdir('./savedModel/4')
        torch.save(model.state_dict(), './savedModel/4/4-model_%f.pt' % valid_loss)
        if savedFiles:
            os.remove('./savedModel/4/' + savedFiles[0])

    print('Epoch: %d | Time: %d m %d s' % (epoch + 1, epoch_mins, epoch_secs))
    print('\tTrain Loss: %f | Train PPL: %f' % (train_loss, math.exp(train_loss)))
    print('\t Valid Loss: %f |  Valid PPL: %f' % (valid_loss, math.exp(valid_loss)))

savedFiles = os.listdir('./savedModel/4')
model.load_state_dict(torch.load('./savedModel/4/' + savedFiles[0]))

test_loss = evaluate(model, test_iterator, criterion)

print('| Test Loss: %f | Test PPL: %f |' % (test_loss, math.exp(test_loss)))
