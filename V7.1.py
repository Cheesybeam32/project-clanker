import os
import torch
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")
Vocab = sp.vocab_size()

LR = 0.001
Random = 0.01

Dims = 256
Heads = 8
Head_Dims = Dims // Heads
FFN_Dims = Dims * 4
FFN_Layers = 6

torch.set_default_device("cpu")
torch.set_num_threads(8)
torch.set_num_interop_threads(2)
def load():
    while True:
        with open('dataset.txt', 'r', encoding='utf-8') as data:
            for line in data:
                miku = sp.encode(line.strip(), out_type=int)
                yield miku

class state_input(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
        self.k = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
        self.v = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
        self.w = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
    def forward(self, state, input):
        Q = torch.matmul(input, self.q)
        K = torch.matmul(state, self.k)
        V = torch.matmul(state, self.v)
        H_Q = Q.view(Heads, Head_Dims)
        H_K = K.view(Heads, Head_Dims)
        H_V = V.view(Heads, Head_Dims)
        scores = torch.matmul(H_Q, H_K.T)
        scores = scores / (Head_Dims ** 0.5)
        scores = torch.softmax(scores, dim=-1)
        KV = torch.matmul(scores, H_V) 
        thingus = KV.view(Dims)
        output = torch.matmul(thingus, self.w) + state
        return output
class ffn_layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w_1 = torch.nn.Parameter(torch.empty(Dims, FFN_Dims).uniform_(-Random, Random))
        self.w_2 = torch.nn.Parameter(torch.empty(FFN_Dims, Dims).uniform_(-Random, Random))
    def forward(self, output):
        hidden = torch.matmul(output, self.w_1)
        n_hidden = torch.relu(hidden)
        ffn = torch.matmul(n_hidden, self.w_2)
        ffn_out = ffn + output
        return ffn_out
class brain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.empty(Vocab, Dims).uniform_(-Random, Random))
        self.attn_layers = torch.nn.ModuleList([state_input() for i in range(FFN_Layers)])
        self.ffn_layers = torch.nn.ModuleList([ffn_layer() for i in range(FFN_Layers)])
        self.out = torch.nn.Parameter(torch.empty(Dims, Vocab).uniform_(-Random, Random))
        self.add = torch.nn.Parameter(torch.empty(Dims).uniform_(-Random, Random))
        self.forget = torch.nn.Parameter(torch.empty(Dims).uniform_(-Random, Random))
    def forward(self, state, pred_miku):
        output = self.embed[pred_miku]
        for i in range(FFN_Layers):
            output = self.attn_layers[i](state, output)
            output = self.ffn_layers[i](output)
        state = state * torch.sigmoid(self.forget)
        state += output * torch.sigmoid(self.add)
        output = torch.matmul(output, self.out)
        return output, state
def main():
    data = load()
    model = brain()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    while True:
        miku = next(data)
        if len(miku) <= 2:
            continue
        total_loss = 0
        first_token = miku[0]
        state = model.embed[first_token].clone()
        optimizer.zero_grad()
        for token in range(1, len(miku)):
            pred_miku = miku[token - 1]
            target_miku = miku[token]
            output, state = model(state, pred_miku)
            target_miku = torch.tensor(target_miku)
            loss = torch.nn.functional.cross_entropy(output, target_miku)
            print(loss.item())
            total_loss += loss
        total_loss.backward()
        optimizer.step()
main()