import os

import torch
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")
S_vocab = sp.vocab_size()

Cells = 8
Dims = 16
Bounds = 1024
LR = 0.001

torch.set_num_threads(6)
torch.set_num_interop_threads(2)
def load():
    while True:
        with open('dataset.txt', 'r', encoding='utf-8') as data:
            for line in data:
                miku = sp.encode(line.strip(), out_type=int)
                yield miku

class Brain(torch.nn.Module):
    def init(self):
        self.V_embed = torch.nn.Parameter(torch.empty(S_vocab, Dims).uniform_(-0.01, 0.01))
        self.Input = torch.nn.Parameter(torch.empty(Cells, Dims).uniform_(-0.01, 0.01))
        self.Cells = torch.zeros(Cells, Dims)
        self.Weights = torch.nn.Parameter(torch.empty(Bounds, Dims).uniform_(-0.01, 0.01))
        self.D_Mixer = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-0.01, 0.01))
        self.Add = torch.nn.Parameter(torch.empty(Cells, Dims).uniform_(-0.01, 0.01))
        self.C_Mixer = torch.nn.Parameter(torch.empty(Cells).uniform_(-0.01, 0.01))
        self.Forget = torch.nn.Parameter(torch.empty(Cells, Dims).uniform_(-0.01, 0.01))
    def reset(self):
        self.Cells = torch.zeros(Cells, Dims)
    def input(self, pred_miku):
        p_embed = self.V_embed[pred_miku]
        embed = p_embed[None, :]
        input_gate = torch.sigmoid(self.Input)
        cell_add = input_gate * embed
        self.Cells += cell_add
    def main_grid(self):
        cell_expand = self.Cells[:, None, :].clone()
        weight_expand = self.Weights[None, :, :]
        multipled = cell_expand * weight_expand
        dot = torch.sum(multipled, dim=2)
        #soft_dot = torch.softmax(dot, dim=1)
        #sig_dot = torch.sigmoid(dot)
        #relu_dot = torch.relu(dot)
        output = torch.matmul(dot, self.Weights)
        return output
    def cell_upd(self, output):
        forget_gate = torch.sigmoid(self.Forget)
        cell_mix = torch.sigmoid(self.C_Mixer)
        dim_mix = torch.sigmoid(self.D_Mixer)
        add_mix = torch.sigmoid(self.Add)
        d_mixed = torch.zeros(Cells, Dims)
        self.Cells *= forget_gate
        d_mixed = torch.matmul(output, dim_mix.T)
        added = d_mixed * add_mix
        self.Cells += (1 - added)
        c_multiplied = d_mixed * cell_mix[:, None]
        c_mixed = torch.sum(c_multiplied, dim=0)
        return c_mixed
    def loss(self, c_mixed, target_miku):
        logits = torch.matmul(c_mixed, self.V_embed.T)
        loss = torch.nn.functional.cross_entropy(logits[None, :], torch.tensor([target_miku]))
        return loss
def main_loop():
    brain = Brain()
    data = load()
    brain.init()
    optimizer = torch.optim.SGD(brain.parameters(), lr=LR)
    params = list(brain.parameters())
    step = 0
    print(f"Number of tracked parameter tensors: {len(params)}")
    for i, p in enumerate(params):
        print(f"Tensor {i} shape: {list(p.shape)}")
    while True:
        miku = next(data)
        if len(miku) <= 1:
            continue
        brain.reset()
        total_loss = 0
        optimizer.zero_grad()
        for token in range(len(miku) - 1):
            step += 1
            pred_miku = miku[token]
            target_miku = miku[token + 1]
            brain.input(pred_miku)
            output = brain.main_grid()
            c_mixed = brain.cell_upd(output)
            loss = brain.loss(c_mixed, target_miku)
            total_loss += loss
            if step == 1:
                p_error = loss.item()
            else:
                p_error = (p_error * 0.9) + (loss.item() * 0.1)
            print(f"{p_error}, {loss.item()}")
        total_loss.backward()
        optimizer.step()
main_loop()