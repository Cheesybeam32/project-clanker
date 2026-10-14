import os
import torch
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")
S_vocab = sp.vocab_size()

Cells = 8
Dims = 16
Bounds = 512
LR = 0.001
Threshold = 0
H_random = 0.01
L_random = -0.01
Gpu = True

if Gpu:
    torch.set_default_device("xpu")
else:
    torch.set_default_device("cpu")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
def load():
    while True:
        with open('dataset.txt', 'r', encoding='utf-8') as data:
            for line in data:
                miku = sp.encode(line.strip(), out_type=int)
                yield miku

class Brain(torch.nn.Module):
    def init(self):
        self.V_embed = torch.nn.Parameter(torch.empty(S_vocab, Dims).uniform_(L_random, H_random))
        self.Input = torch.nn.Parameter(torch.empty(Cells, Dims).uniform_(L_random, H_random))
        self.Cells = torch.zeros(Cells, Dims)
        self.Weights = torch.nn.Parameter(torch.empty(Bounds, Dims).uniform_(L_random, H_random))
        self.D_Mixer = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(L_random, H_random))
        self.Add = torch.nn.Parameter(torch.empty(Cells, Dims).uniform_(L_random, H_random))
        self.C_Mixer = torch.nn.Parameter(torch.empty(Cells).uniform_(L_random, H_random))
        self.Forget = torch.nn.Parameter(torch.empty(Cells, Dims).uniform_(L_random, H_random))
    def reset(self):
        self.Cells = torch.zeros(Cells, Dims)
    def input(self, pred_miku):
        p_embed = self.V_embed[pred_miku]
        embed = p_embed[None, :]
        input_gate = torch.sigmoid(self.Input)
        cell_add = input_gate * embed
        self.Cells += cell_add
    def main_grid(self):
        cells = self.Cells[:, None, :, None].clone()
        weights = self.Weights[None, :, None, :]
        mixer = self.D_Mixer[None, None, :, :]
        interaction = cells * mixer
        interaction = interaction * weights
        scores = torch.sum(interaction, dim=(2, 3))
        selection = torch.softmax(scores, dim=1)
        weighted = selection[:, :, None] * self.Weights[None, :, :]
        output = torch.sum(weighted, dim=1)
        return output
    def cell_upd(self, output):
        forget_gate = torch.sigmoid(self.Forget)
        cell_mix = torch.sigmoid(self.C_Mixer)
        add_mix = torch.sigmoid(self.Add)
        self.Cells *= forget_gate
        added = output * add_mix
        self.Cells += added
        c_multiplied = output * cell_mix[:, None]
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