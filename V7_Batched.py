import os
import torch
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")
Vocab = sp.vocab_size()
Pad = sp.pad_id()

LR = 0.001
Random = 0.01
Batch = 64

Dims = 256
Heads = 8
Head_Dims = Dims // Heads
FFN_Dims = Dims * 4
FFN_Layers = 6

torch.set_default_device("cuda:1")
torch.set_num_threads(18)
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
        self.q = torch.nn.Parameter(torch.empty(Heads, Head_Dims).uniform_(-Random, Random))
        self.k = torch.nn.Parameter(torch.empty(Heads, Head_Dims).uniform_(-Random, Random))
        self.v = torch.nn.Parameter(torch.empty(Heads, Head_Dims).uniform_(-Random, Random))
        self.w = torch.nn.Parameter(torch.empty(Head_Dims, Head_Dims).uniform_(-Random, Random))
    def forward(self, state, input):
        split_state = state.view(Batch, Heads, Head_Dims)
        split_input = input.view(Batch, Heads, Head_Dims)
        q = split_input[:, :, :, None] * self.q[None, :, None, :]
        k = split_state[:, :, :, None] * self.k[None, :, None, :]
        v = split_state[:, :, :, None] * self.v[None, :, None, :]
        Q = torch.layer_norm(q, (Head_Dims, Head_Dims))
        K = torch.layer_norm(k, (Head_Dims, Head_Dims))
        V = torch.layer_norm(v, (Head_Dims, Head_Dims))
        score = torch.matmul(Q, K) / (Head_Dims ** 0.5)
        score = torch.softmax(score, dim=-1)
        result = torch.matmul(score, V)
        p_output = result * self.w[None, None, :, :]
        p_2_output = torch.sum(p_output, dim=3)
        output = p_2_output.reshape(Batch, Dims)
        return output
class ffn_layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w_1 = torch.nn.Parameter(torch.empty(Dims, FFN_Dims).uniform_(-Random, Random))
        self.w_2 = torch.nn.Parameter(torch.empty(FFN_Dims, Dims).uniform_(-Random, Random))
    def forward(self, output):
        x = torch.layer_norm(output, (Dims,))
        hidden = torch.matmul(x, self.w_1)
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
        state = state + output * (1 - torch.sigmoid(self.add))
        output = torch.matmul(output, self.out)
        valid = (pred_miku != Pad).unsqueeze(1)
        state = torch.where(valid, state, state.detach())
        return output, state
def main():
    data = load()
    model = brain()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    p_error = None
    while True:
        inputs = []
        while len(inputs) < Batch:
            miku = next(data)
            if len(miku) <= 2:
                continue
            inputs.append(miku)
        max_len = max(len(x) for x in inputs)
        inputs = [
            x + [Pad] * (max_len - len(x))
            for x in inputs
        ]
        inputs = torch.tensor(inputs, dtype=torch.long)
        total_loss = 0
        state = torch.ones(Batch, Dims)
        optimizer.zero_grad()
        for token in range(max_len - 1):
            pred_miku = inputs[:, token]
            target_miku = inputs[:, token + 1]
            output, state = model(state, pred_miku)
            loss = torch.nn.functional.cross_entropy(output, target_miku, ignore_index=Pad)
            error = loss.item()
            if p_error == None:
                p_error = loss.item()
            else:
                p_error = (p_error * 0.9) + (error * 0.1)
            print(f"{p_error}, {error}")
            total_loss += loss
        total_loss.backward()
        optimizer.step()
main()