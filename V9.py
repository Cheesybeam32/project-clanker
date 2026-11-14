import os
import torch
import sentencepiece as spm
import time

sp = spm.SentencePieceProcessor()
sp.load("spm.model")
Vocab = sp.vocab_size()
torch.set_default_device("cuda:0")
torch.set_default_dtype(torch.float32)
torch.set_num_threads(8)
torch.set_num_interop_threads(2)
torch.autograd.set_detect_anomaly(True, check_nan=False)

Dims = 1300

Random = 0.01
LR = 0.0001

""" 
Matmul symbol is ^*
Just basic recurrent, add GRU structure later
Ignoring heads and other fancy stuff
A = Dims x Dims matrix (decides how much old state to carry)
B = Dims x Dims matrix (decides how much of new token to add)
(A gate) = sigmoid(A ^* (H-1))
(B gate) = sigmoid(B ^* T) * T
h = (A gate) + (B gate)
When chunking:
Example: 4 tokenshg 
Calculate results (Which is just (B gate))
S1   S2   S3   S4
at first timestep, H=0
Calulate combining token pairs
(h-1) = S1 vector
h = (S1 vector * (A gate)) + S2
Repeat for S3 and S4, replacing S1 and S2 respectively
S2   S4
h = (S2 vector * (A gate)) + S4
h = Final answer vector
"""
"""
def load():
    while True:
        with open('dataset.txt', 'r', encoding='utf-8') as data:
            for line in data:
                miku = sp.encode(line.strip(), out_type=int)
                yield miku
"""
def load():
    cache = []
    while True:
        with open('dataset.txt', 'r', encoding='utf-8') as data:
            for line in data:
                miku = sp.encode(line.strip(), out_type=int)
                cache += miku
                #KEEP THIS AS A POWER OF 2 (+1) or tree breaks
                if len(cache) > 1025:
                    returned = cache[:1025]
                    cache = cache[1025:]
                    yield returned

class smth_wicked(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
        self.forget = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
    def forward(self, in_tokens):
        tree_layers = (in_tokens.shape[0] - 1).bit_length() + 1
        c_layer = 0
        residual = in_tokens.clone()
        while c_layer != tree_layers - 1:
            c_layer += 1
            h_stride = 2 ** (c_layer - 1)
            stride = 2 ** c_layer
            h_tok = in_tokens[h_stride-1::stride].clone()
            c_tok = in_tokens[stride-1::stride].clone()
            forget_gate = torch.matmul(h_tok, self.forget)
            add_gate = torch.matmul(c_tok, self.add)
            result = h_tok * torch.sigmoid(forget_gate) + torch.sigmoid(add_gate) * c_tok
            next_tokens = in_tokens.clone()
            next_tokens[stride-1::stride] = result
            in_tokens = next_tokens
        in_tokens += residual
        in_tokens = torch.nn.functional.silu(in_tokens)
        #Applied nonlinear for comparason to layered ver
        return in_tokens
class main_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = smth_wicked()
        self.in_embed = torch.nn.Parameter(torch.empty(Vocab, Dims).uniform_(-Random, Random))
        self.out_embed = torch.nn.Parameter(torch.empty(Vocab, Dims).uniform_(-Random, Random))
    def forward(self, in_tokens):
        state = self.in_embed[in_tokens]
        state = self.layer(state)
        logits = torch.matmul(state[-1], self.out_embed.T)
        return logits
def main():
    data = load()
    model = main_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    p_error = None
    step = 0
    while True:
        step += 1
        start = time.perf_counter()
        miku = next(data)
        if len(miku) <= 2:
            continue
        optimizer.zero_grad()
        #for token in range(1, len(miku)):
        hist_miku = miku[:-1]
        target_miku = miku[-1]
        output = model(hist_miku)
        target_miku = torch.tensor(target_miku)
        loss = torch.nn.functional.cross_entropy(output, target_miku)
        error = loss.item()
        if p_error == None:
            p_error = error
        else:
            p_error = (p_error * 0.95) + (error * 0.05)
        loss.backward()
        optimizer.step()
        end = time.perf_counter()
        torch.cuda.synchronize()
        tps = 1024 / (end - start)
        print(f"{step} {p_error:.2f} {error:.2f} {tps:.1f}")
main()
