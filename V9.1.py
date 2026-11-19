import os
import torch
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")
Vocab = sp.vocab_size()
torch.set_default_device("xpu")
torch.set_default_dtype(torch.float32)
torch.set_num_threads(8)
torch.set_num_interop_threads(2)

Dims = 256

Random = 0.01
LR = 0.001
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
*UPDATE FOR V9.1*
Stage 0:
T0  T1  T2  T3  T4  T5  T6  T7
Stage 1:
T0  S01 S12 S23 S34 S45 S56 S67
Stage 2:
T0  S01 S012 S0123 S1234 S2345 S3456 S4567
Stage 3:
T0  S01 S012 S0123 S01234 S012345 S0123456 S01234567
underlying equation is same, just produces vec for every pos bcus layering
"""

def load():
    while True:
        with open('dataset.txt', 'r', encoding='utf-8') as data:
            for line in data:
                miku = sp.encode(line.strip(), out_type=int)
                yield miku

class smth_wicked(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
        self.forget = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
    def forward(self, in_tokens):
        # in_tokens shape: (len_ctx, Dims)
        len_ctx = in_tokens.shape[0]
        tree_layers = (len_ctx - 1).bit_length()
        for c_layer in range(tree_layers):
            offset = 2 ** c_layer
            p_tok = in_tokens[:-offset]
            c_tok = in_tokens[offset:]
            add_gate = torch.sigmoid(torch.matmul(c_tok, self.add))
            forget_gate = torch.sigmoid(torch.matmul(p_tok, self.forget))
            updated = p_tok * (1.0 - forget_gate) + c_tok * add_gate
            in_tokens = torch.cat([in_tokens[:offset], updated], dim=0)    
        return in_tokens
class main_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = smth_wicked()
        self.in_embed = torch.nn.Parameter(torch.empty(Vocab, Dims).uniform_(-Random, Random))
        self.out_embed = torch.nn.Parameter(torch.empty(Vocab, Dims).uniform_(-Random, Random))
    def forward(self, in_tokens):
        input_matrix = self.in_embed[in_tokens]
        state = self.layer(input_matrix)
        final_vec = state[-1, :]
        logits = torch.matmul(final_vec, self.out_embed.T)
        return logits
def main():
    data = load()
    model = main_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    while True:
        miku = next(data)
        if len(miku) <= 2:
            continue
        total_loss = 0
        optimizer.zero_grad()
        hist_miku = miku[:-1]
        target_miku = miku[-1]
        output = model(hist_miku)
        target_miku = torch.tensor(target_miku)
        loss = torch.nn.functional.cross_entropy(output, target_miku)
        print(loss.item())
        loss.backward()
        optimizer.step()
main()
