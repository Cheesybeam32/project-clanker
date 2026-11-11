import os
import torch
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")
Vocab = sp.vocab_size()
torch.set_default_device("cpu")
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
    def forward(self, input):
        #As this is just a demo, for loops will be used
        #In actual version, replace with matrix ops
        len_tokens = input.shape[0]
        tok_left = []
        for token in range(len_tokens):
            c_token = input[token, :]
            add_gate = torch.matmul(self.add, c_token)
            result = torch.sigmoid(add_gate) * c_token
            tok_left.append(result)
        while len(tok_left) != 1:
            temp_arr = []
            for token in range(0, len(tok_left), 2):
                f_token = token + 1
                p_token = tok_left[token]
                if f_token > len(tok_left) - 1:
                    temp_arr.append(p_token)
                    break
                c_token = tok_left[f_token]
                forget_gate = torch.matmul(p_token, self.forget)
                new_state = p_token * torch.sigmoid(forget_gate) + c_token
                temp_arr.append(new_state)
            tok_left = temp_arr
        state_output = tok_left[0]
        return state_output
class main_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = smth_wicked()
        self.in_embed = torch.nn.Parameter(torch.empty(Vocab, Dims).uniform_(-Random, Random))
        self.out_embed = torch.nn.Parameter(torch.empty(Vocab, Dims).uniform_(-Random, Random))
    def forward(self, in_tokens):
        input_matrix = self.in_embed[in_tokens]
        state = self.layer(input_matrix)
        logits = torch.matmul(state, self.out_embed.T)
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
        for token in range(1, len(miku)):
            hist_miku = miku[:token]
            target_miku = miku[token]
            output = model(hist_miku)
            target_miku = torch.tensor(target_miku)
            loss = torch.nn.functional.cross_entropy(output, target_miku)
            print(loss.item())
            total_loss += loss
        total_loss.backward()
        optimizer.step()
main()
