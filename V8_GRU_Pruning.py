import os
import torch
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")
Vocab = sp.vocab_size()

torch.set_default_device("cuda:1")
torch.set_default_dtype(torch.float32)
torch.set_num_threads(20)
torch.set_printoptions(threshold=float("inf"))

LR = 3e-4
Pad = sp.piece_to_id("<pad>")
Batch = 32
Prune_Interval = 7000
Prune_Threshold = 4

Dims = 4096
Random = 0.01

def load():
    while True:
        with open('dataset.txt', 'r', encoding='utf-8') as data:
            for line in data:
                miku = sp.encode(line.strip(), out_type=int)
                yield miku

def pruning(model, prune_list, step, prune_mask):
    added = 0
    cumulative = 0
    for name, param in model.calculator.named_parameters():
        if param.ndim == 2:
            magnitude = torch.sum(torch.abs(param), dim=1)
            gradient = torch.sum(torch.abs(param.grad), dim=1)
        if param.ndim == 1:
            magnitude = torch.abs(param)
            gradient = torch.abs(param.grad)
        importance = magnitude + gradient
        cumulative += importance
        added += 1
    cumulative /= added
    prune_list += cumulative
    prune_list /= 2
    useless = prune_list < Prune_Threshold
    prune_mask[useless] = 0
    return prune_list, prune_mask
class calculator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_state = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
        self.in_token = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
        self.out_state = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
        self.out_token = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
        self.eval_state = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
        self.eval_token = torch.nn.Parameter(torch.empty(Dims, Dims).uniform_(-Random, Random))
        self.in_bias = torch.nn.Parameter(torch.empty(Dims).uniform_(-Random, Random))
        self.out_bias = torch.nn.Parameter(torch.empty(Dims).uniform_(-Random, Random))
        self.eval_bias = torch.nn.Parameter(torch.empty(Dims).uniform_(-Random, Random))
    def forward(self, state, token):
        state_upd = torch.matmul(state, self.in_state.T)
        token_upd = torch.matmul(token, self.in_token.T)
        state_res = torch.matmul(state, self.out_state.T)
        token_res = torch.matmul(token, self.out_token.T)
        update = torch.sigmoid(state_upd + token_upd + self.in_bias)
        reset = torch.sigmoid(state_res + token_res + self.out_bias)
        forgotten = state * reset
        state_eval = torch.matmul(forgotten, self.eval_state.T)
        token_eval = torch.matmul(token, self.eval_token.T)
        evaluation = torch.tanh(state_eval + token_eval + self.eval_bias)
        state = (1 - update) * state + update * evaluation
        return state
class gru_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.empty(Vocab, Dims).uniform_(-Random, Random))
        self.calculator = calculator()
    def forward(self, state, p_token):
        token = self.embed[p_token]
        state = self.calculator(state, token)
        logits = torch.matmul(state, self.embed.T)
        return state, logits
def main():
    data = load()
    model = gru_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)
    prune_list = torch.zeros(Dims)
    prune_mask = torch.ones(Dims)
    step = 0
    prune_counter = 0
    while True:
        inputs = []
        while len(inputs) < Batch:
            miku = next(data)
            if len(miku) <= 2:
                continue
            inputs.append(miku)
        max_len = max(len(x) for x in inputs)
        inputs = torch.tensor([
            x + [Pad] * (max_len - len(x))
            for x in inputs
        ])
        #state = torch.zeros(Batch, Dims)
        state = model.embed[inputs[:, 0]].clone()
        total_loss = 0
        optimizer.zero_grad()
        for token in range(1, max_len):
            step += 1
            pred_miku = inputs[:, token - 1]
            target_miku = inputs[:, token]
            state, logits = model(state, pred_miku)
            loss = torch.nn.functional.cross_entropy(logits, target_miku, ignore_index=Pad)
            error = loss.item()
            if step == 1:
                p_error = error
            else:
                p_error = (p_error * 0.9) + (error * 0.1)
            print(f"Step: {step}, Avg_Loss: {p_error}, Error:{error}")
            total_loss += loss
        total_loss.backward()
        with torch.no_grad():
            for name, param in model.calculator.named_parameters():
                if param.ndim == 2:
                    param.mul_(prune_mask[:, None] * prune_mask[None, :])
                elif param.ndim == 1:
                    param.mul_(prune_mask)
        if Prune_Interval <= step:
            step = 0
            prune_counter += 1
            prune_list, prune_mask = pruning(model, prune_list, prune_counter, prune_mask)
            print(prune_mask)
            print(prune_list)
            input()
        optimizer.step()
main()