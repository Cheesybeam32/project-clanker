import os
import numpy as np
import sentencepiece as spm
#Insert comment here idk
sp = spm.SentencePieceProcessor()
sp.load("spm.model")
S_vocab = sp.vocab_size()

Cells = 4
Dims = 8
Bounds = 1024
LR = 0.001

def load():
    while True:
        with open('dataset.txt', 'r', encoding='utf-8') as data:
            for line in data:
                miku = sp.encode(line.strip(), out_type=int)
                yield miku

def softmax(logits, axis=None):
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    ans = exp / np.sum(exp, axis=axis, keepdims=True)
    return ans

def sigmoid(x):
    ans = 1 / (1 + np.exp(-x))
    return ans

class brain:
    def init(self):
        if os.path.exists("Save.npz"):
            data = np.load("Save.npz")
            self.V_embed = data["V_embed"]
            self.Input = data["Input"]
            self.Cells = np.zeros((Cells, Dims), dtype=np.float32)
            self.Weights = data["Weights"]
            self.D_Mixer = data["D_Mixer"]
            self.C_Mixer = data["C_Mixer"]
            self.Forget = data["Forget"]
            print("Using save file")
        else:
            self.V_embed = np.random.uniform(-1, 1, size=(S_vocab, Dims)).astype(np.float32)
            self.Input = np.random.uniform(-1, 1, size=(Cells, Dims)).astype(np.float32)
            self.Cells = np.zeros((Cells, Dims), dtype=np.float32)
            self.Weights = np.random.uniform(-1, 1, size=(Bounds, Dims)).astype(np.float32)
            self.D_Mixer = np.random.uniform(-1, 1, size=(Dims, Dims)).astype(np.float32)
            self.C_Mixer = np.random.uniform(-1, 1, size=(Cells)).astype(np.float32)
            self.Forget = np.random.uniform(-1, 1, size=(Cells, Dims)).astype(np.float32)   
    def input(self, pred_miku):
        p_embed = self.V_embed[pred_miku]
        embed = p_embed[None, :]
        input_gate = sigmoid(self.Input)
        cell_add = input_gate * embed
        self.Cells += cell_add
    def main_grid(self):
        #Theres 2 ways to do this, give each cell a vector or keep a single scalar but separate by dimentions
        #I chose the former for memory saving + more predictable behavior
        output = np.zeros((Cells, Dims), dtype=np.float32)
        for cell in range(Cells):
            vec = self.Cells[cell, :]
            #This is a dense scan for now but can be made sparse
            #For the sake of testing it is non sparse
            #DO NOT FORGET TO CHANGE THIS, OTHERWISE THE ARCHETECTURE IS POINTLESS
            output[cell, :] = vec * np.sum(self.Weights, axis=0)
        return output
    def cell_upd(self, output):
        forget_gate = sigmoid(self.Forget)
        cell_mix = sigmoid(self.C_Mixer)
        dim_mix = sigmoid(self.D_Mixer)
        self.Cells * forget_gate
        self.Cells + output
        c_multiplied = output * cell_mix[:, None]
        c_mixed = np.sum(c_multiplied, axis=0)
        d_multiplied = c_mixed[:, None] * dim_mix
        d_mixed = np.sum(d_multiplied, axis=0)
        return d_mixed