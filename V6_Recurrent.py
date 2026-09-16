import os
import numpy as np
import sentencepiece as spm
#I think i gotta add comments to this one idk
sp = spm.SentencePieceProcessor()
sp.load("spm.model")
S_vocab = sp.vocab_size()
#Settings, should be obvious lol
Cells = 8
Dims = 16
Bounds = 2048
Radius = 2
LR = 0.001
F_Radius = Radius * 2 + 1
#Tools that the math uses
def load():
    while True:
        with open('dataset.txt', 'r', encoding='utf-8') as data:
            for line in data:
                miku = sp.encode(line.strip(), out_type=int)
                yield miku
def softmax(logits):
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)
class brain:
    def init(self):
        if os.path.exists("brain.npz"):
            data = np.load("brain.npz")
            self.V_embed = data["V_embed"]
            self.Input = data["Input"]
            self.Cells = np.zeros((Cells, Dims), dtype=np.float64)
            self.Weights = data["Weights"]
            self.D_Mixer = data["D_Mixer"]
            self.C_Mixer = data["C_Mixer"]
            self.Forget = data["Forget"]
        else:
            #Makes the initial grids
            self.V_embed = np.random.uniform(-1, 1, size=(S_vocab, Dims)).astype(np.float64)
            self.Input = np.random.uniform(-1, 1, size=(Cells, Dims)).astype(np.float64)
            self.Cells = np.zeros((Cells, Dims), dtype=np.float64)
            self.Weights = np.random.uniform(-1, 1, size=(Bounds, Dims)).astype(np.float64)
            self.D_Mixer = np.random.uniform(-1, 1, size=(Dims, Dims)).astype(np.float64)
            self.C_Mixer = np.random.uniform(-1, 1, size=(Cells)).astype(np.float64)
            self.Forget = np.random.uniform(-1, 1, size=(Cells, Dims)).astype(np.float64)
            """
            self.V_embed = np.zeros((S_vocab, Dims), dtype=np.float64)
            self.Input = np.zeros((Cells, Dims), dtype=np.float64)
            self.Cells = np.zeros((Cells, Dims), dtype=np.float64)
            self.Weights = np.zeros((Bounds, Dims), dtype=np.float64)
            self.D_Mixer = np.zeros((Dims, Dims), dtype=np.float64)
            self.C_Mixer = np.zeros((Cells), dtype=np.float64)
            self.Forget = np.zeros((Cells, Dims), dtype=np.float64)
            """
        G_Input_total = np.zeros_like(brain.Input)
        G_Weights_total = np.zeros_like(brain.Weights)
        G_D_Mixer_total = np.zeros_like(brain.D_Mixer)
        G_C_Mixer_total = np.zeros_like(brain.C_Mixer)
        G_Forget_total = np.zeros_like(brain.Forget)
        G_V_embed_total = np.zeros_like(brain.V_embed)
        return G_Input_total, G_Weights_total, G_D_Mixer_total, G_C_Mixer_total, G_Forget_total, G_V_embed_total
    def reset(self):
        self.Cells = np.zeros((Cells, Dims), dtype=np.float64)
        G_Cells = np.zeros((Cells, Dims), dtype=np.float64)
        teto = []
        G_Input_total = np.zeros_like(brain.Input)
        G_Weights_total = np.zeros_like(brain.Weights)
        G_D_Mixer_total = np.zeros_like(brain.D_Mixer)
        G_C_Mixer_total = np.zeros_like(brain.C_Mixer)
        G_Forget_total = np.zeros_like(brain.Forget)
        G_V_embed_total = np.zeros_like(brain.V_embed)
        return G_Cells, teto, G_Input_total, G_Weights_total, G_D_Mixer_total, G_C_Mixer_total, G_Forget_total, G_V_embed_total
    def apply(self, G_Input_total, G_Weights_total, G_D_Mixer_total, G_C_Mixer_total, G_Forget_total, G_V_embed_total, joseph):
        self.Input -= G_Input_total/joseph * LR
        self.Weights -= G_Weights_total/joseph * LR
        self.D_Mixer -= G_D_Mixer_total/joseph  * LR
        self.C_Mixer -= G_C_Mixer_total/joseph  * LR
        self.Forget -= G_Forget_total/joseph  * LR
        self.V_embed -= G_V_embed_total/joseph  * LR
    def cache(self):
        self.C_Weights = self.Weights.copy()
        self.C_Cells = self.Cells.copy()
    def input(self, pred_miku):
        #Adds input token to cells
        p_embed = self.V_embed[pred_miku, :]
        embed = p_embed[None, :]
        cell_add = self.Input * embed
        self.Cells += cell_add
        forget_input = self.Cells.copy()
        return forget_input
    def thinkinator(self):
        book_keep = []
        #Main thinking grid that the cells sample from
        output = np.zeros((Cells, Dims), dtype=np.float64)
        info = np.zeros((Cells, Dims, F_Radius, 3), dtype=np.float64)
        for cell in range(Cells):
            for dim in range(Dims):
                influence = 0
                dist = self.Cells[cell, dim]
                wp = int(dist)
                for index in range(-(Radius + 1), Radius + 2):
                    rp = wp + index
                    pos = np.abs(rp - dist)
                    s_pos = np.sign(rp - dist)
                    strength = 1 - (pos / Radius)
                    if strength <= 0:
                        continue
                    weight = self.Weights[rp, dim]
                    weight *= strength
                    info[cell, dim, index, 0] = weight
                    info[cell, dim, index, 1] = strength
                    info[cell, dim, index, 2] = s_pos
                    book_keep.append((cell, rp, dim, index + 2))
                    influence += weight
                output[cell, dim] = influence
        return info, output, book_keep
    def cell_upd(self, output):
        #Forgets previous state and adds new vector back to cells
        self.Cells *= self.Forget
        #Mixes dimentions w eachother
        D_Mixer_Output = output.copy()
        for cell in range(Cells):
            split = output[cell]
            p_add = split[None, :] * self.D_Mixer
            add = np.sum(p_add, axis=0)
            output[cell, :] = add
        self.Cells += output
        #Mixes the cell answers together for the usable answer
        p_ans = output * self.C_Mixer[:, None]
        ans = np.sum(p_ans, axis=0)
        #Saving output for backward pass
        return ans, output, D_Mixer_Output
    def learn(self, ans, output, D_Mixer_Output, pred_miku, info, book_keep, forget_input, G_Cells):
        #Init for gradient caches
        p_error = np.zeros((S_vocab), dtype=np.float64)
        G_C_Mixer = np.zeros((Cells), dtype=np.float64)
        G_D_Mixer = np.zeros((Dims, Dims), dtype=np.float64)
        #Non sparse grid bcus too lazy, fix to save memory
        G_Weights = np.zeros((Bounds, Dims), dtype=np.float64)
        G_Forget = np.zeros((Cells, Dims), dtype=np.float64)
        G_Input = np.zeros((Cells, Dims), dtype=np.float64)
        G_V_embed = np.zeros((S_vocab, Dims), dtype=np.float64)
        G_P_Cells = np.zeros((Cells, Dims), dtype=np.float64)
        #Multiplies then sums every embedding
        for token in range(S_vocab):
            t_token = self.V_embed[token]
            error = np.dot(ans, t_token)
            p_error[token] = error
        #Normalised between 0-1
        p_error = softmax(p_error)
        p_vec = np.dot(p_error, self.V_embed)
        d_error = p_vec - self.V_embed[pred_miku]
        error = -np.log(p_error[pred_miku] + 1e-8)
        #Gradient = the input
        G_Output = np.zeros((Cells, Dims), dtype=np.float64)
        for cell in range(Cells):
            G_C_Mixer[cell] = np.dot(output[cell], d_error)
            G_Output[cell] = d_error * self.C_Mixer[cell]
            G_Output[cell] += np.dot(self.D_Mixer, G_Cells[cell])
            G_D_Mixer += np.outer(D_Mixer_Output[cell], d_error * self.C_Mixer[cell])
            G_D_Mixer += np.outer(D_Mixer_Output[cell], G_Cells[cell])
        for cell, rp, dim, index in book_keep:
            F_error = G_Output[cell]
            #Gradient for main grid weights
            G_Weights[rp, dim] += F_error[dim] * info[cell, dim, index, 1]
            #Gradient for cells, only used to carry forward gradient
            G_P_Cells[cell, dim] += F_error[dim] * self.Weights[rp, dim] * info[cell, dim, index, 2] / Radius
        for cell in range(Cells):
            for dim in range(Dims):
                #Gradient of previous state
                G_P_Cells[cell, dim] += G_Cells[cell, dim] * self.Forget[cell, dim]
                #Gradient of forget grid
                G_Forget[cell, dim] = G_Cells[cell, dim] * forget_input[cell, dim]
                #Gradient of input grid
                G_Input[cell, dim] = G_P_Cells[cell, dim] * self.V_embed[pred_miku, dim]
                #Gradient of token embedding
                G_V_embed[pred_miku, dim] += G_P_Cells[cell, dim] * self.Input[cell, dim]
        #Resetting the changed grids to their original
        #self.Weights = self.C_Weights
        self.Cells = self.C_Cells
        return error, G_C_Mixer, G_D_Mixer, G_Weights, G_Forget, G_Input, G_V_embed, G_P_Cells
brain = brain()
load = load()
def main():
    G_Input_total, G_Weights_total, G_D_Mixer_total, G_C_Mixer_total, G_Forget_total, G_V_embed_total = brain.init()
    step = 0
    c_step = 0
    while True:
        miku = next(load)
        joseph = 0
        G_Cells, teto, G_Input_total, G_Weights_total, G_D_Mixer_total, G_C_Mixer_total, G_Forget_total, G_V_embed_total = brain.reset()
        for token in range(1, len(miku)):
            step += 1
            c_step += 1
            pred_miku = miku[token]
            ctx_miku = miku[token - 1]
            brain.cache()
            #Adding token to the cell states
            forget_input = brain.input(ctx_miku)
            #Scanning main grid
            info, output, book_keep = brain.thinkinator()
            #Updating to new state
            ans, output, D_Mixer_Output = brain.cell_upd(output)
            #Creating list to reverse for BBTT
            teto.append((ans, output, D_Mixer_Output, pred_miku, info, book_keep, forget_input))
            kaai_yuki = teto.copy()
            kaai_yuki.reverse()
            for ans, output, D_Mixer_Output, pred_miku, info, book_keep, forget_input in kaai_yuki:
                error, G_C_Mixer, G_D_Mixer, G_Weights, G_Forget, G_Input, G_V_embed, G_Cells = brain.learn(ans, output, D_Mixer_Output, pred_miku, info, book_keep, forget_input, G_Cells)
                G_Input_total += G_Input
                G_Weights_total += G_Weights
                G_D_Mixer_total += G_D_Mixer
                G_C_Mixer_total += G_C_Mixer
                G_Forget_total += G_Forget
                G_V_embed_total += G_V_embed
                joseph += 1
            brain.apply(G_Input_total, G_Weights_total, G_D_Mixer_total, G_C_Mixer_total, G_Forget_total, G_V_embed_total, joseph)
            print(f"Step: {step}, Error: {error}")
            if c_step == 100:
                np.savez(
                    "Save.npz",
                    V_embed=brain.V_embed,
                    Input=brain.Input,
                    Weights=brain.Weights,
                    D_Mixer=brain.D_Mixer,
                    C_Mixer=brain.C_Mixer,
                    Forget=brain.Forget
                )
                c_step = 0
                print("Saved")
main()