import os
import numpy as np
import sentencepiece as spm
#I think i gotta add comments to this one idk
sp = spm.SentencePieceProcessor()
sp.load("spm.model")
S_vocab = sp.vocab_size()
#Settings, should be obvious lol
Cells = 4
Dims = 4
Bounds = 1024
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
def linmax(logits): 
    #Its softmax but linear, turns distances into number 0-1 depending on its share
    logits = np.abs(logits)
    scores = np.max(logits) - logits
    scores += 1e-8
    total = np.sum(scores)
    error = scores / (total)
    return error
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
    def reset(self):
        self.Cells = np.zeros((Cells, Dims), dtype=np.float64)
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
    def learn(self, ans, output, D_Mixer_Output, pred_miku, info, book_keep, forget_input):
        #Init for gradient caches
        p_error = np.zeros((S_vocab), dtype=np.float64)
        G_C_Mixer = np.zeros((Cells), dtype=np.float64)
        G_D_Mixer = np.zeros((Dims, Dims), dtype=np.float64)
        #Non sparse grid bcus too lazy, fix to save memory
        G_Weights = np.zeros((Bounds, Dims), dtype=np.float64)
        G_Forget = np.zeros((Cells, Dims), dtype=np.float64)
        G_Input = np.zeros((Cells, Dims), dtype=np.float64)
        G_V_embed = np.zeros((S_vocab, Dims), dtype=np.float64)
        G_Cells = np.zeros((Cells, Dims), dtype=np.float64)
        #Multiplies then sums every embedding
        for token in range(S_vocab):
            t_token = self.V_embed[token]
            error = np.dot(ans, t_token)
            p_error[token] = error
        #Normalised between 0-1
        p_error = linmax(p_error)
        p_error = p_error[pred_miku]
        error = (1 - p_error)
        d_error = error * self.V_embed[pred_miku]
        #Gradient = the input
        for cell in range(Cells):
            #C_Mixer Gradient
            G_C_Mixer[cell] = np.dot(output[cell], d_error)
            #The gradient that flows back to D_Mixer
            D_Mixer_Error = d_error * self.C_Mixer[cell]
            #D_Mixer gradient
            G_D_Mixer += np.outer(D_Mixer_Output[cell], D_Mixer_Error)
            F_error = np.sum(self.D_Mixer * D_Mixer_Error[None, :], axis=1)
            #Weights gradient
        for cell, rp, dim, index in book_keep:
            #Recalculating for each cell, maybe cache for optimisation
            D_Mixer_Error = d_error * self.C_Mixer[cell]
            F_error = np.sum(self.D_Mixer * D_Mixer_Error[None, :], axis=1)
            #Gradient for main grid weights
            G_Weights[rp, dim] += F_error[dim] * info[cell, dim, index, 1]
            #Gradient for cells, only used to carry forward gradient
            G_Cells[cell, dim] += F_error[dim] * self.Weights[rp, dim] * info[cell, dim, index, 2] / Radius
        for cell in range(Cells):
            for dim in range(Dims):
                #Forget grid gradient
                G_Forget[cell, dim] = G_Cells[cell, dim] * forget_input[cell, dim]
                #Input grid gradient
                G_Input[cell, dim] = G_Cells[cell, dim] * self.V_embed[pred_miku, dim]
                #Token embedding table gradient
                G_V_embed[pred_miku, dim] += G_Cells[cell, dim] * self.Input[cell, dim]
                #Cell gradient, not used to update cell but flow gradient
        #Resetting the changed grids to their original
        #self.Weights = self.C_Weights
        self.Cells = self.C_Cells
        #Applying gradients
        self.V_embed -= G_V_embed * LR
        self.Input -= G_Input * LR
        #No Cells update, wipe clean after ctx done
        self.Weights -= G_Weights * LR
        self.D_Mixer -= G_D_Mixer * LR
        self.C_Mixer -= G_C_Mixer * LR
        self.Forget -= G_Forget * LR
        return error
brain = brain()
load = load()
def main():
    brain.init()
    step = 0
    c_step = 0
    while True:
        miku = next(load)
        brain.reset()
        for token in range(1, len(miku)):
            step += 1
            c_step += 1
            pred_miku = miku[token]
            brain.cache()
            forget_input = brain.input(pred_miku)
            info, output, book_keep = brain.thinkinator()
            ans, output, D_Mixer_Output = brain.cell_upd(output)
            error = brain.learn(ans, output, D_Mixer_Output, pred_miku, info, book_keep, forget_input)
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