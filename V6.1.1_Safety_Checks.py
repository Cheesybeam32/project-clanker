import os
import numpy as np
import sentencepiece as spm
#I uh forgot to add comments to this one so deal with it :P
sp = spm.SentencePieceProcessor()
sp.load("spm.model")
S_vocab = sp.vocab_size()

Cells = 64
Dims = 128
Bounds = 16384
Radius = 5
LR = 0.001
Punish_LR = 0.0001
F_Radius = Radius * 2 + 1

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

def o_crud(input, flag):
    if not np.isfinite(input).all():
        print(f"SOMETHING WENT WRONG AT CHECK {flag}")
        raise ValueError("Yea...")

class brain:

    def init(self):
        if os.path.exists("Save.npz"):
            data = np.load("Save.npz")
            self.V_embed = data["V_embed"]
            self.Input = data["Input"]
            self.Cells = np.zeros((Cells, Dims), dtype=np.float64)
            self.Weights = data["Weights"]
            self.D_Mixer = data["D_Mixer"]
            self.C_Mixer = data["C_Mixer"]
            self.Forget = data["Forget"]
            print("Using save file")
        else:
            self.V_embed = np.random.uniform(-1, 1, size=(S_vocab, Dims)).astype(np.float64)
            self.Input = np.random.uniform(-1, 1, size=(Cells, Dims)).astype(np.float64)
            self.Cells = np.zeros((Cells, Dims), dtype=np.float64)
            self.Weights = np.random.uniform(-1, 1, size=(Bounds, Dims)).astype(np.float64)
            self.D_Mixer = np.random.uniform(-1, 1, size=(Dims, Dims)).astype(np.float64)
            self.C_Mixer = np.random.uniform(-1, 1, size=(Cells)).astype(np.float64)
            self.Forget = np.random.uniform(-1, 1, size=(Cells, Dims)).astype(np.float64)
        G_Input_total = np.zeros_like(self.Input)
        G_Weights_total = np.zeros_like(self.Weights)
        G_D_Mixer_total = np.zeros_like(self.D_Mixer)
        G_C_Mixer_total = np.zeros_like(self.C_Mixer)
        G_Forget_total = np.zeros_like(self.Forget)
        G_V_embed_total = np.zeros_like(self.V_embed)
        return G_Input_total, G_Weights_total, G_D_Mixer_total, G_C_Mixer_total, G_Forget_total, G_V_embed_total

    def reset(self):
        self.Cells = np.zeros((Cells, Dims), dtype=np.float64)
        G_Cells = np.zeros((Cells, Dims), dtype=np.float64)
        teto = []
        G_Input_total = np.zeros_like(self.Input)
        G_Weights_total = np.zeros_like(self.Weights)
        G_D_Mixer_total = np.zeros_like(self.D_Mixer)
        G_C_Mixer_total = np.zeros_like(self.C_Mixer)
        G_Forget_total = np.zeros_like(self.Forget)
        G_V_embed_total = np.zeros_like(self.V_embed)
        return G_Cells, teto, G_Input_total, G_Weights_total, G_D_Mixer_total, G_C_Mixer_total, G_Forget_total, G_V_embed_total

    def apply(self, G_Input_total, G_Weights_total, G_D_Mixer_total, G_C_Mixer_total, G_Forget_total, G_V_embed_total):
        self.Input -= G_Input_total * LR
        self.Weights -= G_Weights_total * LR
        self.D_Mixer -= G_D_Mixer_total * LR
        self.C_Mixer -= G_C_Mixer_total * LR
        self.Forget -= G_Forget_total * LR
        self.V_embed -= G_V_embed_total * LR

    def cache(self):
        self.C_Weights = self.Weights.copy()
        self.C_Cells = self.Cells.copy()

    def input(self, pred_miku):
        p_embed = self.V_embed[pred_miku]
        embed = p_embed[None, :]
        input_gate = sigmoid(self.Input)
        cell_add = input_gate * embed
        self.Cells += cell_add
        o_crud(cell_add, flag=1)
        forget_input = self.Cells.copy()
        return forget_input, input_gate

    def thinkinator(self):
        book_keep = []
        output = np.zeros((Cells, Dims), dtype=np.float64)
        info = np.zeros((Cells, Dims, F_Radius, 3), dtype=np.float64)
        for cell in range(Cells):
            for dim in range(Dims):
                influence = 0
                dist = self.Cells[cell, dim]
                o_crud(dist, flag=2)
                wp = int(dist)

                for index in range(-Radius, Radius + 1):
                    rp = wp + index
                    if rp < 0 or rp >= Bounds:
                        continue
                    pos = abs(rp - dist)
                    o_crud(pos, flag=3)
                    s_pos = np.sign(rp - dist)
                    strength = 1 - (pos / Radius)
                    o_crud(strength, flag=4)
                    if strength <= 0:
                        continue
                    weight = self.Weights[rp, dim]
                    o_crud(weight, flag=5)
                    weight *= strength
                    info[cell, dim, index + Radius, 0] = weight
                    info[cell, dim, index + Radius, 1] = strength
                    info[cell, dim, index + Radius, 2] = s_pos
                    book_keep.append((cell, rp, dim, index + Radius))
                    influence += weight
                    o_crud(influence, flag=6)
                output[cell, dim] = influence
        return info, output, book_keep
    def cell_upd(self, output):
        forget_gate = sigmoid(self.Forget)
        o_crud(forget_gate, flag=7)
        self.Cells *= forget_gate
        D_Mixer_Output = output.copy()
        D_Mixer_Soft = softmax(self.D_Mixer, axis=0)
        o_crud(D_Mixer_Soft, flag=8)
        for cell in range(Cells):
            split = output[cell]
            p_add = split[:, None] * D_Mixer_Soft
            add = np.sum(p_add, axis=0)
            o_crud(add, flag=9)
            output[cell, :] = add
        self.Cells += output
        C_Mixer_Soft = softmax(self.C_Mixer)
        p_ans = output * C_Mixer_Soft[:, None]
        ans = np.sum(p_ans, axis=0)
        o_crud(ans, flag=10)
        return ans, output, D_Mixer_Output, forget_gate, D_Mixer_Soft, C_Mixer_Soft

    def learn(self, ans, output, D_Mixer_Output, pred_miku, ctx_miku, info, book_keep, forget_input, G_Cells, forget_gate, D_Mixer_Soft, C_Mixer_Soft, post_cells):
        p_error = np.zeros((S_vocab), dtype=np.float64)
        G_C_Mixer_Soft = np.zeros((Cells), dtype=np.float64)
        G_C_Mixer = np.zeros((Cells), dtype=np.float64)
        G_D_Mixer_Soft = np.zeros((Dims, Dims), dtype=np.float64)
        G_D_Mixer = np.zeros((Dims, Dims), dtype=np.float64)
        G_Weights = np.zeros((Bounds, Dims), dtype=np.float64)
        G_Forget = np.zeros((Cells, Dims), dtype=np.float64)
        G_Input = np.zeros((Cells, Dims), dtype=np.float64)
        G_V_embed = np.zeros((S_vocab, Dims), dtype=np.float64)
        G_P_Cells = np.zeros((Cells, Dims), dtype=np.float64)
        for token in range(S_vocab):
            t_token = self.V_embed[token]
            error = np.dot(ans, t_token)
            p_error[token] = error
        p_error = softmax(p_error)
        p_vec = np.dot(p_error, self.V_embed)
        d_error = p_vec - self.V_embed[pred_miku]
        error = -np.log(p_error[pred_miku] + 1e-8)
        for token in range(S_vocab):
            G_V_embed[token] += p_error[token] * ans
        G_V_embed[pred_miku] -= ans
        G_Output = np.zeros((Cells, Dims), dtype=np.float64)
        for cell in range(Cells):
            G_C_Mixer_Soft[cell] = np.dot(output[cell], d_error)
            G_Output[cell] = d_error * C_Mixer_Soft[cell]
            G_Output[cell] += np.dot(D_Mixer_Soft, G_Cells[cell])
            G_D_Mixer_Soft += np.outer(D_Mixer_Output[cell], d_error * C_Mixer_Soft[cell])
            G_D_Mixer_Soft += np.outer(D_Mixer_Output[cell], G_Cells[cell])
        c_sum = np.sum(G_C_Mixer_Soft * C_Mixer_Soft)
        for cell in range(Cells):
            G_C_Mixer[cell] = C_Mixer_Soft[cell] * (G_C_Mixer_Soft[cell] - c_sum)
        for out_dim in range(Dims):
            d_sum = 0
            for in_dim in range(Dims):
                d_sum += G_D_Mixer_Soft[in_dim, out_dim] * D_Mixer_Soft[in_dim, out_dim]
            for in_dim in range(Dims):
                G_D_Mixer[in_dim, out_dim] = D_Mixer_Soft[in_dim, out_dim] * (G_D_Mixer_Soft[in_dim, out_dim] - d_sum)
        for cell, rp, dim, index in book_keep:
            F_error = G_Output[cell]
            G_Weights[rp, dim] += F_error[dim] * info[cell, dim, index, 1]
            G_P_Cells[cell, dim] += F_error[dim] * self.Weights[rp, dim] * info[cell, dim, index, 2] / Radius
        punish = self.crowd_punish(post_cells)
        G_Cells += punish * Punish_LR
        for cell in range(Cells):
            for dim in range(Dims):
                G_P_Cells[cell, dim] += G_Cells[cell, dim] * forget_gate[cell, dim]
                G_Forget[cell, dim] = G_Cells[cell, dim] * forget_input[cell, dim] * forget_gate[cell, dim] * (1 - forget_gate[cell, dim])
                G_Input[cell, dim] = G_P_Cells[cell, dim] * self.V_embed[ctx_miku, dim] * sigmoid(self.Input[cell, dim]) * (1 - sigmoid(self.Input[cell, dim]))
                G_V_embed[ctx_miku, dim] += G_P_Cells[cell, dim] * sigmoid(self.Input[cell, dim])
        self.Cells = self.C_Cells
        return error, G_C_Mixer, G_D_Mixer, G_Weights, G_Forget, G_Input, G_V_embed, G_P_Cells

    def crowd_punish(self, cells):
        punish = np.zeros((Cells, Dims), dtype=np.float64)
        for cell in range(Cells):
            for other in range(cell + 1, Cells):
                dist = cells[cell] - cells[other]
                for dim in range(Dims):
                    distance = abs(dist[dim])
                    if distance >= F_Radius:
                        continue
                    strength = 1 - distance / F_Radius
                    punish[cell, dim] -= np.sign(dist[dim]) * strength
                    punish[other, dim] += np.sign(dist[dim]) * strength
        return punish
    def flush(self):
        print(f"""
        {self.V_embed}
        {self.Input}
        {self.Cells}
        {self.C_Cells}
        {self.Weights}
        {self.C_Mixer}
        {self.D_Mixer}
        {self.Forget}
        """)
        input("Continue")

brain = brain()
load = load()

def main():
    G_Input_total, G_Weights_total, G_D_Mixer_total, G_C_Mixer_total, G_Forget_total, G_V_embed_total = brain.init()
    step = 0
    c_step = 0
    while True:
        miku = next(load)
        G_Cells, teto, G_Input_total, G_Weights_total, G_D_Mixer_total, G_C_Mixer_total, G_Forget_total, G_V_embed_total = brain.reset()
        if len(miku) <= 1:
            continue
        for token in range(1, len(miku)):
            step += 1
            c_step += 1
            pred_miku = miku[token]
            ctx_miku = miku[token - 1]
            brain.cache()
            forget_input, input_gate = brain.input(ctx_miku)
            info, output, book_keep = brain.thinkinator()
            ans, output, D_Mixer_Output, forget_gate, D_Mixer_Soft, C_Mixer_Soft = brain.cell_upd(output)
            post_cells = brain.Cells.copy()
            teto.append((ans, output, D_Mixer_Output, pred_miku, ctx_miku, info, book_keep, forget_input, forget_gate, D_Mixer_Soft, C_Mixer_Soft, post_cells))
            if c_step == 1000:
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
        kaai_yuki = teto.copy()
        kaai_yuki.reverse()
        for ans, output, D_Mixer_Output, pred_miku, ctx_miku, info, book_keep, forget_input, forget_gate, D_Mixer_Soft, C_Mixer_Soft, post_cells in kaai_yuki:
            error, G_C_Mixer, G_D_Mixer, G_Weights, G_Forget, G_Input, G_V_embed, G_Cells = brain.learn(ans, output, D_Mixer_Output, pred_miku, ctx_miku, info, book_keep, forget_input, G_Cells, forget_gate, D_Mixer_Soft, C_Mixer_Soft, post_cells)
            G_Input_total += G_Input
            G_Weights_total += G_Weights
            G_D_Mixer_total += G_D_Mixer
            G_C_Mixer_total += G_C_Mixer
            G_Forget_total += G_Forget
            G_V_embed_total += G_V_embed
            print(f"Step: {step}, Error: {error}")
        brain.apply(G_Input_total, G_Weights_total, G_D_Mixer_total, G_C_Mixer_total, G_Forget_total, G_V_embed_total)
        #brain.flush()
main()