"""
Session 14 assignment: train a dense (standard FFN) transformer, convert it into a
Mixture-of-Experts model by upcycling, and show the MoE keeps training and reducing loss.

Pipeline
  1. Dense phase   : char-level GPT on Tiny Shakespeare, SwiGLU FFN (width 512), trained from scratch.
  2. Upcycle       : every FFN -> router + N experts, each expert initialised from the dense FFN.
                       - "copy"  : sparse upcycling (Komatsuzaki et al., 2022), all experts identical.
                                   Router starts ~0 and top-k weights are renormalised to sum to 1,
                                   so the MoE computes exactly the dense function at step 0.
                       - "drop"  : drop-upcycling (r = 0.5), each expert re-draws half its neurons
                                   so experts start different.
  3. Continue      : train each branch for the same number of steps, alongside a dense baseline
                     that just keeps training, and log train/val loss plus expert load statistics.

Balancing: aux-loss-free per-expert selection bias (Wang et al., 2024 / DeepSeek-V3) plus a small
Switch-style auxiliary loss computed over the whole batch. Router runs in fp32 and starts small.

Run:  python moe_upcycle.py            (CPU is fine, ~20-30 min total)
"""
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(os.cpu_count() or 4)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- config
@dataclass
class Cfg:
    # model
    n_layer: int = 4
    d_model: int = 128
    n_head: int = 4
    d_ff: int = 512           # dense FFN inner width
    ctx: int = 64
    # MoE
    n_experts: int = 8
    top_k: int = 2
    drop_ratio: float = 0.5   # used by the "drop" branch
    router_init_std: float = 1e-3
    aux_alpha: float = 1e-3   # Switch aux loss weight
    bias_gamma: float = 5e-3  # aux-loss-free bias update speed (raised from 1e-3: short run)
    # training
    batch: int = 32
    lr: float = 1e-3
    warmup: int = 100
    dense_steps: int = 1200
    moe_steps: int = 1000
    eval_every: int = 100
    eval_batches: int = 20
    seed: int = 1337


CFG = Cfg()


# ----------------------------------------------------------------------------- data
def load_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")
    if not os.path.exists(path):
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], chars


TRAIN, VAL, CHARS = load_data()
VOCAB = len(CHARS)


def get_batch(split, gen):
    d = TRAIN if split == "train" else VAL
    ix = torch.randint(len(d) - CFG.ctx - 1, (CFG.batch,), generator=gen)
    x = torch.stack([d[i:i + CFG.ctx] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + CFG.ctx] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ----------------------------------------------------------------------------- model
class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh = c.n_head
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.nh, C // self.nh).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    """Dense feed-forward network: W_down( SiLU(W_gate x) * W_up x )."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """Router (softmax, top-k, renormalised) + n_experts SwiGLU experts, dropless."""

    def __init__(self, c, d_ff):
        super().__init__()
        self.E, self.k = c.n_experts, c.top_k
        self.alpha, self.gamma = c.aux_alpha, c.bias_gamma
        self.router = nn.Linear(c.d_model, c.n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=c.router_init_std)
        self.experts = nn.ModuleList(SwiGLU(c.d_model, d_ff) for _ in range(c.n_experts))
        # selection-only bias, updated by a sign rule, never by gradients
        self.register_buffer("sel_bias", torch.zeros(c.n_experts))
        self.last_load = None
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, T, C = x.shape
        h = x.reshape(-1, C)
        logits = self.router(h.float())                       # router in fp32
        probs = logits.softmax(-1)                            # (N, E)
        _, idx = (probs + self.sel_bias).topk(self.k, dim=-1) # bias steers choice only
        w = probs.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True)                       # weights sum to 1

        out = torch.zeros_like(h)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            y = expert(h[tok]) * w[tok, slot].unsqueeze(-1).to(h.dtype)
            out.index_add_(0, tok, y)

        # load = fraction of routed slots per expert, counted over the whole batch
        f = torch.bincount(idx.flatten(), minlength=self.E).float() / idx.numel()
        P = probs.mean(0)
        self.aux_loss = self.alpha * self.E * (f.detach() * P).sum()
        self.last_load = f.detach()
        return out.view(B, T, C)

    @torch.no_grad()
    def update_bias(self):
        if self.last_load is not None:
            self.sel_bias += self.gamma * torch.sign(self.last_load.mean() - self.last_load)


class Block(nn.Module):
    def __init__(self, c, moe=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.ffn = MoE(c, c.d_ff) if moe else SwiGLU(c.d_model, c.d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c, moe=False):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(VOCAB, c.d_model)
        self.pos = nn.Embedding(c.ctx, c.d_model)
        self.blocks = nn.ModuleList(Block(c, moe) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, VOCAB, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear) and m.weight.shape[0] != CFG.n_experts:
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def moe_layers(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoE)]

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
        return logits, loss

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        active = total
        for m in self.moe_layers():
            per_expert = sum(p.numel() for p in m.experts[0].parameters())
            active -= per_expert * (m.E - m.k)
        return total, active


# ----------------------------------------------------------------------------- upcycling
@torch.no_grad()
def upcycle(dense: GPT, mode: str, seed: int) -> GPT:
    """Build an MoE GPT whose experts start from the dense FFN weights."""
    g = torch.Generator().manual_seed(seed)
    moe = GPT(CFG, moe=True).to(DEVICE)
    # copy everything that is not an FFN (embeddings, attention, norms, head)
    dense_sd = dense.state_dict()
    moe_sd = moe.state_dict()
    for k, v in dense_sd.items():
        if ".ffn." not in k:
            moe_sd[k].copy_(v)

    for bi, (db, mb) in enumerate(zip(dense.blocks, moe.blocks)):
        src = db.ffn
        for e, expert in enumerate(mb.ffn.experts):
            gate, up, down = src.gate.weight.clone(), src.up.weight.clone(), src.down.weight.clone()
            if mode == "drop":
                # drop-upcycling: re-initialise a random fraction r of the inner neurons,
                # using the dense weights' own statistics (Nakamura et al., 2025)
                n_drop = int(CFG.drop_ratio * CFG.d_ff)
                drop = torch.randperm(CFG.d_ff, generator=g)[:n_drop]
                for W, rows in ((gate, True), (up, True), (down, False)):
                    ref = W[drop] if rows else W[:, drop]
                    new = torch.randn(ref.shape, generator=g) * ref.std() + ref.mean()
                    if rows:
                        W[drop] = new.to(W)
                    else:
                        W[:, drop] = new.to(W)
            expert.gate.weight.copy_(gate)
            expert.up.weight.copy_(up)
            expert.down.weight.copy_(down)
        # router: tiny random init so the scores start near uniform and differ only slightly
        mb.ffn.router.weight.copy_(torch.randn(mb.ffn.router.weight.shape, generator=g) * CFG.router_init_std)
    return moe


# ----------------------------------------------------------------------------- training
@torch.no_grad()
def evaluate(model, gen):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = [model(*get_batch(split, gen))[1].item() for _ in range(CFG.eval_batches)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def lr_at(step):
    return CFG.lr * min(1.0, (step + 1) / CFG.warmup)


def train(model, steps, name, step_offset=0, log=None):
    """Train for `steps` steps. Returns a log list of dicts."""
    log = log if log is not None else []
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, betas=(0.9, 0.95), weight_decay=0.1)
    gen_train = torch.Generator().manual_seed(CFG.seed + step_offset)
    gen_eval = torch.Generator().manual_seed(999)
    t0 = time.time()
    for s in range(steps + 1):
        gstep = step_offset + s
        if s % CFG.eval_every == 0 or s == steps:
            ev = evaluate(model, torch.Generator().manual_seed(999))
            rec = {"run": name, "step": gstep, "train": ev["train"], "val": ev["val"]}
            moes = model.moe_layers()
            if moes:
                loads = [m.last_load.tolist() if m.last_load is not None else None for m in moes]
                rec["load"] = loads
                rec["maxvio"] = [max(l) / (sum(l) / len(l)) - 1 if l else None for l in loads]
                rec["dead"] = [sum(1 for v in l if v < 0.1 / len(l)) if l else None for l in loads]
            log.append(rec)
            extra = ""
            if moes and rec["load"][0] is not None:
                extra = f" | MaxVio/layer {[round(v, 2) for v in rec['maxvio']]}"
            print(f"[{name}] step {gstep:5d} train {ev['train']:.4f} val {ev['val']:.4f}"
                  f" ({time.time() - t0:.0f}s){extra}", flush=True)
        if s == steps:
            break
        for pg in opt.param_groups:
            pg["lr"] = lr_at(s)
        x, y = get_batch("train", gen_train)
        _, loss = model(x, y)
        aux = sum((m.aux_loss for m in model.moe_layers()), torch.tensor(0.0, device=DEVICE))
        opt.zero_grad(set_to_none=True)
        (loss + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for m in model.moe_layers():
            m.update_bias()
    return log


# ----------------------------------------------------------------------------- analysis
def char_class(c):
    if c == " ":
        return "space"
    if c == "\n":
        return "newline"
    if c.isupper():
        return "upper"
    if c in "aeiou":
        return "vowel"
    if c.isalpha():
        return "consonant"
    return "punct"


@torch.no_grad()
def expert_specialisation(model, layer=-1):
    """Share of each character class routed (top-1) to each expert in one layer."""
    model.eval()
    moe = model.moe_layers()[layer]
    captured = {}
    hook = moe.register_forward_hook(lambda m, i, o: captured.__setitem__("h", i[0]))
    x, _ = get_batch("val", torch.Generator().manual_seed(7))
    model(x)
    hook.remove()
    h = captured["h"].reshape(-1, CFG.d_model)
    top1 = (moe.router(h.float()).softmax(-1) + moe.sel_bias).argmax(-1)
    classes = ["space", "newline", "upper", "vowel", "consonant", "punct"]
    table = {c: [0] * moe.E for c in classes}
    for ch, e in zip(x.flatten().tolist(), top1.tolist()):
        table[char_class(CHARS[ch])][e] += 1
    for c in classes:
        s = sum(table[c]) or 1
        table[c] = [v / s for v in table[c]]
    model.train()
    return table


@torch.no_grad()
def sample(model, n=300, seed=0):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    for _ in range(n):
        logits, _ = model(idx[:, -CFG.ctx:])
        p = logits[:, -1].softmax(-1)
        idx = torch.cat([idx, torch.multinomial(p, 1, generator=g)], 1)
    model.train()
    return "".join(CHARS[i] for i in idx[0].tolist())


# ----------------------------------------------------------------------------- main
def main():
    torch.manual_seed(CFG.seed)
    summary = {"config": asdict(CFG), "vocab": VOCAB, "device": DEVICE}

    # 1) dense phase -----------------------------------------------------------
    dense_ckpt = os.path.join(OUT, "dense.pt")
    dense = GPT(CFG).to(DEVICE)
    if os.path.exists(dense_ckpt):
        dense.load_state_dict(torch.load(dense_ckpt))
        dense_log = json.load(open(os.path.join(OUT, "dense_log.json")))
        print("loaded dense checkpoint")
    else:
        print(f"dense params: {dense.param_counts()[0]:,}")
        dense_log = train(dense, CFG.dense_steps, "dense")
        torch.save(dense.state_dict(), dense_ckpt)
        json.dump(dense_log, open(os.path.join(OUT, "dense_log.json"), "w"))
    summary["dense_params"] = dense.param_counts()[0]
    summary["dense_final"] = dense_log[-1]

    # 2) upcycle + sanity check ------------------------------------------------
    gen = lambda: torch.Generator().manual_seed(999)
    base = evaluate(dense, gen())
    checks = {}
    for mode in ("copy", "drop"):
        m = upcycle(dense, mode, seed=CFG.seed)
        ev = evaluate(m, gen())
        checks[mode] = {"dense_val": base["val"], "moe_val_at_conversion": ev["val"],
                        "params_total_active": m.param_counts()}
        print(f"[upcycle-{mode}] val loss dense {base['val']:.5f} -> MoE at step 0 {ev['val']:.5f} | "
              f"total/active params {m.param_counts()}")
    summary["conversion"] = checks

    # 3) continue training: dense baseline vs the two upcycled MoEs --------------
    logs = {}
    start = CFG.dense_steps
    torch.manual_seed(CFG.seed + 1)
    d2 = GPT(CFG).to(DEVICE)
    d2.load_state_dict(dense.state_dict())
    logs["dense-continued"] = train(d2, CFG.moe_steps, "dense-continued", step_offset=start)
    finals = {"dense-continued": d2}

    for mode in ("copy", "drop"):
        torch.manual_seed(CFG.seed + 1)
        m = upcycle(dense, mode, seed=CFG.seed)
        name = f"moe-{mode}"
        logs[name] = train(m, CFG.moe_steps, name, step_offset=start)
        finals[name] = m
        torch.save(m.state_dict(), os.path.join(OUT, f"{name}.pt"))
        summary[f"{name}_specialisation_last_layer"] = expert_specialisation(m)

    logs["dense"] = dense_log
    json.dump(logs, open(os.path.join(OUT, "all_logs.json"), "w"))
    summary["finals"] = {k: v[-1] for k, v in logs.items()}
    summary["samples"] = {k: sample(v) for k, v in finals.items()}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps({k: {"train": round(v["train"], 4), "val": round(v["val"], 4)}
                      for k, v in summary["finals"].items()}, indent=2))


if __name__ == "__main__":
    main()
