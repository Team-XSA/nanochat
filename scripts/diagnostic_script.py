"""
script for replicating figure 1 from the XSA paper.

written by Hayden Free. Written with aid from Claude 4.7 Opus and the nanochat DeepWiki (https://deepwiki.com/karpathy/nanochat)
"""
import json
from nanochat.gpt import norm
from nanochat.gpt import apply_rotary_emb
import argparse
from pathlib import Path
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

OUTPUT_DIR = Path('./diagnostics')
DEVICE = torch.device('cuda') 
N_BATCHES = 8
BATCH_SIZE = 1
SEQ_LEN = 2048

def create_attention_mask(T, window_size, device):
    left, right = window_size
    mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=right + 1)
    if left != -1:
        idx = torch.arange(T, device=device)
        mask = mask | (idx[None, :] < (idx[:, None] - left))
    return mask

def create_patched_forward(attn_module, layer_idx, sums):
    original_forward = attn_module.forward
    n_head = attn_module.n_head
    n_kv_head = attn_module.n_kv_head
    head_dim = attn_module.head_dim

    def patched(x, ve, cos_sin, window_size, kv_cache):
        with torch.no_grad():
            B, T, _ = x.size()

            # below copied from nanochat gpt.py
            q = attn_module.c_q(x).view(B, T, n_head, head_dim)
            k = attn_module.c_k(x).view(B, T, n_kv_head, head_dim)
            v = attn_module.c_v(x).view(B, T, n_kv_head, head_dim)
            if ve is not None:
                ve_view = ve.view(B, T, n_kv_head, head_dim)
                gate = 3 * torch.sigmoid(
                    attn_module.ve_gate(x[..., :attn_module.ve_gate_channels])
                )
                v = v + gate.unsqueeze(-1) * ve_view
            cos, sin = cos_sin
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)
            q = q * 1.2
            k = k * 1.2


            Q = q.transpose(1, 2)
            K = k.transpose(1, 2)
            V = v.transpose(1, 2)

            # need K and V to have n_head heads, so replicate for dims to line up
            if n_kv_head < n_head:
                repeat = n_head // n_kv_head
                K = K.repeat_interleave(repeat, dim=1)
                V = V.repeat_interleave(repeat, dim=1)

            # casting to float for numerical fidelity
            Q_f, K_f, V_f = Q.float(), K.float(), V.float()

            # need full attention score matrix for figure
            scores = (Q_f @ K_f.transpose(-2, -1) / math.sqrt(head_dim))
            scores = scores.masked_fill(create_attention_mask(T, window_size, x.device), float('-inf'))
            A = F.softmax(scores, dim=-1)

            # calculate first metric (avg parwise cos similarity vetween value vectors)
            V_norm = F.normalize(V_f, dim=-1)
            s = V_norm.sum(dim=-2)
            metric_1 = (((s * s).sum(-1) - T) / (T * (T - 1))).mean().item()
            
            # second metric (diagonal attn)
            metric_2 = A.diagonal(dim1 = -2, dim2 = -1).mean().item()

            # third metric, attention similarity bias
            Y = A @ V_f
            Y_norm = F.normalize(Y, dim=-1)
            metric_3 = (Y_norm * V_norm).sum(dim=-1).mean().item()

            # accumlate per layer metrics
            sums[layer_idx]['M1'] += metric_1
            sums[layer_idx]['M2'] += metric_2
            sums[layer_idx]['M3'] += metric_3

        return original_forward(x, ve, cos_sin, window_size, kv_cache)

    return patched



def measure(model, val_loader, n_batches, device):
    n_layers = len(model.transformer.h)
    sums = [{'M1': 0.0, 'M2': 0.0, 'M3': 0.0} for _ in range (n_layers)]
    for i, block in enumerate(model.transformer.h):
        block.attn.forward = create_patched_forward(block.attn, i, sums)

    model.eval()
    with torch.no_grad():
        for batch_idx, (idx, _) in enumerate(val_loader):
            if batch_idx >= n_batches:
                break
            model(idx.to(device))
            print(f"  BATCH {batch_idx + 1} / {n_batches} DONE")

    results = {}
    for i in range(n_layers):
        results[i] = {
            'M1': sums[i]['M1'] / n_batches,
            'M2': sums[i]['M2'] / n_batches,
            'M3': sums[i]['M3'] / n_batches
        }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, required=True)
    args = parser.parse_args()

    outdir = OUTPUT_DIR / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== LOADING BASE CHECKPOINTS/{args.tag} ===")
    model, tokenizer, _ = load_model('base', DEVICE, phase='eval', model_tag=args.tag)
    validation_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, BATCH_SIZE, SEQ_LEN, split='val', device=DEVICE)

    print(f"\n === MEASURING ON {N_BATCHES} BATCHES ===")
    results = measure(model, validation_loader, N_BATCHES, DEVICE)

    out_json = outdir / 'diagnostics.json'
    with open(out_json, 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nSaved {out_json}")

    layers = sorted(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = (
        r'Metric 1: $\langle v_i, v_j \rangle$ (i<j)',
        r'Metric 2: $a_{i,i}$ (diagonal attention)',
        r'Metric 3: $\langle y_i, v_i \rangle$ (similarity bias)',
    )
    for ax, m, title in zip(axes, ('M1', 'M2', 'M3'), titles):
        ax.plot(layers, [results[l][m] for l in layers], 'o-')
        ax.set_xlabel('Layer index')
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle(args.tag)
    fig.tight_layout()
    out_png = outdir / 'diagnostics.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved {out_png}")

if __name__ == '__main__':                                                                                                                                                            
    main()