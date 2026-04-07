"""
src/evaluate.py

Evaluation logic shared by train.py and standalone eval runs.
Person 3 (Jian) adds WER computation here when the CTC decoder is extended.
"""

import torch
from tqdm import tqdm

from src.decode import greedy_decode
from src.model import make_padding_mask


@torch.no_grad()
def evaluate(model, loader, device, vocab_size: int) -> tuple[float, float]:
    """
    Run top-1 / top-5 evaluation on a dataloader.

    Returns:
        top1: float — top-1 accuracy
        top5: float — top-5 accuracy
    """
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total        = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        kpts    = batch["keypoints"].to(device)
        labels  = batch["label"].to(device).squeeze(1)
        in_lens = batch["input_length"].to(device)

        mask     = make_padding_mask(in_lens, max_len=kpts.size(1)).to(device)
        log_prob = model(kpts, src_key_padding_mask=mask)  # (T, B, C)

        preds_seq = log_prob.argmax(dim=-1).permute(1, 0)  # (B, T)
        preds     = greedy_decode(preds_seq, blank=vocab_size)

        for i, pred_seq in enumerate(preds):
            if len(pred_seq) == 0:
                continue

            if pred_seq[0] == labels[i].item():
                correct_top1 += 1

            T_i      = in_lens[i].item()
            avg_prob = log_prob[:T_i, i, :].mean(dim=0)  # (C,)
            top5     = avg_prob.topk(5).indices.tolist()
            if labels[i].item() in top5:
                correct_top5 += 1

            total += 1

    top1 = correct_top1 / max(total, 1)
    top5 = correct_top5 / max(total, 1)
    return top1, top5
