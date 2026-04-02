"""
src/train.py

Training loop for SignTransformer with CTC loss.
Includes checkpointing, learning rate scheduling, and logging.

Usage:
    python src/train.py --vocab 2000 --epochs 50 --d_model 128 --n_layers 3
    python src/train.py --vocab 2000 --epochs 100 --teacher
"""

import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import CHECKPOINT_DIR
from src.dataloader import get_dataloader
from src.model import build_student_model, build_teacher_model, make_padding_mask
from src.evaluate import evaluate


def train_one_epoch(model, loader, optimizer, ctc_loss, device, vocab_size):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        kpts    = batch["keypoints"].to(device)
        labels  = batch["label"].to(device).squeeze(1)
        in_lens = batch["input_length"].to(device)
        lb_lens = batch["label_length"].to(device)

        mask     = make_padding_mask(in_lens, max_len=kpts.size(1)).to(device)
        log_prob = model(kpts, src_key_padding_mask=mask)  # (T, B, C)

        loss = ctc_loss(log_prob, labels, in_lens, lb_lens)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def main(args):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = get_dataloader("train", vocab_size=args.vocab,
                                  batch_size=args.batch_size, num_workers=args.workers)
    val_loader   = get_dataloader("val",   vocab_size=args.vocab,
                                  batch_size=args.batch_size, num_workers=args.workers)

    if args.teacher:
        model    = build_teacher_model(n_classes=args.vocab)
        run_name = f"teacher_v{args.vocab}"
    else:
        model    = build_student_model(n_classes=args.vocab)
        run_name = f"student_d{args.d_model}_l{args.n_layers}_v{args.vocab}"

    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    ctc_loss  = nn.CTCLoss(blank=args.vocab, reduction="mean", zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_top1 = 0.0
    history   = []

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     ctc_loss, device, args.vocab)
        top1, top5 = evaluate(model, val_loader, device, args.vocab)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Top-1: {top1:.3f} | Top-5: {top5:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed:.1f}s")

        history.append({"epoch": epoch, "loss": train_loss, "top1": top1, "top5": top5})

        if top1 > best_top1:
            best_top1 = top1
            ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "top1":        top1,
                "top5":        top5,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  ✓ New best: {top1:.3f} → saved to {ckpt_path}")

    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best Top-1: {best_top1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SignTransformer on WLASL")
    parser.add_argument("--vocab",      type=int,   default=2000)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--n_layers",   type=int,   default=3)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--teacher",    action="store_true")
    args = parser.parse_args()
    main(args)
