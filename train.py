"""Train the generic transformer on WikiText-103."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset
from svd_embeds import get_tokenizer
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model import COMPILE_BACKENDS, GenericTransformer, TransformerConfig


class TokenDataset(Dataset[Tensor]):
    """Contiguous next-token examples cut from one token stream."""

    def __init__(self, tokens: Tensor, seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        self.num_sequences = max(0, (tokens.numel() - 1) // seq_len)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> Tensor:
        start = index * self.seq_len
        return self.tokens[start : start + self.seq_len + 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--embed-path",
        type=Path,
        default=None,
        help="Optional local OpenAI SVD table (default: download/cache from HF)",
    )
    model.add_argument("--d-model", type=int, default=128)
    model.add_argument("--n-heads", type=int, default=8)
    model.add_argument("--n-layers", type=int, default=6)
    model.add_argument("--d-ff", type=int, default=512)
    model.add_argument("--seq-len", type=int, default=512)
    model.add_argument("--dropout", type=float, default=0.1)

    training = parser.add_argument_group("training")
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=0.1)
    training.add_argument("--warmup-steps", type=int, default=500)
    training.add_argument("--grad-accum-steps", type=int, default=1)
    training.add_argument("--max-grad-norm", type=float, default=1.0)
    training.add_argument(
        "--ce-chunk-size",
        type=int,
        default=1024,
        help="Flattened token positions per tied-output loss chunk (0=full logits)",
    )
    training.add_argument("--max-steps", type=int, default=0)
    training.add_argument("--eval-every", type=int, default=500)
    training.add_argument("--eval-batches", type=int, default=50)
    training.add_argument("--save-every", type=int, default=1000)
    training.add_argument("--seed", type=int, default=42)

    system = parser.add_argument_group("system")
    system.add_argument("--device", default="auto")
    system.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    system.add_argument("--compile", action="store_true")
    system.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
        help="torch.compile mode for the token/transformer encoder path",
    )
    system.add_argument(
        "--compile-backend",
        choices=COMPILE_BACKENDS,
        default="auto",
        help="Compiler backend (auto avoids CUDA Inductor when Triton is absent)",
    )
    system.add_argument("--num-workers", type=int, default=2)
    system.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    system.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    system.add_argument("--resume", type=Path)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_wikitext(cache_dir: Path, seq_len: int, split: str) -> TokenDataset:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"wikitext103_{split}_cl100k.pt"
    if cache_path.exists():
        tokens = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        raw = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split=split
        )
        tokenizer = get_tokenizer()
        token_ids: list[int] = []
        for text in tqdm(raw["text"], desc=f"Tokenizing {split}", unit="doc"):
            if text:
                token_ids.extend(tokenizer.encode_ordinary(text))
                token_ids.append(tokenizer.eot_token)
        tokens = torch.tensor(token_ids, dtype=torch.long)
        torch.save(tokens, cache_path)
    print(f"{split}: {tokens.numel():,} cached tokens")
    return TokenDataset(tokens, seq_len)


def make_scheduler(
    optimizer: AdamW, warmup_steps: int, total_steps: int
) -> LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


def autocast_context(device: torch.device, dtype: torch.dtype, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


@torch.no_grad()
def evaluate(
    model: GenericTransformer,
    loader: DataLoader[Tensor],
    device: torch.device,
    dtype: torch.dtype,
    amp_enabled: bool,
    max_batches: int,
    loss_chunk_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    progress_total = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    for chunk in tqdm(
        loader,
        total=progress_total,
        desc="Validating",
        unit="batch",
        leave=False,
    ):
        chunk = chunk.to(device, non_blocking=True)
        with autocast_context(device, dtype, amp_enabled):
            _, loss = model(
                chunk[:, :-1],
                chunk[:, 1:],
                loss_chunk_size=loss_chunk_size,
            )
        assert loss is not None
        total_loss += loss.float().item()
        batches += 1
        if max_batches > 0 and batches >= max_batches:
            break
    model.train()
    return total_loss / max(1, batches)


def save_checkpoint(
    path: Path,
    model: GenericTransformer,
    optimizer: AdamW,
    scheduler: LambdaLR,
    config: TransformerConfig,
    step: int,
    epoch: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": asdict(config),
            "step": step,
            "epoch": epoch,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps <= 0:
        raise ValueError("grad-accum-steps must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]
    amp_enabled = args.dtype != "fp32" and device.type in {"cuda", "cpu"}
    if device.type == "cpu" and dtype == torch.float16:
        amp_enabled = False

    config = TransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    )
    model = GenericTransformer(config, args.embed_path).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = model.num_parameters()
    print(f"device={device} dtype={args.dtype}")
    print(
        f"vocab={model.vocab_size:,} parameters={total:,} total, "
        f"{trainable:,} trainable"
    )

    train_data = load_wikitext(args.cache_dir, args.seq_len, "train")
    val_data = load_wikitext(args.cache_dir, args.seq_len, "validation")
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_data, shuffle=True, drop_last=True, **loader_args
    )
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = args.max_steps or updates_per_epoch * args.epochs
    scheduler = make_scheduler(optimizer, args.warmup_steps, total_steps)

    step = 0
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = int(checkpoint["step"])
        start_epoch = int(checkpoint["epoch"])
        print(f"resumed {args.resume} at step {step}")

    if args.compile:
        try:
            compile_backend = model.compile_encoder(
                mode=args.compile_mode,
                backend=args.compile_backend,
            )
            print(
                f"torch.compile enabled for encoder "
                f"(backend={compile_backend}, mode={args.compile_mode})"
            )
            if args.compile_backend == "auto" and compile_backend == "aot_eager":
                print(
                    "Triton is unavailable for this accelerator; selected "
                    "aot_eager instead of Inductor."
                )
        except Exception as error:
            print(f"torch.compile unavailable; using eager encoder: {error}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    stop = False
    last_epoch = start_epoch

    with tqdm(
        total=total_steps,
        initial=min(step, total_steps),
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    ) as progress:
        for epoch in range(start_epoch, args.epochs):
            last_epoch = epoch
            for batch_index, chunk in enumerate(train_loader):
                chunk = chunk.to(device, non_blocking=True)
                with autocast_context(device, dtype, amp_enabled):
                    _, loss = model(
                        chunk[:, :-1],
                        chunk[:, 1:],
                        loss_chunk_size=args.ce_chunk_size,
                    )
                    assert loss is not None
                    scaled_loss = loss / args.grad_accum_steps
                scaled_loss.backward()

                if (batch_index + 1) % args.grad_accum_steps:
                    continue

                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.update(1)
                progress.set_postfix(
                    epoch=f"{epoch + 1}/{args.epochs}",
                    loss=f"{loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

                if args.eval_every > 0 and step % args.eval_every == 0:
                    val_loss = evaluate(
                        model,
                        val_loader,
                        device,
                        dtype,
                        amp_enabled,
                        args.eval_batches,
                        args.ce_chunk_size,
                    )
                    progress.write(
                        f"step {step}: validation loss={val_loss:.4f} "
                        f"ppl={math.exp(min(val_loss, 20)):.2f}"
                    )

                if args.save_every > 0 and step % args.save_every == 0:
                    save_checkpoint(
                        args.checkpoint_dir / f"step_{step:08d}.pt",
                        model,
                        optimizer,
                        scheduler,
                        config,
                        step,
                        epoch,
                    )

                if args.max_steps > 0 and step >= args.max_steps:
                    stop = True
                    break
            if stop:
                break

    save_checkpoint(
        args.checkpoint_dir / "latest.pt",
        model,
        optimizer,
        scheduler,
        config,
        step,
        last_epoch,
    )
    print(f"finished at step {step}; wrote {args.checkpoint_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
