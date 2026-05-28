import argparse
import timeit
import statistics

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark transformer forward/backward/optimizer step")
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--mode", type=str, default="all",
                   choices=["forward", "forward_backward", "full", "all"],
                   help="Which passes to benchmark")
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    return p.parse_args()


def make_batch(batch_size, context_length, vocab_size, device):
    inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return inputs, targets


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_steps(fn, warmup, steps, device):
    for _ in range(warmup):
        fn()
        sync(device)

    times = []
    for _ in range(steps):
        t0 = timeit.default_timer()
        fn()
        sync(device)
        times.append(timeit.default_timer() - t0)
    return times


def benchmark(args):
    dtype = getattr(torch, args.dtype)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    model.to(device=args.device, dtype=dtype)

    optimizer = AdamW(model.parameters(), lr=1e-3)
    inputs, targets = make_batch(args.batch_size, args.context_length, args.vocab_size, args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Device: {args.device}  dtype: {args.dtype}")
    print(f"Batch: {args.batch_size}  context: {args.context_length}")
    print(f"Warmup steps: {args.warmup_steps}  Measurement steps: {args.steps}")
    print()

    def forward_only():
        model.eval()
        with torch.no_grad():
            model(inputs)

    def forward_backward():
        model.train()
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))
        loss.backward()

    def full_step():
        model.train()
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

    modes = {
        "forward": forward_only,
        "forward_backward": forward_backward,
        "full": full_step,
    }

    run_modes = list(modes.keys()) if args.mode == "all" else [args.mode]

    for mode_name in run_modes:
        fn = modes[mode_name]
        times = time_steps(fn, args.warmup_steps, args.steps, args.device)
        mean_ms = statistics.mean(times) * 1000
        std_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
        print(f"[{mode_name}]  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms  "
              f"(min={min(times)*1000:.2f}ms  max={max(times)*1000:.2f}ms)")


if __name__ == "__main__":
    benchmark(parse_args())