import argparse
import timeit
import statistics

import torch
import torch.nn.functional as F
import torch.nn as nn

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        out_fc1 = self.relu(self.fc1(x))
        out_ln = self.ln(out_fc1)
        x = self.fc2(out_ln)
        return x


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark transformer forward/backward/optimizer step")
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=4096)
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

    with torch.autocast(device_type="cuda", dtype=dtype):
    #     model = ToyModel(args.d_model, args.vocab_size)
        model.to(device=args.device)

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
        torch.cuda.memory._record_memory_history(max_entries=1000)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type="cuda", dtype=dtype):
            fn = modes[mode_name]
            times = time_steps(fn, args.warmup_steps, args.steps, args.device)
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_mib = peak_bytes / (1024 * 1024)
            print(f"[{mode_name}]  peak memory: {peak_mib:.2f} MiB")
            mean_ms = statistics.mean(times) * 1000
            std_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
            print(f"[{mode_name}]  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms  "
                  f"(min={min(times)*1000:.2f}ms  max={max(times)*1000:.2f}ms)")
            torch.cuda.memory._dump_snapshot(f"memory_snapshot_{mode_name}.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)

if __name__ == "__main__":
    benchmark(parse_args())