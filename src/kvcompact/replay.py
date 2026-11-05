import argparse
import json
from pathlib import Path

from kvcompact.engine import Engine
from kvcompact.eval import score
from kvcompact.methods import Full, Recent, SnapKV, SummaryTail
from kvcompact.sessions import Transcript


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("sessions", nargs="+", type=Path)
    args.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args.add_argument("--device", default="cpu")
    args.add_argument("--limit", type=int, default=32_768)
    args.add_argument("--budget", type=int, default=4_096)
    args.add_argument("--tail", type=int, default=1_024)
    args.add_argument("--continuation", type=int, default=4_096)
    args.add_argument("--methods", default="full,recent,summary+tail,snapkv")
    args.add_argument("--out", type=Path, default=Path("runs/replay.jsonl"))
    opts = args.parse_args()

    engine = Engine(opts.model, opts.device)
    methods = {m.name: m for m in (Full(), Recent(), SummaryTail(opts.tail), SnapKV())}
    chosen = [methods[m] for m in opts.methods.split(",")]
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    with opts.out.open("a") as out:
        for path in opts.sessions:
            rendered = Transcript.from_claude_code(path).render(engine.tokenizer)
            prefix, continuation = rendered.cut(opts.limit, opts.continuation)
            session = engine.prefill(prefix.ids)
            for method in chosen:
                result = score(engine, path.stem, method.name, method.compact(engine, session, prefix, opts.budget), continuation, opts.budget)
                out.write(json.dumps({"session": result.session, "method": result.method, "slots": result.slots,
                                      "nll": result.nll.flatten().tolist()}) + "\n")
                out.flush()
                print(f"{path.stem} {method.name:>14} slots={result.slots:>6} nll={result.mean_nll:.4f}")


if __name__ == "__main__":
    main()
