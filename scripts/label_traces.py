"""Label a directory of traces into a corpus manifest, without re-running anything.

Separate from `build_corpus.py` because generating runs and labelling them are different jobs with
different failure modes, and coupling them means a labelling change costs a full regeneration.
Given the traces are on disk, labelling is seconds.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from testbed.corpus.runner import ENV_DEFAULTS
from testbed.health import healthy_band, random_policy_return, run_health
from testbed.label import label_run
from testbed.telemetry import read_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="directory of .jsonl.gz traces")
    ap.add_argument("--out", required=True, help="directory to write manifest.jsonl into")
    ap.add_argument("--min-controls", type=int, default=2)
    args = ap.parse_args()

    traces = sorted(Path(args.traces).glob("*.jsonl.gz"))
    if not traces:
        raise SystemExit(f"no traces in {args.traces}")

    by_cell: dict[str, list[Path]] = defaultdict(list)
    for p in traces:
        try:
            meta = read_meta(p)
        except Exception as exc:  # truncated or partial
            print(f"skipping unreadable {p.name}: {exc}")
            continue
        spec = meta["spec"]
        by_cell[f"{spec['env_name']}/{spec['algo']}"].append(p)

    manifest: list[dict] = []
    for cell, paths in sorted(by_cell.items()):
        env_name, _algo = cell.split("/")
        controls = []
        for p in paths:
            if "__P0_control__" not in p.name:
                continue
            try:
                controls.append(run_health(p).windowed)
            except (EOFError, OSError, ValueError):
                print(f"  skipping unreadable control {p.name}")
        if len(controls) < args.min_controls:
            print(f"{cell}: only {len(controls)} controls, skipping cell")
            continue
        band = healthy_band(controls, q=5.0)
        rand = random_policy_return(env_name, ENV_DEFAULTS[env_name], n_episodes=20)
        lift = band.mean - rand
        span = max(abs(rand), abs(band.mean), 1e-9)
        usable = lift > 0.1 * span
        print(
            f"{cell}: n_control={band.n} mean={band.mean:.3f} random={rand:.3f} "
            f"floor={band.threshold:.3f} {'' if usable else '[UNUSABLE: control never learned]'}"
        )
        if not usable:
            continue

        for p in sorted(paths):
            try:
                lab = label_run(p, floor=band.threshold)
            except (EOFError, OSError, ValueError):
                continue
            manifest.append(
                {
                    "run_id": lab.run_id,
                    "path": str(p.resolve()),
                    "cell": cell,
                    "env": lab.env_name,
                    "algo": lab.algo,
                    "pathology": lab.pathology,
                    "family": lab.pathology.split("@", 1)[0],
                    "seed": lab.seed,
                    "failed": lab.failed,
                    "windowed_eval": lab.windowed,
                    "degrade_step": lab.degrade_step,
                    "floor": band.threshold,
                }
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, sort_keys=True) for m in manifest) + "\n"
    )

    n_failed = sum(1 for m in manifest if m["failed"])
    print(f"\n{len(manifest)} runs labelled, {n_failed} failed")
    by_family: dict[str, list[bool]] = defaultdict(list)
    for m in manifest:
        by_family[m["family"]].append(m["failed"])
    for fam, flags in sorted(by_family.items()):
        print(f"  {fam:<26} {sum(flags):>3}/{len(flags):<3} failed")
    print(f"wrote {out/'manifest.jsonl'}")


if __name__ == "__main__":
    main()
