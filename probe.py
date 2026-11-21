"""Diagnostic build: report what the bot received in `hello`, then fail on purpose.

Only the bot's own inputs are reported. Lines that are not JSON are echoed by the
checker (first 80 characters each), and stderr goes to the checker's log.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def flat(value, path=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from flat(value[key], f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        yield path, json.dumps(value, sort_keys=True, separators=(",", ":"))
    else:
        yield path, value


def main():
    hello = json.loads(sys.stdin.readline())
    cfg = hello.get("config", {})
    with open(os.path.join(HERE, "kitcfg.json")) as f:
        kit = json.load(f)
    digest = hashlib.sha1(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:10]
    mine, ref = dict(flat(cfg)), dict(flat(kit))
    diffs = [f"{k}={mine.get(k)!r}/{ref.get(k)!r}" for k in sorted(set(mine) | set(ref)) if mine.get(k) != ref.get(k)]
    head = f"P slot={hello.get('slot')} seed={hello.get('seed')} cfg={digest} diffs={len(diffs)}"
    files = hello.get("map", {}).get("files")
    lines = [head, f"P proto={hello.get('protocol')} map={files} keys={sorted(cfg)}"]
    lines += [f"P d{i} {d}" for i, d in enumerate(diffs[:40])]
    for line in lines:
        print(line, file=sys.stderr, flush=True)
        for i in range(0, len(line), 76):
            sys.stdout.write("~" + line[i:i + 76] + "\n")
    sys.stdout.write(json.dumps({"type": "ready", "name": f"s{hello.get('slot')} {digest} d{len(diffs)}"[:32]}) + "\n")
    sys.stdout.flush()
    sys.exit(3)


if __name__ == "__main__":
    main()
