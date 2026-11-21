"""Replay mechreplay/1 recordings through the bot's FFA detector, one mech's view at a time.

Each frame is turned into the observation that mech would have received: mechs and drones it can see (200 m in a
120 deg wedge around its aim or 45 m all round, with line of sight, kept 5 s once seen), visible gun bullets and
shells that are not its own (velocity from the neighbouring frames), visible beams, and its own hit/damaged/kill
events. Then bot.FfaEvidence decides. Prints the share of lives detected by 10/15/20/30/45 s and the median time,
next to the v2 rule (2 unexplained deaths by 22 s, else 3).

usage: python3 tools/ffa_replay.py FILE.jsonl.gz ... [--title Showdown] [--until 45]
"""
import gzip
import json
import math
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bot as botmod  # noqa: E402
from world import World  # noqa: E402

MAP = os.environ.get("MECHFIGHT_MAP", os.path.join(HERE, "..", "..", "mechfight-kit", "arena"))
W = None
botmod.log = lambda *a: None
botmod.PARAMS.update(json.loads(os.environ.get("FFA_PARAMS", "{}")))


def world():
    global W
    if W is None:
        W = World(MAP, ["v2/world.json"], {"step_up": 1.2, "height": 9.0, "max_slope_deg": 50})
    return W


def load(path):
    rows = []
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                rows.append(json.loads(line))
    except (OSError, EOFError, ValueError):
        pass
    title = rows[0].get("title", "") if rows else ""
    hdr = next((o for o in rows if o.get("type") == "header"), None)
    frames = [o for o in rows if "m" in o]
    return title, hdr, frames


def sees(v, p):
    """v: replay row of the viewer; p: (x, y, z) of the target's centre."""
    dx, dy = p[0] - v[1], p[1] - v[2]
    d = math.hypot(dx, dy)
    if d > 200.0:
        return False
    if d > 45.0 and abs(botmod.wrap(math.atan2(dy, dx) - v[4])) > math.radians(60):
        return False
    eye = (v[1], v[2], v[3] + 8.0)
    w = world()
    return w.los(eye, p) or w.los(eye, (p[0], p[1], p[2] + 4.5))


def run_file(path, until_s=45.0):
    title, hdr, frames = load(path)
    if hdr is None or not frames:
        return None
    n = len(frames)
    stop = next((k for k, f in enumerate(frames) if f["t"] / 20.0 > until_s), n)
    # projectile tracks: id -> {frame index: pos}; first position gives the owner
    tracks = {}
    for k in range(min(n, stop + 1)):
        for p in frames[k].get("p", []):
            tracks.setdefault(p[0], {})[k] = (p[2], p[3], p[4])
    owner = {}
    for pid, tr in tracks.items():
        k0 = min(tr)
        x, y, _z = tr[k0]
        best, bd = None, 12.0
        for r in frames[k0]["m"]:
            d = math.hypot(r[1] - x, r[2] - y)
            if d < bd:
                best, bd = r[0], d
        owner[pid] = best

    def vel(pid, k):
        tr = tracks[pid]
        if k + 1 in tr:
            a, b, dt = tr[k], tr[k + 1], 0.1
        elif k - 1 in tr:
            a, b, dt = tr[k - 1], tr[k], 0.1
        else:
            return None
        return [(b[i] - a[i]) / dt for i in range(3)]

    ids = [r[0] for r in frames[0]["m"]]
    out = []
    for me in ids:
        det = botmod.FfaEvidence()
        seen_t = {}
        kills = 0
        hit_t = None
        old_t, streak, alive0 = None, 0, None
        truth_mm = None  # first real mech-on-mech hit (not involving us) anywhere, for reference
        udeaths = []  # times of deaths we did not cause
        vis = []  # real mech-on-mech hits (not involving us) whose target we could see: the ceiling
        for k in range(stop):
            f = frames[k]
            t = f["t"] / 20.0
            rows = {r[0]: r for r in f["m"]}
            v = rows.get(me)
            if v is None or v[6] <= 0:
                break
            alive = sum(1 for r in f["m"] if r[6] > 0)
            events, beams = [], []
            for e in f.get("ev", []):
                et = e.get("e")
                if et == "hit" and e.get("src") == me:
                    events.append({"type": "hit", "target": e.get("dst"), "amount": e.get("amt", 0), "weapon": e.get("w")})
                elif et == "hit" and e.get("dst") == me:
                    events.append({"type": "damaged", "by": e.get("src"), "amount": e.get("amt", 0), "weapon": e.get("w")})
                elif et == "hit" and truth_mm is None and e.get("src") in rows and e.get("dst") in rows:
                    truth_mm = t
                elif et == "death" and e.get("by") == me and e.get("id") in ids:
                    kills += 1
                    events.append({"type": "kill", "target": e.get("id")})
                elif et in ("laser", "zap") and e.get("id") != me:
                    bp = [e["from"], e["to"]] if et == "laser" else e.get("path") or []
                    if len(bp) >= 2:
                        mid = [(bp[0][i] + bp[1][i]) / 2 for i in range(3)]
                        if any(sees(v, q) for q in (bp[0], bp[-1], mid)):
                            beams.append({"by": e["id"], "kind": "laser" if et == "laser" else "lightning", "path": bp})
            for e in f.get("ev", []):
                if e.get("e") == "hit" and e.get("src") in rows and e.get("dst") in rows and me not in (e["src"], e["dst"]) \
                        and e["src"] != e["dst"] and e["dst"] in ids:
                    d = rows[e["dst"]]
                    if sees(v, (d[1], d[2], d[3] + 4.5)):
                        vis.append((t, e["src"], e["dst"], e.get("w")))
            mechs = []
            for r in f["m"]:
                if r[0] == me or r[6] <= 0:
                    continue
                if sees(v, (r[1], r[2], r[3] + 4.5)):
                    seen_t[r[0]] = t
                if t - seen_t.get(r[0], -1e9) <= 5.0:
                    mechs.append({"id": r[0], "kind": "mech", "pos": [r[1], r[2], r[3]], "hp": r[6], "max_hp": r[10]})
            for r in f.get("n", []):
                if r[5] > 0 and sees(v, (r[1], r[2], r[3] + 1.0)):
                    seen_t[r[0]] = t
                if t - seen_t.get(r[0], -1e9) <= 5.0 and r[5] > 0:
                    mechs.append({"id": r[0], "kind": "npc", "pos": [r[1], r[2], r[3]], "hp": r[5]})
            projs = []
            for p in f.get("p", []):
                if owner.get(p[0]) == me:
                    continue
                pos = (p[2], p[3], p[4])
                vv = vel(p[0], k)
                if vv is None or not sees(v, pos):
                    continue
                projs.append({"id": p[0], "kind": ("gun", "mortar", "npc")[p[1]], "pos": list(pos), "vel": vv})
            obs = {"t": t, "tick": f["t"], "alive": alive, "self": {"id": me, "pos": [v[1], v[2], v[3]]},
                   "mechs": mechs, "projectiles": projs, "beams": beams, "events": events}
            if hit_t is None and det.update(obs, kills):
                hit_t = t
            if alive0 is None:
                alive0 = alive
            need = 2 if t <= 22.0 else 3
            while len(udeaths) < alive0 - alive - kills:
                udeaths.append(t)
            streak = streak + 1 if alive0 - alive - kills >= need else 0
            if old_t is None and streak >= 2:
                old_t = t
        out.append({"me": me, "new": hit_t, "old": old_t, "hits": [list(h) for h in det.hits], "truth": truth_mm, "vis": vis, "deaths": udeaths})
    return {"file": os.path.basename(path), "title": title, "seed": hdr.get("seed"), "lives": out}


def main():
    args = sys.argv[1:]
    title = "Showdown"
    until = 45.0
    if "--title" in args:
        i = args.index("--title")
        title = args[i + 1]
        del args[i:i + 2]
    if "--until" in args:
        i = args.index("--until")
        until = float(args[i + 1])
        del args[i:i + 2]
    files = []
    seeds = set()
    for p in args:
        t, hdr, _f = load(p)
        if title in t and hdr is not None and hdr.get("seed") not in seeds:
            seeds.add(hdr.get("seed"))
            files.append(p)
    with Pool() as pool:
        res = [r for r in pool.starmap(run_file, [(p, until) for p in files]) if r]
    lives = [L for r in res for L in r["lives"]]
    N = len(lives)

    def share(key, s):
        return sum(1 for L in lives if L[key] is not None and L[key] <= s) / max(1, N)

    def med(key):
        xs = sorted(L[key] for L in lives if L[key] is not None)
        return xs[len(xs) // 2] if xs else None

    print(f"matches {len(res)} lives {N}")
    for key in ("new", "old"):
        print(f"{key}: detected by 10/15/20/30/45 s " +
              " ".join(f"{share(key, s):.2f}" for s in (10, 15, 20, 30, 45)) + f"  median {med(key)}")
    both = [min(x for x in (L["new"], L["old"]) if x is not None) if (L["new"] or L["old"]) else None for L in lives]
    print("either: " + " ".join(f"{sum(1 for x in both if x is not None and x <= s) / max(1, N):.2f}"
                               for s in (10, 15, 20, 30, 45)))
    if os.environ.get("FFA_DUMP"):
        with open(os.environ["FFA_DUMP"], "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
