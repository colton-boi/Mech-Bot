"""Drive Bot.act directly on the kit's recorded stream with injected situations, so exceptions surface
(mf.run swallows them) and rarely-hit states run. Usage: python3 tools/harness.py [strategy ...]"""
import copy
import gzip
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BOT_DIR = os.path.dirname(HERE)
KIT = os.path.join(os.path.dirname(BOT_DIR), "mechfight-kit")
sys.path.insert(0, BOT_DIR)

import bot as botmod  # noqa: E402


def load_stream():
    with gzip.open(os.path.join(KIT, "check", "stream.jsonl.gz"), "rt") as f:
        msgs = [json.loads(line) for line in f if line.strip()]
    hello = msgs[0]
    hello["map"]["dir"] = os.path.join(KIT, "arena")
    return hello, [m for m in msgs if m.get("type") == "obs"]


def enemy(i, me, dist, ang, weapons=("gun",), charging=False, aim_at_me=True):
    x, y, z = me["pos"]
    ex, ey = x + dist * math.cos(ang), y + dist * math.sin(ang)
    yaw = math.atan2(y - ey, x - ex) if aim_at_me else 0.0
    return {"id": i, "kind": "mech", "pos": [ex, ey, z], "vel": [3.0, -2.0, 0.0], "aim": [yaw, 0.0],
            "hp": 900, "max_hp": 1400, "cubes": 0, "weapons": list(weapons), "charging": charging}


def scenarios(o, k):
    """Return a mutated copy of observation o; k picks the scenario."""
    o = copy.deepcopy(o)
    me = o["self"]
    x, y, z = me["pos"]
    s = k % 9
    if s == 1:
        o["mechs"] = [enemy(i, me, 30 + 12 * i, 0.4 * i) for i in range(1, 16)]
        o["events"] = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.4, "by": 1, "rear": True}]
    elif s == 2:
        o["projectiles"] = [{"id": 99, "kind": "mortar", "pos": [x - 40, y, z + 30], "vel": [20.0, 0.0, 0.0]}]
    elif s == 3:
        me["hp"] = 300
        o["mechs"] = [enemy(4, me, 70, 1.0)]
        o["events"] = [{"type": "damaged", "amount": 90, "weapon": "gun", "bearing": 1.0, "by": 4}]
    elif s == 4:
        o["offer"] = {"id": 1, "expires": o["t"] + 5, "choices": [
            {"id": "sensors", "name": "", "desc": ""}, {"id": "armor", "name": "", "desc": ""},
            {"id": "gun_rate", "name": "", "desc": ""}]}
        o["pickups"] = [{"id": 8100, "kind": "lightning", "pos": [x + 1, y + 1, z]}]
    elif s == 5:
        zc = o["zone"]["next_center"]
        o["drops"] = [{"id": 20000, "pos": [zc[0], zc[1], 0.0], "radius": 8, "lands_in": 0, "capture_s": 4,
                       "progress": 0, "holder": None}]
        o["t"] = max(o["t"], 100.0)
    elif s == 6:
        me["in_zone"] = False
        o["zone"]["next_radius"] = 20
    elif s == 7:
        o["mechs"] = [enemy(5, me, 120, -2.0, ("gun", "laser"), charging=True)]
        me["weapons"][1] = {"kind": "mortar", "cooldown": 0, "stats": {"speed": 72, "gravity": 22, "cooldown": 2.6}}
    elif s == 8:
        o["mechs"] = [{"id": 101, "kind": "npc", "pos": [x + 40, y, z], "vel": [0, 0, 0], "aim": [0, 0],
                       "hp": 350, "max_hp": 350}]
        me["weapons"][1] = {"kind": "laser", "cooldown": 0, "charge": 0.8,
                            "stats": {"charge": 0.75, "cooldown": 1.65, "range": 280}}
    return o


def valid(a, tick):
    assert a["tick"] == tick
    assert all(math.isfinite(v) and abs(v) <= 1.0001 for v in a["move"]), a
    assert len(a["aim"]) == 2 and all(math.isfinite(v) for v in a["aim"]), a
    assert isinstance(a["fire"], list) and len(a["fire"]) == 2
    for key in ("pick", "take", "drop"):
        if key in a:
            assert isinstance(a[key], int), (key, a)
    json.dumps(a, allow_nan=False)


def main():
    strategies = sys.argv[1:] or list(botmod.PARAMS["strategies"])
    hello, obs = load_stream()
    for strat in strategies:
        botmod.read_strategy = lambda s=strat: s
        b = botmod.Bot(hello)
        times, states = [], {}
        for k, o in enumerate(obs * 3):
            o = scenarios(o, k)
            o["tick"] = k
            t0 = time.perf_counter()
            a = b.act(o)
            times.append((time.perf_counter() - t0) * 1000)
            valid(a, k)
            states[b.state] = states.get(b.state, 0) + 1
        times.sort()
        print(f"{strat}: p50 {times[len(times) // 2]:.2f} p99 {times[int(len(times) * 0.99)]:.2f} "
              f"max {times[-1]:.2f} ms  states {states}")



def walk(strat, decisions=1500):
    """Kinematic walk: feed the bot its own moves (walls block per the raster) from the stream's spawn,
    with a static zone, and report whether it ends inside the target region and which states it used."""
    hello, obs = load_stream()
    botmod.read_strategy = lambda: strat
    b = botmod.Bot(hello)
    w = b.world
    o = copy.deepcopy(obs[0])
    o["mechs"], o["projectiles"], o["sounds"], o["events"] = [], [], [], []
    x, y, _ = o["self"]["pos"]
    trail = {}
    for k in range(decisions):
        o["tick"], o["t"] = k, k * 0.1
        o["self"]["pos"] = [x, y, w.ground(x, y)]
        a = b.act(o)
        trail[b.state] = trail.get(b.state, 0) + 1
        sp = 19.5 if a["boost"] else 11.7
        nx, ny = x + a["move"][0] * sp * 0.1, y + a["move"][1] * sp * 0.1
        for cx, cy in ((nx, ny), (nx, y), (x, ny)):
            cell = w.rcell(cx, cy)
            if cell >= 0 and not w.rwall[cell]:
                nx, ny = cx, cy
                break
        else:
            nx, ny = x, y
        o["self"]["vel"] = [(nx - x) * 10, (ny - y) * 10, 0]
        x, y = nx, ny
    zc, zr = o["zone"]["next_center"], o["zone"]["next_radius"]
    print(f"walk {strat}: end ({x:.0f},{y:.0f}) dist_to_next_center {math.hypot(x - zc[0], y - zc[1]):.0f} "
          f"(radius {zr}) spot {b.spot and tuple(round(v) for v in b.spot[:2])} states {trail}")


def skip_check():
    """Aggro skips a mortar at its feet and drops a held one; survive still takes it."""
    hello, obs = load_stream()
    mortar = {"kind": "mortar", "cooldown": 0, "stats": {"speed": 72, "gravity": 22, "cooldown": 2.6}}

    def run(strat, held, pickup):
        botmod.read_strategy = lambda: strat
        b = botmod.Bot(hello)
        o = copy.deepcopy(obs[0])
        o["mechs"], o["projectiles"], o["events"] = [], [], []
        x, y, z = o["self"]["pos"]
        o["self"]["weapons"][1] = held
        o["pickups"] = [{"id": 8100, "kind": pickup, "pos": [x + 1, y + 1, z]}] if pickup else []
        a = b.act(o)
        return a.get("take"), a.get("drop")

    cases = [("aggro", None, "mortar", (None, None)), ("aggro", mortar, None, (None, 1)),
             ("aggro", mortar, "laser", (8100, 1)), ("survive", None, "mortar", (8100, None))]
    for strat, held, pickup, want in cases:
        got = run(strat, copy.deepcopy(held), pickup)
        print(f"skip {strat} held={held and held['kind']} pickup={pickup}: take/drop {got} "
              f"{'PASS' if got == want else 'FAIL want ' + str(want)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["walk"]:
        for st in sys.argv[2:] or ["survive", "hide", "aggro"]:
            walk(st)
    elif sys.argv[1:2] == ["skip"]:
        skip_check()
    else:
        main()
