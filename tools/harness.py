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
              f"max {times[-1]:.2f} ms  ffa {b.ffa}  states {states}")
        ffa_check(strat, hello, obs)


def ffa_check(strat, hello, obs):
    """FFA mode: no detection while alive is explained by our kills; once detected, no unprovoked shot at a
    mech before ffa_hold_t, return fire on the mech that is hitting us, and never keep a mortar."""
    botmod.read_strategy = lambda: strat
    b = botmod.Bot(hello)
    for k, o in enumerate(obs[:300]):
        o = copy.deepcopy(o)
        o["tick"], o["alive"] = k, 15 if k < 50 else 13
        o["events"] = [{"type": "kill", "target": 9}] if k == 50 else []
        b.act(o)
        assert not b.ffa or k < 50, "kill we made was taken as FFA evidence"
    assert not b.ffa
    b = botmod.Bot(hello)
    for k, o in enumerate(obs):  # deaths we did not cause after the early window are never evidence
        o = copy.deepcopy(o)
        o["tick"], o["t"] = k, k * 0.1
        o["alive"] = 16 if o["t"] < 20.5 else max(4, 16 - int(o["t"] - 20.5))
        b.act(o)
    assert not b.ffa, "late deaths triggered FFA"
    b = botmod.Bot(hello)
    for k, o in enumerate(obs[:200]):  # one early death plus one at the end of the fade is not enough
        o = copy.deepcopy(o)
        o["tick"], o["t"] = k, k * 0.1
        o["alive"] = 16 - (o["t"] >= 9.0) - (o["t"] >= 19.0)
        b.act(o)
    assert not b.ffa, "one early death triggered FFA"
    b = botmod.Bot(hello)
    for k, o in enumerate(obs[:200]):  # two early deaths we did not cause: FFA, once the second persists
        o = copy.deepcopy(o)
        o["tick"], o["t"] = k, k * 0.1
        o["alive"] = 16 - (o["t"] >= 9.0) - (o["t"] >= 14.0)
        b.act(o)
        assert b.ffa == (o["t"] > 14.05), ("two early deaths", o["t"], b.ffa)
    b = botmod.Bot(hello)
    for o in obs:  # the kit stream (three drone-era deaths by 25 s, one mech shooting us) must stay practice
        b.act(copy.deepcopy(o))
        assert not b.ffa, ("stream triggered FFA", o["t"])
    ev = b.evidence
    print(f"{strat}: kit stream stays practice: hits {len(ev.hits)}, deaths {ev.deaths}, death score "
          f"{ev.death_score:.2f} of {botmod.PARAMS['ffa_death_score']}")
    evidence_check(obs)
    b = botmod.Bot(hello)
    fired, aim = 0, None
    for k, o in enumerate(obs[:400]):
        o = copy.deepcopy(o)
        o["tick"], o["t"], o["alive"] = k, k * 0.1, 16 if k < 5 else 14
        b.ffa = b.ffa or k == 5  # detection has its own check
        me = o["self"]
        me["aim"] = aim or me["aim"]  # the torso follows the command, unlike the recording
        o["mechs"] = [enemy(i, me, 40 + 10 * i, 0.5 * i) for i in range(1, 4)]
        me["weapons"][1] = {"kind": "mortar", "cooldown": 0, "stats": {"speed": 72, "gravity": 22, "cooldown": 2.6}}
        hit = 200 <= k < 260
        o["events"] = [{"type": "damaged", "amount": 20, "weapon": "gun", "bearing": 0.5, "by": 1}] if hit else []
        a = b.act(o)
        valid(a, k)
        aim = a["aim"]
        if k >= 6:
            assert b.ffa and a.get("drop") == 1, a
            fired += hit and any(a["fire"])
            assert k >= 200 or not any(a["fire"]), (k, b.state, a)  # unprovoked: never
    assert fired, "never returned fire"
    print(f"{strat}: ffa check ok (returned fire on {fired} decisions)")
    if "FIGHT" in botmod.PARAMS["strategies"][strat]["states"]:
        ffa_late_check(strat, hello, obs)


def evidence_check(obs):
    """FfaEvidence: a beam or a traced gun bullet from one mech hitting another, confirmed by an HP drop, counts;
    shots near our line, shots by a mech that just hit us, drops our own hits explain, and mortars do not."""
    base = copy.deepcopy(obs[50])
    me = base["self"]
    x, y, z = me["pos"]
    my_id = me["id"]

    def mech(i, dx, dy, hp):
        return {"id": i, "kind": "mech", "pos": [x + dx, y + dy, z], "vel": [0, 0, 0], "aim": [0, 0], "hp": hp,
                "max_hp": 1400, "weapons": ["gun", "laser"], "charging": False}

    def feed(steps, t0=10.0):
        """steps: list of (shot, target hp, extra events). shot: None | 'laser' | 'gun' | 'mortar' | 'laser-me'."""
        ev = botmod.FfaEvidence()
        hit = False
        for k, (shot, hp, extra) in enumerate(steps):
            o = copy.deepcopy(base)
            o["t"], o["alive"] = t0 + 0.1 * k, 16
            sx, sy, tx, ty = 0.0, 150.0, 100.0, 150.0  # shooter 1 north of us, target 2 to its east
            if shot == "laser-me":
                sx, sy, tx, ty = 0.0, 150.0, 0.0, 70.0  # target 2 stands between the shooter and us
            o["mechs"] = [mech(1, sx, sy, 1400), mech(2, tx, ty, hp)]
            o["beams"], o["projectiles"], o["events"] = [], [], list(extra)
            if shot in ("laser", "laser-me"):
                o["beams"] = [{"by": 1, "kind": "laser", "path": [[x + sx, y + sy, z + 7], [x + tx, y + ty, z + 5]]}]
            elif shot in ("gun", "mortar"):
                o["projectiles"] = [{"id": 300 + k, "kind": shot, "pos": [x + tx - 20, y + ty, z + 6],
                                     "vel": [252.0, 0.0, 0.0]}]
            hit = ev.update(o, 0) or hit
        return hit, len(ev.hits)

    full, hurt = 1400, 1300
    two = lambda shot: [(None, full, []), (shot, full, []), (None, hurt, [])] + [(None, hurt, [])] * 12 + \
        [(shot, hurt, []), (None, 1200, []), (None, 1200, [])]
    assert feed(two("laser")) == (True, 2), feed(two("laser"))
    assert feed(two("gun"))[0], feed(two("gun"))
    assert feed(two("mortar")) == (False, 0)
    assert feed(two("laser-me")) == (False, 0)
    us = {"type": "damaged", "by": 1, "amount": 45, "weapon": "gun", "bearing": 1.5}
    assert feed([(s, h, [us]) for s, h, _e in two("laser")]) == (False, 0), "shooter was hitting us"
    ours = {"type": "hit", "target": 2, "amount": 100, "weapon": "gun"}
    assert feed([(s, h, [ours] if k in (2, 16) else []) for k, (s, h, _e) in
                 enumerate(two("gun"))]) == (False, 0), "our own hits explain the drop"
    assert feed([(None, full, [])] + [(None, hurt, [])] * 20) == (False, 0), "an HP drop alone"
    print("ffa evidence check ok")


def ffa_late_run(strat, hello, obs, mechs, hit_by=None, alive=12, zone_out=False, n=120):
    """FFA after the hold: feed n decisions with fixed mechs (dist, angle) and return (bot, states, fire count)."""
    botmod.read_strategy = lambda: strat
    b = botmod.Bot(hello)
    b.ffa = True
    states, fired, aim = {}, 0, None
    for k, o in enumerate(obs[:n]):
        o = copy.deepcopy(o)
        o["tick"], o["t"], o["alive"] = k, 50.0 + k * 0.1, alive
        me = o["self"]
        me["aim"] = aim or me["aim"]
        o["mechs"] = [enemy(i + 1, me, d, ang) for i, (d, ang) in enumerate(mechs)]
        o["events"] = [{"type": "damaged", "amount": 20, "weapon": "gun", "bearing": mechs[hit_by][1],
                        "by": hit_by + 1}] if hit_by is not None else []
        if zone_out:
            me["in_zone"] = False
            o["zone"]["next_radius"] = 20
        a = b.act(o)
        valid(a, k)
        aim = a["aim"]
        states[b.state] = states.get(b.state, 0) + 1
        fired += any(a["fire"])
    return b, states, fired


def ffa_late_check(strat, hello, obs):
    """FFA after the hold: engage a lone mech, decline a duel in a crowd (hide instead), fight back when cornered,
    break off when hit with others near, lapse the crowd rule in the final circle, and always obey the zone."""
    _b, st, f = ffa_late_run(strat, hello, obs, [(80, 0.5)])
    assert st.get("FIGHT", 0) > 60, st
    lone = f
    crowd = [(80, 0.5), (100, 1.2), (120, 2.0)]
    b, st, f = ffa_late_run(strat, hello, obs, crowd)
    assert "FIGHT" not in st and f == 0, (st, f)
    hide = (st, dict(b.ffa_counts))
    _b, st, _f = ffa_late_run(strat, hello, obs, crowd, alive=4)
    assert st.get("FIGHT", 0) > 60, st
    _b, st, f = ffa_late_run(strat, hello, obs, [(15, 0.5)] + crowd[1:], hit_by=0)
    assert st.get("FIGHT", 0) > 60 and f, (st, f)
    _b, st, _f = ffa_late_run(strat, hello, obs, crowd, hit_by=0)
    assert st.get("EVADE", 0) > 60, st
    _b, st, _f = ffa_late_run(strat, hello, obs, crowd, hit_by=0, zone_out=True)
    assert st.get("ESCAPE", 0) > 100, st
    print(f"{strat}: ffa late check ok (lone fired {lone}; crowd {hide[0]} {hide[1]})")



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


if __name__ == "__main__":
    if sys.argv[1:2] == ["walk"]:
        for st in sys.argv[2:] or ["survive", "hide", "aggro"]:
            walk(st)
    else:
        main()
