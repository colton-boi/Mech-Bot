"""Survival-first mech bot. Three layers per decision:
  Memory   (what we know: threats, last enemy sighting, hold spot, roof target)
  Movement (a state machine; each strategy is an ordered priority list of states)
  Gunnery  (torso aim and trigger, independent of the legs, gated by the strategy's fire policy)
plus reflex overlays (shell and laser dodge, stuck recovery, pickups, upgrades)."""
import gc
import math
import os
import random
import sys
import time

import mechfight as mf
from world import World

PARAMS = {
    "name": "cbot",
    "default_strategy": "survive",
    "strategies": {
        # states: priority order, first whose predicate holds wins. fire: any | engaged | attacked
        "survive": {"states": ["ESCAPE", "DODGE", "RETREAT", "FIGHT", "ROTATE", "DROP", "LOOT", "FARM", "HOLD"],
                    "fire": "engaged", "spot": "cover"},
        "hide": {"states": ["ESCAPE", "DODGE", "ROTATE", "HOLD"], "fire": "attacked", "spot": "corner"},
        "roof": {"states": ["ESCAPE", "DODGE", "CLIMB", "PERCH", "ROTATE", "HOLD"], "fire": "attacked", "spot": "cover"},
        "aggro": {"states": ["ESCAPE", "DODGE", "FIGHT", "LOOT", "FARM", "HUNT", "ROTATE", "HOLD"],
                  "fire": "any", "spot": "cover"},
        # standoff kiting: orbit the nearest enemy at a range set by loadout, dodge bullets, hold fire to regen
        # when hurt and unpressed, shoot crates in quiet moments, otherwise sit in the next circle
        "kite": {"states": ["ESCAPE", "DODGE", "KITE", "LOOT", "CRATE", "CENTER"], "fire": "any", "spot": "cover",
                 "weapon_pref": ["laser", "lightning"],
                 "upgrade_pref": ["nanites", "armor", "plating", "laser_damage", "servos", "gun_damage", "gun_rate",
                                  "laser_charge", "lightning_efficiency", "gyros", "reactor", "capacitor",
                                  "lightning_range", "gun_velocity", "gun_mag", "sensors", "jets",
                                  "lightning_chain", "laser_pierce", "mortar_reload", "mortar_splash",
                                  "mortar_velocity"]},
    },
    # zone
    "zone_margin": 30.0, "zone_margin_frac": 0.4, "zone_slack_s": 10.0,
    # health
    "retreat_frac": 0.4, "resume_frac": 0.85, "retreat_gives_up_dist": 25.0, "retreat_memory_s": 20.0,
    "retreat_quiet_fire_s": 2.0,
    "threat_memory_s": 6.0, "face_threat_s": 10.0, "alert_s": 3.0,
    # engagement
    "engage_range": 130.0, "fire_range": 210.0, "fight_min_dist": 40.0, "fight_max_dist": 110.0,
    "aggro_close_dist": 30.0, "strafe_s": (1.0, 2.4), "aim_tol_min": 0.012, "aim_at_me_rad": 0.18,
    "max_los_checks": 5,
    # loot and drops
    "loot_detour": 35.0, "cube_detour": 30.0, "quiet_s": 12.0, "loot_enemy_clear": 80.0,
    "drop_max_dist": 170.0, "drop_enemy_clear": 60.0,
    "farm_range": 75.0, "farm_min_hp": 0.75, "farm_until_t": 200.0,
    "weapon_pref": ["lightning", "laser", "mortar"],
    "weapon_skip": {"aggro": ["mortar"], "kite": ["mortar"]},
    "upgrade_pref": ["plating", "armor", "nanites", "gyros", "gun_rate", "gun_damage", "lightning_efficiency",
                     "laser_damage", "laser_charge", "mortar_reload", "lightning_range", "lightning_chain",
                     "sensors", "servos", "gun_velocity", "gun_mag", "reactor", "capacitor", "jets",
                     "mortar_splash", "mortar_velocity", "laser_pierce"],
    # reflexes
    "shell_margin": 5.0, "laser_dodge_rad": 0.12, "stuck_speed": 1.5, "stuck_decisions": 6, "stuck_jet_s": 0.9,
    # hold spots
    "spot_search_r": 140.0, "spot_w_cover": 8.0, "spot_w_center": 0.35, "spot_w_enemy": 200.0, "spot_reach": 5.0,
    "scan_amp": 0.9, "scan_rate": 0.5,
    # roof
    "roof_min_h": 6.0, "roof_max_h": 42.0, "roof_min_area": 150.0, "roof_max_dist": 280.0, "roof_inset": 8.0,
    "roof_energy_per_m": 1.8, "roof_max_tries": 3,
    # kite
    "kite_engage": 100.0, "kite_close": 55.0, "kite_back": 45.0, "kite_push": 8.0, "kite_side": 22.0,
    "kite_hurt": 0.75, "kite_lightning_dist": 35.0, "kite_hold_hp": 0.62, "kite_hold_clear": 42.0,
    "kite_hold_quiet_s": 2.0, "kite_disengage_hp": 0.72, "kite_crate_clear": 80.0, "kite_crate_walk_clear": 115.0,
    "kite_crate_range": 110.0, "kite_crate_walk": 75.0, "dodge_window_s": 0.6, "dodge_weight": 7.0,
    "dodge_max_proj": 24,
    # timing
    "bfs_budget_s": 0.010,
}

STRATEGY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy.txt")


def read_strategy():
    try:
        with open(STRATEGY_FILE) as f:
            name = f.read().strip().lower()
    except OSError:
        name = ""
    return name if name in PARAMS["strategies"] else PARAMS["default_strategy"]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def unit(dx, dy):
    L = math.hypot(dx, dy)
    return (dx / L, dy / L) if L > 1e-9 else (0.0, 0.0)


def finite(v, default=0.0):
    return v if isinstance(v, (int, float)) and math.isfinite(v) else default


class Ctx:
    """Per-decision facts derived from one observation."""

    def __init__(self, bot, obs):
        P = PARAMS
        self.obs = obs
        me = obs["self"]
        self.me = me
        self.t = obs["t"]
        self.x, self.y, self.z = me["pos"]
        self.vx, self.vy = me["vel"][0], me["vel"][1]
        self.hp_frac = me["hp"] / max(1.0, me["max_hp"])
        self.stats = me.get("stats", {})
        self.muzzle = (self.x, self.y, self.z + self.stats.get("muzzle_height", 7.0))
        self.eye = (self.x, self.y, self.z + self.stats.get("eye_height", 8.0))
        z = obs["zone"]
        margin = min(P["zone_margin"], P["zone_margin_frac"] * z["next_radius"])
        self.region = (z["next_center"][0], z["next_center"][1], max(0.0, z["next_radius"] - margin))
        self.zone = z
        self.in_zone = me.get("in_zone", True)
        self.enemies = []
        self.npcs = []
        for m in obs.get("mechs", []):
            d = math.hypot(m["pos"][0] - self.x, m["pos"][1] - self.y)
            (self.enemies if m.get("kind") == "mech" else self.npcs).append((d, m))
        self.enemies.sort(key=lambda e: e[0])
        self.npcs.sort(key=lambda e: e[0])
        self.ground = bot.world.ground(self.x, self.y)

    def dist_region(self, x=None, y=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        cx, cy, r = self.region
        return math.hypot(x - cx, y - cy) - r


class Bot:
    name = PARAMS["name"]

    def __init__(self, hello):
        t0 = time.perf_counter()
        self.cfg = hello["config"]
        self.slot = hello.get("slot", 0)
        self.rng = random.Random(hello.get("seed", 0))
        self.strategy = read_strategy()
        self.profile = PARAMS["strategies"][self.strategy]
        m = hello.get("map", {})
        self.world = World(m.get("dir", "/arena"), m.get("files", []), self.cfg["mech"])
        self.state = "HOLD"
        self.state_since = 0.0
        self.threat = None          # (bearing, t, attacker_id)
        self.alert = None           # (bearing, t) from heard weapon sounds
        self.last_enemy_t = -1e9
        self.last_damage_t = -1e9
        self.last_dealt_t = -1e9
        self.spot = None            # (x, y, watch_yaw)
        self.spot_key = None
        self.roof = None
        self.roof_fail = {}
        self.climb_marked = False
        self.strafe_sign = 1
        self.strafe_until = 0.0
        self.slow_count = 0
        self.stuck_until = -1.0
        self.stuck_turn = 1
        self.shell_threat = None
        self.laser_threat = None
        self.handlers = {s: getattr(self, "do_" + s.lower()) for s in self.profile["states"]}
        self.wants = {s: getattr(self, "want_" + s.lower()) for s in self.profile["states"]}
        self.timings = []
        self.holding = False
        self.side = 1 if self.slot % 2 == 0 else -1
        gc.collect()
        gc.freeze()
        log(f"[cbot] strategy={self.strategy} map={self.world.nw}x{self.world.nh} roofs={len(self.world.roofs())} "
            f"map_ms={self.world.load_ms:.0f} init_ms={(time.perf_counter() - t0) * 1000:.0f}")

    # ================= main =================
    def act(self, obs):
        t0 = time.perf_counter()
        c = Ctx(self, obs)
        self.remember(c)
        self.reflex_scan(c)
        for s in self.profile["states"]:
            if self.wants[s](c):
                if s != self.state:
                    log(f"[cbot] t={c.t:.1f} {self.state} -> {s} hp={c.hp_frac:.2f}")
                    self.state, self.state_since = s, c.t
                break
        self.jet_hint = False
        move, boost, jump, look = self.handlers[self.state](c)
        if self.jet_hint:
            grounded = c.me.get("grounded", True)
            if grounded and c.me["energy"] > 60 and not self.world.clear_ray(c.x, c.y, move[0], move[1], 6.0):
                jump = boost = True
            elif not grounded and c.me["energy"] > 3:
                boost = True
        move, boost, jump = self.unstick(c, move, boost, jump)
        aim, fire = self.gunnery(c, look)
        act = {"tick": obs["tick"], "move": [round(finite(move[0]), 3), round(finite(move[1]), 3)],
               "boost": bool(boost), "jump": bool(jump), "aim": aim, "fire": fire}
        self.items(c, act)
        self.world.pump(max(0.0, PARAMS["bfs_budget_s"] - (time.perf_counter() - t0)))
        mf.draw.label(f"{self.strategy}:{self.state} hp{c.hp_frac:.2f}")
        dt = (time.perf_counter() - t0) * 1000
        self.timings.append(dt)
        if len(self.timings) % 600 == 0:
            ts = sorted(self.timings)
            log(f"[cbot] act ms p50={ts[len(ts) // 2]:.2f} p99={ts[int(len(ts) * 0.99)]:.2f} max={ts[-1]:.2f}")
        return act

    # ================= memory =================
    def remember(self, c):
        for ev in c.obs.get("events", []):
            et = ev.get("type")
            if et == "damaged" and ev.get("by") is not None:
                self.last_damage_t = c.t
                b = ev.get("bearing")
                by = ev.get("by")
                for _d, m in c.enemies + c.npcs:
                    if m["id"] == by:
                        b = math.atan2(m["pos"][1] - c.y, m["pos"][0] - c.x)
                if b is not None:
                    self.threat = (b, c.t, by)
            elif et == "hit":
                self.last_dealt_t = c.t
            elif et == "take_failed":
                log(f"[cbot] take failed {ev.get('reason')}")
        for s in c.obs.get("sounds", []):
            if s.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet") \
                    and s.get("dist") in ("near", "mid"):
                self.alert = (s["bearing"], c.t)
        if c.enemies:
            self.last_enemy_t = c.t

    def threat_recent(self, c, s):
        return self.threat is not None and c.t - self.threat[1] < s

    def attacker(self, c):
        """The visible mech that hurt us most recently, else None."""
        if not self.threat_recent(c, PARAMS["threat_memory_s"]):
            return None
        for d, m in c.enemies + c.npcs:
            if m["id"] == self.threat[2]:
                return d, m
        return None

    # ================= reflexes =================
    def reflex_scan(self, c):
        self.shell_threat = None
        g = self.cfg["weapons"]["mortar"].get("gravity", 22.0)
        sr = self.cfg["weapons"]["mortar"].get("splash_radius", 7.0) + c.stats.get("radius", 2.5) + PARAMS["shell_margin"]
        for p in c.obs.get("projectiles", []):
            if p.get("kind") != "mortar":
                continue
            land = self.shell_landing(p["pos"], p["vel"], g)
            if land and math.hypot(land[0] - c.x, land[1] - c.y) < sr:
                self.shell_threat = land
        self.laser_threat = None
        for d, m in c.enemies:
            if m.get("charging") and d < 280:
                to_me = math.atan2(c.y - m["pos"][1], c.x - m["pos"][0])
                if abs(wrap(m["aim"][0] - to_me)) < PARAMS["laser_dodge_rad"] + math.atan2(4.0, max(d, 1.0)):
                    self.laser_threat = to_me

    def shell_landing(self, pos, vel, g, dt=0.05, ttl=7.0):
        x, y, z = pos
        vx, vy, vz = vel
        w = self.world
        t = 0.0
        while t < ttl:
            x += vx * dt
            y += vy * dt
            vz -= g * dt
            z += vz * dt
            t += dt
            if vz < 0:
                top = w.roof_top(x, y)
                if z <= w.ground(x, y) or (top > -1e8 and z <= top and z >= top - 3.0):
                    return (x, y, t)
        return None

    def unstick(self, c, move, boost, jump):
        want = math.hypot(move[0], move[1]) > 0.5
        speed = math.hypot(c.vx, c.vy)
        if want and speed < PARAMS["stuck_speed"] and c.me.get("grounded", True) and c.t >= self.stuck_until:
            self.slow_count += 1
        elif speed >= PARAMS["stuck_speed"]:
            self.slow_count = 0
        if self.slow_count >= PARAMS["stuck_decisions"]:
            self.slow_count = 0
            self.stuck_until = c.t + PARAMS["stuck_jet_s"]
            self.stuck_turn = -self.stuck_turn
        if c.t < self.stuck_until and want:
            a = math.atan2(move[1], move[0]) + self.stuck_turn * 0.6
            return (math.cos(a), math.sin(a)), True, True
        return move, boost, jump

    # ================= navigation =================
    def go(self, c, x, y, r=3.0):
        """Unit move toward (x, y): flow field when available, sampled clear headings otherwise."""
        w = self.world
        if c.z - c.ground > 3.0:  # on a roof or in the air: the ground grid does not apply
            return unit(x - c.x, y - c.y)
        w.request_field(x, y, r)
        d = None
        if w.field_key == (int(x // w.cs), int(y // w.cs), int(r // w.cs)):
            d = w.field_dir(c.x, c.y)
            if d == (0.0, 0.0):
                return unit(x - c.x, y - c.y) if math.hypot(x - c.x, y - c.y) > r else (0.0, 0.0)
            if d is None:  # goal is in another walkable component: head straight at it and jet over
                self.jet_hint = True
                return unit(x - c.x, y - c.y)
        if d is None:
            gx, gy = unit(x - c.x, y - c.y)
            d = self.best_heading(c, lambda dx, dy: dx * gx + dy * gy)
        return d

    def best_heading(self, c, score, length=10.0, n=16):
        best, bs = (0.0, 0.0), -1e18
        for k in range(n):
            a = 2 * math.pi * k / n
            dx, dy = math.cos(a), math.sin(a)
            if not self.world.clear_ray(c.x, c.y, dx, dy, length):
                continue
            s = score(dx, dy)
            if s > bs:
                best, bs = (dx, dy), s
        return best

    def region_penalty(self, c, dx, dy, ahead=12.0):
        cx, cy, r = c.region
        out = math.hypot(c.x + dx * ahead - cx, c.y + dy * ahead - cy) - max(r, 10.0)
        zr = c.zone["radius"]
        out_now = math.hypot(c.x + dx * ahead - c.zone["center"][0], c.y + dy * ahead - c.zone["center"][1]) - zr
        return 0.03 * max(0.0, out) + 0.2 * max(0.0, out_now + 15.0)

    def zone_urgent(self, c):
        """Not safely inside the next circle with enough time to walk there."""
        if not c.in_zone:
            return True
        d = c.dist_region()
        if d <= 0:
            return False
        speed = c.stats.get("walk_speed", 11.7)
        time_left = c.zone["shrink_end"] - c.t if c.t >= c.zone["shrink_start"] else c.zone["shrink_start"] - c.t + \
            (c.zone["shrink_end"] - c.zone["shrink_start"])
        return d / speed + PARAMS["zone_slack_s"] > time_left

    # ================= hold spots =================
    def choose_spot(self, c):
        P = PARAMS
        w = self.world
        cx, cy, r = c.region
        ax, ay = (c.x, c.y) if c.dist_region() <= 0 else (cx + (c.x - cx) * r / max(1.0, math.hypot(c.x - cx, c.y - cy)),
                                                         cy + (c.y - cy) * r / max(1.0, math.hypot(c.x - cx, c.y - cy)))
        sr = P["spot_search_r"]
        stride = 2 if sr > 80 else 1
        threats = [m["pos"] for _d, m in c.enemies[:4]]
        best, bs = None, 1e18
        i0, j0 = int((ax - w.rx0) / w.cs), int((ay - w.ry0) / w.cs)
        span = int(sr / w.cs)
        corner = self.profile["spot"] == "corner"
        my_comp = w.component(c.x, c.y)
        for j in range(max(1, j0 - span), min(w.nh - 1, j0 + span + 1), stride):
            for i in range(max(1, i0 - span), min(w.nw - 1, i0 + span + 1), stride):
                cc = j * w.nw + i
                if w.blocked[cc] or (my_comp and w.comp[cc] != my_comp):
                    continue
                x, y = w.ncenter(cc)
                dc = math.hypot(x - cx, y - cy)
                if dc > max(r, 8.0):
                    continue
                cov = w.cover[cc]
                s = math.hypot(x - c.x, y - c.y) + P["spot_w_cover"] * w.cs * (cov - 1) ** 2 * 0.5 + P["spot_w_center"] * dc
                if corner:
                    walls = sum(w.blocked[cc + o] for o in (1, -1, w.nw, -w.nw, w.nw + 1, w.nw - 1, -w.nw + 1, -w.nw - 1))
                    s -= 15.0 * min(walls, 4)
                for tx, ty, _tz in threats:
                    dd = math.hypot(x - tx, y - ty)
                    if dd < 100:
                        s += P["spot_w_enemy"] * (1 - dd / 100)
                if s < bs:
                    best, bs = (x, y), s
        if best is None:
            best = (cx, cy)
        x, y = best
        cc = w.ncell(x, y)
        ox = oy = 0.0
        for di, dj, _ in w.nbr:
            k = cc + dj * w.nw + di
            if 0 <= k < len(w.blocked) and w.blocked[k]:
                ox -= di
                oy -= dj
        watch = math.atan2(oy, ox) if (ox or oy) else math.atan2(cy - y, cx - x)
        self.spot = (x, y, watch)
        self.spot_key = (round(cx), round(cy), round(r))
        log(f"[cbot] t={c.t:.1f} spot ({x:.0f},{y:.0f}) cover={w.cover[cc]} region=({cx:.0f},{cy:.0f},{r:.0f})")

    def ensure_spot(self, c):
        cx, cy, r = c.region
        if self.spot is None or self.spot_key != (round(cx), round(cy), round(r)) \
                or math.hypot(self.spot[0] - cx, self.spot[1] - cy) > max(r, 8.0):
            self.choose_spot(c)

    def watch_yaw(self, c, base):
        return base + PARAMS["scan_amp"] * math.sin(c.t * PARAMS["scan_rate"] + self.slot)

    # ================= roof =================
    def choose_roof(self, c):
        P = PARAMS
        w = self.world
        cx, cy, r = c.region
        best, bs = None, 1e18
        for i, rf in enumerate(w.roofs()):
            if self.roof_fail.get(i, 0) >= P["roof_max_tries"] or rf["area"] < P["roof_min_area"]:
                continue
            x, y = rf["c"]
            h = rf["top"] - w.ground(x, y)
            if not (P["roof_min_h"] <= h <= P["roof_max_h"]):
                continue
            if math.hypot(x - cx, y - cy) > r - 10:
                continue
            d = math.hypot(x - c.x, y - c.y)
            if d > P["roof_max_dist"]:
                continue
            if d < bs:
                best, bs = i, d
        if best is None:
            self.roof = None
            return
        rf = w.roofs()[best]
        ap = w.approach_point(rf, c.x, c.y)
        if ap is None:
            self.roof_fail[best] = P["roof_max_tries"]
            return self.choose_roof(c)
        self.roof = {"i": best, "rf": rf, "ap": ap, "key": c.region}
        log(f"[cbot] t={c.t:.1f} roof {best} top={rf['top']:.1f} at ({rf['c'][0]:.0f},{rf['c'][1]:.0f}) "
            f"approach ({ap[0]:.0f},{ap[1]:.0f})")

    def on_roof(self, c):
        return self.roof is not None and c.me.get("grounded") and abs(c.z - self.roof["rf"]["top"]) < 1.5 \
            and abs(self.world.roof_top(c.x, c.y) - self.roof["rf"]["top"]) < 1.5

    def roof_valid(self, c):
        if self.roof is None or self.roof["key"][:2] != c.region[:2]:
            if self.roof is not None and self.on_roof(c):
                cx, cy, r = c.region
                x, y = self.roof["rf"]["c"]
                if math.hypot(x - cx, y - cy) <= r:
                    self.roof["key"] = c.region
                    return True
            self.choose_roof(c)
        return self.roof is not None

    # ================= state predicates =================
    def want_escape(self, c):
        return self.zone_urgent(c)

    def want_dodge(self, c):
        return self.shell_threat is not None or self.laser_threat is not None

    def want_retreat(self, c):
        P = PARAMS
        window = P["retreat_memory_s"] if self.state == "RETREAT" else P["threat_memory_s"]
        if not self.threat_recent(c, window):
            return False
        att = self.attacker(c)
        if att and att[0] < P["retreat_gives_up_dist"]:
            return False  # cornered: fight it out
        if self.state == "RETREAT":
            return c.hp_frac < P["resume_frac"]
        return c.hp_frac < P["retreat_frac"]

    def fight_target(self, c):
        P = PARAMS
        att = self.attacker(c)
        if att and att[0] < P["fire_range"]:
            return att
        for d, m in c.enemies:
            if d < P["engage_range"] or self.strategy == "aggro":
                return d, m
        return None

    def want_fight(self, c):
        return self.fight_target(c) is not None

    def want_rotate(self, c):
        if self.strategy == "roof" and self.roof is not None and not self.on_roof(c):
            return False
        self.ensure_spot(c)
        return math.hypot(self.spot[0] - c.x, self.spot[1] - c.y) > PARAMS["spot_reach"] * (2.0 if self.state == "HOLD" else 1.0)

    def drop_target(self, c):
        P = PARAMS
        if c.t - self.last_enemy_t < P["quiet_s"]:
            return None
        for d in c.obs.get("drops", []):
            if d.get("holder") not in (None, c.me["id"]):
                continue
            x, y = d["pos"][0], d["pos"][1]
            dist = math.hypot(x - c.x, y - c.y)
            if dist > P["drop_max_dist"] or c.dist_region(x, y) > 0:
                continue
            if dist / c.stats.get("walk_speed", 11.7) + 3 < d.get("lands_in", 0):
                continue
            if any(math.hypot(m["pos"][0] - x, m["pos"][1] - y) < P["drop_enemy_clear"] for _e, m in c.enemies):
                continue
            return d
        return None

    def want_drop(self, c):
        return self.drop_target(c) is not None

    def wanted_weapon(self, kind, c):
        skip = PARAMS["weapon_skip"].get(self.strategy, ())
        pref = self.profile.get("weapon_pref", PARAMS["weapon_pref"])
        if kind not in pref or kind in skip:
            return False
        held = [w["kind"] for w in c.me["weapons"] if w]
        if kind in held:
            return False
        s1 = c.me["weapons"][1] if len(c.me["weapons"]) > 1 else None
        return s1 is None or s1["kind"] in skip or (s1["kind"] in pref and pref.index(kind) < pref.index(s1["kind"]))

    def loot_target(self, c):
        P = PARAMS
        if any(d < P["loot_enemy_clear"] for d, _m in c.enemies):
            return None
        best, bd = None, 1e18
        for p in c.obs.get("pickups", []):
            if not self.wanted_weapon(p["kind"], c):
                continue
            d = math.hypot(p["pos"][0] - c.x, p["pos"][1] - c.y)
            if d < P["loot_detour"] and abs(p["pos"][2] - c.z) < 4 and d < bd:
                best, bd = p["pos"], d
        for cu in c.obs.get("cubes", []):
            d = math.hypot(cu["pos"][0] - c.x, cu["pos"][1] - c.y)
            if d < P["cube_detour"] and abs(cu["pos"][2] - c.z) < 4 and d < bd:
                best, bd = cu["pos"], d
        return best

    def want_loot(self, c):
        return self.loot_target(c) is not None

    def farm_target(self, c):
        P = PARAMS
        if c.hp_frac < P["farm_min_hp"] or c.t > P["farm_until_t"] or c.t - self.last_enemy_t < P["quiet_s"]:
            return None
        for d, m in c.npcs:
            if d < P["farm_range"]:
                return d, m
        return None

    def want_farm(self, c):
        return self.farm_target(c) is not None

    def want_hunt(self, c):
        return True

    def want_climb(self, c):
        return self.roof_valid(c) and not self.on_roof(c)

    def want_perch(self, c):
        return self.roof is not None and self.on_roof(c)

    def want_hold(self, c):
        return True

    # ---- kite ----
    def kite_target(self, c):
        att = self.attacker(c)
        if att and att[1].get("kind") == "mech" and att[0] < PARAMS["fire_range"]:
            return att
        if c.enemies and c.enemies[0][0] < PARAMS["kite_engage"]:
            return c.enemies[0]
        return None

    def want_kite(self, c):
        return self.kite_target(c) is not None

    def crate_goal(self, c):
        P = PARAMS
        if c.enemies and c.enemies[0][0] < P["kite_crate_walk_clear"]:
            return None
        best, bd = None, 1e18
        for cr in c.obs.get("crates", []):
            x, y, z = cr["pos"]
            d = math.hypot(x - c.x, y - c.y)
            if d < P["kite_crate_walk"] and abs(z - c.z) < 5 and c.dist_region(x, y) <= 0 and d < bd:
                best, bd = (x, y), d
        return best

    def want_crate(self, c):
        return self.crate_goal(c) is not None

    def want_center(self, c):
        return True

    # ================= state behaviors: return (move, boost, jump, look_yaw) =================
    def threat_look(self, c, fallback):
        if self.threat_recent(c, PARAMS["face_threat_s"]):
            return self.threat[0]
        if self.alert and c.t - self.alert[1] < PARAMS["alert_s"]:
            return self.alert[0]
        return fallback

    def do_escape(self, c):
        self.ensure_spot(c)
        mv = self.go(c, self.spot[0], self.spot[1], 3.0)
        boost = (not c.in_zone) or c.me["energy"] > 50
        return mv, boost, False, self.threat_look(c, math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else c.me["aim"][0])

    def do_dodge(self, c):
        if self.shell_threat is not None:
            lx, ly, _t = self.shell_threat
            ax, ay = unit(c.x - lx, c.y - ly)
            if ax == 0 and ay == 0:
                ax, ay = math.cos(c.me["aim"][0] + 1.57), math.sin(c.me["aim"][0] + 1.57)
        else:
            b = self.laser_threat
            ax, ay = math.cos(b + math.pi / 2) * self.strafe_sign, math.sin(b + math.pi / 2) * self.strafe_sign
        mv = self.best_heading(c, lambda dx, dy: dx * ax + dy * ay - self.region_penalty(c, dx, dy), length=8.0)
        return mv, True, False, self.threat_look(c, c.me["aim"][0])

    def do_retreat(self, c):
        b = self.threat[0]
        ax, ay = -math.cos(b), -math.sin(b)
        w = self.world

        def score(dx, dy):
            nc = w.ncell(c.x + dx * 12, c.y + dy * 12)
            cov = w.cover[nc] if nc >= 0 else 15
            return dx * ax + dy * ay + 0.08 * (3 - min(cov, 6)) - self.region_penalty(c, dx, dy)

        mv = self.best_heading(c, score)
        return mv, c.me["energy"] > 25, False, b

    def do_fight(self, c):
        P = PARAMS
        tgt = self.fight_target(c)
        d, m = tgt
        bx, by = unit(m["pos"][0] - c.x, m["pos"][1] - c.y)
        if c.t > self.strafe_until:
            self.strafe_sign = self.rng.choice((-1, 1))
            self.strafe_until = c.t + self.rng.uniform(*P["strafe_s"])
        px, py = -by * self.strafe_sign, bx * self.strafe_sign
        radial = 0.0
        lo, hi = P["fight_min_dist"], P["fight_max_dist"]
        if "lightning" in (m.get("weapons") or []):
            lo = max(lo, 55.0)
        if self.strategy == "aggro":
            lo, hi = 10.0, P["aggro_close_dist"]
        if d < lo:
            radial = -1.0
        elif d > hi:
            radial = 1.0 if self.strategy == "aggro" else 0.0
        if self.strategy == "aggro" and d > hi:
            mv = self.go(c, round(m["pos"][0] / 20) * 20, round(m["pos"][1] / 20) * 20, hi * 0.8)
            return mv, c.me["energy"] > 40, False, math.atan2(by, bx)
        gx, gy = px + radial * bx, py + radial * by
        mv = self.best_heading(c, lambda dx, dy: dx * gx + dy * gy - self.region_penalty(c, dx, dy), length=8.0)
        return mv, False, False, math.atan2(by, bx)

    def kite_heading(self, c, gx, gy, hurt):
        """Heading score: progress toward the goal, zone edge, enemy repulsion, bullet near-misses."""
        P = PARAMS
        w = self.world
        speed = c.stats.get("walk_speed", 11.7)
        radius = c.stats.get("radius", 2.5)
        height = c.stats.get("height", 9.0)
        ux, uy = unit(gx - c.x, gy - c.y)
        dgoal = math.hypot(gx - c.x, gy - c.y)
        reach = min(1.0, dgoal / 7.0)
        rep_r, rep_k = (90.0, 0.55) if hurt else (40.0, 0.3)
        enemies = [m["pos"] for _d, m in c.enemies[:6]]
        projs = []
        for p in c.obs.get("projectiles", []):
            if p.get("kind") == "mortar":
                continue
            pp, pv = p["pos"], p["vel"]
            if abs(pp[0] - c.x) > 200 or abs(pp[1] - c.y) > 200:
                continue
            projs.append((pp, pv))
            if len(projs) >= P["dodge_max_proj"]:
                break
        zc, zr = c.zone["center"], c.zone["radius"]
        best, bs = (0.0, 0.0), -1e18
        win = P["dodge_window_s"]
        for k in range(16):
            a = 2 * math.pi * k / 16
            dx, dy = math.cos(a), math.sin(a)
            if not w.clear_ray(c.x, c.y, dx, dy, 8.0):
                continue
            qx, qy = c.x + dx * 7, c.y + dy * 7
            s = 21.0 * reach * (dx * ux + dy * uy)
            s -= 12.0 * max(0.0, math.hypot(qx - zc[0], qy - zc[1]) - zr + 8.0)
            s -= 20.0 * self.region_penalty(c, dx, dy)
            for ex, ey, _ez in enemies:
                s -= max(0.0, rep_r - math.hypot(qx - ex, qy - ey)) * rep_k
            vx, vy = dx * speed, dy * speed
            for pp, pv in projs:
                rx, ry = pp[0] - c.x, pp[1] - c.y
                wx, wy = pv[0] - vx, pv[1] - vy
                den = wx * wx + wy * wy
                tau = max(0.0, min(win, -(rx * wx + ry * wy) / den)) if den > 1e-9 else 0.0
                miss = math.hypot(rx + wx * tau, ry + wy * tau)
                if abs(pp[2] + pv[2] * tau - (c.z + height * 0.5)) < height * 0.7:
                    s -= max(0.0, radius + 4.0 - miss) * P["dodge_weight"]
            if s > bs:
                best, bs = (dx, dy), s
        return best

    def do_kite(self, c):
        P = PARAMS
        d, m = self.kite_target(c)
        ex, ey = m["pos"][0], m["pos"][1]
        ax, ay = unit(c.x - ex, c.y - ey)
        if ax == 0 and ay == 0:
            ax, ay = 1.0, 0.0
        hurt = c.hp_frac < P["kite_hurt"]
        lightning = next((wp for wp in c.me["weapons"] if wp and wp["kind"] == "lightning"), None)
        laser = any(wp and wp["kind"] == "laser" for wp in c.me["weapons"])
        if lightning and not laser and not hurt and c.me["energy"] > 20:
            want = P["kite_lightning_dist"]
        elif hurt or d < P["kite_close"]:
            want = P["kite_back"]
        else:
            want = P["kite_push"]
        side = P["kite_side"] * self.side
        gx, gy = ex + ax * want - ay * side, ey + ay * want + ax * side
        cx, cy, r = c.region
        dg = math.hypot(gx - cx, gy - cy)
        if dg > r and dg > 0:
            gx, gy = cx + (gx - cx) * r / dg, cy + (gy - cy) * r / dg
        mv = self.kite_heading(c, gx, gy, c.hp_frac < 0.65)
        disengage = c.hp_frac < P["kite_disengage_hp"] and d < P["kite_engage"] and c.me["energy"] > 30
        closing = want == P["kite_lightning_dist"] and d > want + 15 and c.me["energy"] > 60
        boost = disengage or closing or not c.in_zone
        return mv, boost, False, math.atan2(ey - c.y, ex - c.x)

    def do_crate(self, c):
        x, y = self.crate_goal(c)
        d = math.hypot(x - c.x, y - c.y)
        mv = (0.0, 0.0) if d < 25 else self.go(c, x, y, 20.0)
        return mv, False, False, math.atan2(y - c.y, x - c.x)

    def do_center(self, c):
        cx, cy, r = c.region
        d = math.hypot(cx - c.x, cy - c.y)
        if d < max(6.0, min(20.0, r * 0.3)):
            mv = (0.0, 0.0)
        else:
            mv = self.go(c, cx, cy, 5.0)
        heading = math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else self.watch_yaw(c, math.atan2(cy - c.y, cx - c.x) if d > 1 else 0.0)
        if c.enemies:
            m = c.enemies[0][1]
            heading = math.atan2(m["pos"][1] - c.y, m["pos"][0] - c.x)
        boost = self.zone_urgent(c) or (c.me["energy"] > 70 and d > 80)
        return mv, boost, False, self.threat_look(c, heading)

    def do_rotate(self, c):
        self.ensure_spot(c)
        x, y, watch = self.spot
        mv = self.go(c, x, y, 3.0)
        boost = self.zone_urgent(c) or (c.me["energy"] > 70 and math.hypot(x - c.x, y - c.y) > 60)
        heading = math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else watch
        return mv, boost, False, self.threat_look(c, heading)

    def do_hold(self, c):
        self.ensure_spot(c)
        x, y, watch = self.spot
        d = math.hypot(x - c.x, y - c.y)
        mv = (0.0, 0.0) if d < 2.0 else self.go(c, x, y, 2.0)
        if d < 4.0:
            mv = (mv[0] * 0.3, mv[1] * 0.3)  # under step_speed: silent
        return mv, False, False, self.threat_look(c, self.watch_yaw(c, watch))

    def do_drop(self, c):
        d = self.drop_target(c)
        x, y = d["pos"][0], d["pos"][1]
        dist = math.hypot(x - c.x, y - c.y)
        mv = (0.0, 0.0) if dist < d.get("radius", 8) * 0.5 else self.go(c, x, y, 2.0)
        return mv, dist > 40 and c.me["energy"] > 60, False, self.threat_look(c, self.watch_yaw(c, c.me["aim"][0]))

    def do_loot(self, c):
        x, y, _z = self.loot_target(c)
        mv = unit(x - c.x, y - c.y) if math.hypot(x - c.x, y - c.y) < 10 else self.go(c, x, y, 1.5)
        return mv, False, False, self.threat_look(c, math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else c.me["aim"][0])

    def do_farm(self, c):
        d, m = self.farm_target(c)
        b = math.atan2(m["pos"][1] - c.y, m["pos"][0] - c.x)
        mv = (0.0, 0.0)
        if d > 60:
            mv = self.go(c, m["pos"][0], m["pos"][1], 50.0)
        return mv, False, False, b

    def do_hunt(self, c):
        if c.enemies:
            m = c.enemies[0][1]
            mv = self.go(c, round(m["pos"][0] / 20) * 20, round(m["pos"][1] / 20) * 20, 20.0)
            return mv, c.me["energy"] > 60, False, math.atan2(m["pos"][1] - c.y, m["pos"][0] - c.x)
        if self.alert and c.t - self.alert[1] < 8:
            b = self.alert[0]
            mv = self.best_heading(c, lambda dx, dy: dx * math.cos(b) + dy * math.sin(b) - self.region_penalty(c, dx, dy))
            return mv, False, False, b
        return self.do_rotate(c)

    def do_climb(self, c):
        rf, ap = self.roof["rf"], self.roof["ap"]
        qx, qy, ex, ey = ap
        tx, ty = rf["c"]
        need = (rf["top"] - c.ground) * PARAMS["roof_energy_per_m"] + 10
        d_ap = math.hypot(qx - c.x, qy - c.y)
        airborne = not c.me.get("grounded", True)
        near_wall = math.hypot(ex - c.x, ey - c.y) < 9.0
        if airborne and c.z > c.ground + 1.0:
            # in the climb: jet until the feet clear the roof, then move over it
            mv = unit(tx - c.x, ty - c.y)
            clear = c.z > rf["top"] + 0.4
            if not clear and c.me["energy"] < 2 and not self.climb_marked:
                self.climb_marked = True
                self.climb_failed(c)
            return mv if clear or near_wall else unit(ex - c.x, ey - c.y), not clear, False, math.atan2(ty - c.y, tx - c.x)
        self.climb_marked = False
        if self.roof is None:
            return (0.0, 0.0), False, False, self.threat_look(c, c.me["aim"][0])
        if d_ap > 4.0 and not near_wall:
            mv = self.go(c, qx, qy, 2.0)
            return mv, False, False, self.threat_look(c, math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else 0.0)
        if c.me["energy"] < need:
            return (0.0, 0.0), False, False, self.threat_look(c, math.atan2(qy - ty, qx - tx))
        return unit(ex - c.x, ey - c.y), True, True, math.atan2(ty - c.y, tx - c.x)

    def climb_failed(self, c):
        i = self.roof["i"]
        self.roof_fail[i] = self.roof_fail.get(i, 0) + 1
        log(f"[cbot] t={c.t:.1f} climb attempt failed on roof {i} ({self.roof_fail[i]})")
        if self.roof_fail[i] >= PARAMS["roof_max_tries"]:
            self.roof = None

    def do_perch(self, c):
        rf = self.roof["rf"]
        tx, ty = rf["c"]
        d = math.hypot(tx - c.x, ty - c.y)
        mv = (0.0, 0.0) if d < 3.0 else unit(tx - c.x, ty - c.y)
        if d < 6.0:
            mv = (mv[0] * 0.3, mv[1] * 0.3)
        cx, cy, _r = c.region
        return mv, False, False, self.threat_look(c, self.watch_yaw(c, math.atan2(c.y - cy, c.x - cx)))

    # ================= gunnery =================
    def kite_hold(self, c):
        P = PARAMS
        nd = c.enemies[0][0] if c.enemies else 1e9
        hold = c.hp_frac < P["kite_hold_hp"] and nd > P["kite_hold_clear"] and c.t - self.last_damage_t > P["kite_hold_quiet_s"]
        if hold != self.holding:
            log(f"[cbot] t={c.t:.1f} holdfire {'on' if hold else 'off'} hp={c.hp_frac:.2f} nd={nd:.0f}")
            self.holding = hold
        return hold

    def pick_target_kite(self, c):
        P = PARAMS
        if self.kite_hold(c):
            return None
        cands = []
        for d, m in c.enemies + c.npcs:
            if d < P["fire_range"]:
                cands.append((d + 0.025 * m.get("hp", 1000) + (30 if m.get("kind") != "mech" else 0), d, m))
        if not c.enemies or c.enemies[0][0] > P["kite_crate_clear"]:
            for cr in c.obs.get("crates", []):
                d = math.hypot(cr["pos"][0] - c.x, cr["pos"][1] - c.y)
                if d < P["kite_crate_range"]:
                    cands.append((d - 10, d, {"id": cr["id"], "kind": "crate", "pos": cr["pos"], "vel": [0.0, 0.0, 0.0],
                                              "hp": cr.get("hp", 300)}))
        cands.sort(key=lambda e: e[0])
        for _s, d, m in cands[:P["max_los_checks"]]:
            k = m.get("kind")
            center = (m["pos"][0], m["pos"][1], m["pos"][2] + (4.5 if k == "mech" else 1.75 if k == "npc" else 2.0))
            if self.world.los(c.muzzle, center):
                return d, m, center
        return None

    def pick_target(self, c):
        P = PARAMS
        if self.strategy == "kite":
            return self.pick_target_kite(c)
        policy = self.profile["fire"]
        attacked = self.threat_recent(c, P["threat_memory_s"])
        if policy == "attacked" and not attacked:
            return None
        if self.state == "RETREAT" and c.t - self.last_damage_t > P["retreat_quiet_fire_s"]:
            return None  # once the hits stop, dealing damage would only reset the regen clock
        att = self.attacker(c)
        cands = []
        if att:
            cands.append(att)
        cands += [e for e in c.enemies if not att or e[1]["id"] != att[1]["id"]]
        if self.state == "FARM":
            ft = self.farm_target(c)
            if ft:
                cands.insert(0, ft)
        checks = 0
        for d, m in cands:
            if d > P["fire_range"]:
                continue
            if policy == "engaged" and m.get("kind") == "mech" and not (att and att[1]["id"] == m["id"]) \
                    and self.state not in ("FIGHT", "FARM"):
                to_me = math.atan2(c.y - m["pos"][1], c.x - m["pos"][0])
                if d > P["engage_range"] and abs(wrap(m["aim"][0] - to_me)) > P["aim_at_me_rad"]:
                    continue
            if checks >= P["max_los_checks"]:
                break
            checks += 1
            center = (m["pos"][0], m["pos"][1], m["pos"][2] + (4.5 if m.get("kind") == "mech" else 1.75))
            if self.world.los(c.muzzle, center):
                return d, m, center
        return None

    def gunnery(self, c, look):
        P = PARAMS
        aim_yaw, aim_pitch = look, 0.0
        fire = [False, False]
        weapons = c.me["weapons"]
        tgt = self.pick_target(c)
        laser_slot = next((i for i, w in enumerate(weapons) if w and w["kind"] == "laser"), None)
        if tgt:
            d, m, center = tgt
            mx, my, mz = c.muzzle
            kinds = {w["kind"]: i for i, w in enumerate(weapons) if w}
            mortar = kinds.get("mortar")
            mort_ready = mortar is not None and weapons[mortar]["cooldown"] <= 0.05 and 30 < d < 220
            if mort_ready:
                sol = self.mortar_solution(c, weapons[mortar]["stats"], m)
                if sol:
                    aim_yaw, aim_pitch = sol
                    if self.aimed(c, aim_yaw, aim_pitch, 0.03):
                        fire[mortar] = True
                    return [round(aim_yaw, 4), round(aim_pitch, 4)], fire
            gun = kinds.get("gun")
            speed = weapons[gun]["stats"].get("speed", 252.0) if gun is not None else 252.0
            px, py, pz = center
            vx, vy, vz = m["vel"]
            tt = d / speed
            for _ in range(2):
                ax, ay, az = px + vx * tt, py + vy * tt, pz + vz * tt * 0.5
                tt = math.sqrt((ax - mx) ** 2 + (ay - my) ** 2 + (az - mz) ** 2) / speed
            aim_yaw = math.atan2(ay - my, ax - mx)
            aim_pitch = math.atan2(az - mz, math.hypot(ax - mx, ay - my))
            tol = max(P["aim_tol_min"], math.atan2(2.0, max(d, 1.0)))
            on = self.aimed(c, aim_yaw, aim_pitch, tol)
            if gun is not None:
                gw = weapons[gun]
                if on and gw.get("ammo", 1) > 0 and gw.get("reload", 0) <= 0 and d < gw["stats"].get("range", 230):
                    fire[gun] = True
            lt = kinds.get("lightning")
            if lt is not None:
                ls = weapons[lt]["stats"]
                direct = math.atan2(py - my, px - mx)
                if d < ls.get("range", 45) * 0.95 and abs(wrap(c.me["aim"][0] - direct)) < math.radians(ls.get("cone_deg", 30)) * 0.4 \
                        and c.me["energy"] > 10:
                    fire[lt] = True
            if laser_slot is not None:
                lw = weapons[laser_slot]
                direct_yaw = math.atan2(py - my, px - mx)
                direct_pitch = math.atan2(pz - mz, math.hypot(px - mx, py - my))
                charged = lw.get("charge", 0) >= lw["stats"].get("charge", 0.75) * 0.95
                if lw.get("charge", 0) > 0 or lw["cooldown"] <= 0:
                    aim_yaw, aim_pitch = direct_yaw, direct_pitch
                    fire[laser_slot] = not (charged and self.aimed(c, direct_yaw, direct_pitch, tol))
                    if fire[laser_slot] and gun is not None:
                        fire[gun] = fire[gun] and self.aimed(c, direct_yaw, direct_pitch, tol)
        elif laser_slot is not None and weapons[laser_slot].get("charge", 0) > 0:
            fire[laser_slot] = False
        mp = c.stats.get("max_pitch", 1.3)
        np_ = c.stats.get("min_pitch", -0.7)
        aim_pitch = min(mp, max(np_, finite(aim_pitch)))
        return [round(wrap(finite(aim_yaw)), 4), round(aim_pitch, 4)], fire

    def aimed(self, c, yaw, pitch, tol):
        ay, ap = c.me["aim"]
        return abs(wrap(ay - yaw)) < tol and abs(ap - pitch) < max(tol, 0.03)

    def mortar_solution(self, c, st, m):
        v = st.get("speed", 72.0)
        g = st.get("gravity", 22.0)
        mx, my = c.x, c.y
        mz = c.z + c.stats.get("height", 9.0) + 0.5
        tx, ty, tz = m["pos"]
        vx, vy = m["vel"][0], m["vel"][1]
        high = not self.world.los(c.muzzle, (tx, ty, tz + 4.5))
        sol = None
        t_f = 1.0
        for _ in range(2):
            px, py = tx + vx * t_f, ty + vy * t_f
            d = math.hypot(px - mx, py - my)
            dz = tz - mz
            disc = v ** 4 - g * (g * d * d + 2 * dz * v * v)
            if disc < 0 or d < 1:
                return None
            root = math.sqrt(disc)
            th = math.atan((v * v + root) / (g * d)) if high else math.atan((v * v - root) / (g * d))
            if th > c.stats.get("max_pitch", 1.3):
                th = math.atan((v * v - root) / (g * d))
            t_f = d / max(1e-3, v * math.cos(th))
            sol = (math.atan2(py - my, px - mx), th)
        return sol

    # ================= items =================
    def items(self, c, act):
        P = PARAMS
        offer = c.obs.get("offer")
        if offer and offer.get("choices"):
            ids = [ch["id"] for ch in offer["choices"]]
            pref = self.profile.get("upgrade_pref", P["upgrade_pref"])
            act["pick"] = min(range(len(ids)), key=lambda i: pref.index(ids[i]) if ids[i] in pref else 99)
        reach = self.cfg["loot"].get("pickup_radius", 4.0)
        weapons = c.me["weapons"]
        for p in c.obs.get("pickups", []):
            if math.hypot(p["pos"][0] - c.x, p["pos"][1] - c.y) < reach and abs(p["pos"][2] - c.z) < 4 \
                    and self.wanted_weapon(p["kind"], c):
                act["take"] = int(p["id"])
                s1 = weapons[1] if len(weapons) > 1 else None
                if s1 is not None:
                    if s1["kind"] == "laser" and s1.get("charge", 0) > 0:
                        act.pop("take")
                        continue
                    act["drop"] = 1
                break
        if "take" not in act:
            skip = PARAMS["weapon_skip"].get(self.strategy, ())
            for i, w in enumerate(weapons):
                if w and w["kind"] in skip:
                    act["drop"] = i
                    break
        gun = weapons[0] if weapons and weapons[0] and weapons[0]["kind"] == "gun" else None
        if gun and not act["fire"][0] and gun.get("reload", 0) <= 0 \
                and gun.get("ammo", 99) < gun["stats"].get("mag", 12) * 0.5 and (not c.enemies or self.holding):
            act["reload"] = True


if __name__ == "__main__":
    mf.run(Bot)
