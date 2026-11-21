"""cand_tempo_drop: Supply drop prediction & cover-assisted contest + fast 1/3/6/10 cube
upgrade snowball + 3.0-4.8s anti-linear boost-weave strafe + crossfire FLEE break-off.
Forked strictly from our wt-push-squad/ (`squad`) codebase."""
import gc
import math
import os
import random
import sys
import time

import mechfight as mf
from world import World

PARAMS = {
    "name": "cand_tempo_drop",
    "default_strategy": "cand_tempo_drop",
    "strategies": {
        "survive": {"states": ["ESCAPE", "DODGE", "RETREAT", "FIGHT", "ROTATE", "DROP", "LOOT", "FARM", "HOLD"],
                    "fire": "engaged", "spot": "cover"},
        "hide": {"states": ["ESCAPE", "DODGE", "ROTATE", "HOLD"], "fire": "attacked", "spot": "corner"},
        "roof": {"states": ["ESCAPE", "DODGE", "CLIMB", "PERCH", "ROTATE", "HOLD"], "fire": "attacked", "spot": "cover"},
        "aggro": {"states": ["ESCAPE", "DODGE", "FIGHT", "LOOT", "FARM", "HUNT", "ROTATE", "HOLD"],
                  "fire": "any", "spot": "cover"},
        # High-tempo drop contester & cube/upgrade snowballer with anti-linear 3.0-4.8s boost-weave
        "cand_tempo_drop": {
            "states": ["ESCAPE", "DODGE", "FLEE", "ROOFTAKE", "DROP", "DUEL", "SEEK", "HARVEST", "SCOUT", "ROTATE", "HOLD"],
            "fire": "any",
            "spot": "cover",
            "weapon_pref": ["laser", "lightning"],
            "upgrade_pref_laser": [
                "plating", "armor", "laser_damage", "laser_charge", "nanites", "gun_damage",
                "gun_rate", "lightning_efficiency", "lightning_range", "gyros", "gun_mag",
                "lightning_chain", "sensors", "servos", "gun_velocity", "reactor", "capacitor",
                "jets", "laser_pierce",
            ],
        },
        "squad": {
            "states": ["ESCAPE", "DODGE", "FLEE", "ROOFTAKE", "DROP", "DUEL", "SEEK", "HARVEST", "SCOUT", "ROTATE", "HOLD"],
            "fire": "any",
            "spot": "cover",
            "weapon_pref": ["laser", "lightning"],
            "upgrade_pref_laser": [
                "plating", "armor", "laser_damage", "laser_charge", "nanites", "gun_damage",
                "gun_rate", "lightning_efficiency", "lightning_range", "gyros", "gun_mag",
                "lightning_chain", "sensors", "servos", "gun_velocity", "reactor", "capacitor",
                "jets", "laser_pierce",
            ],
        },
    },
    # squad: engagement triggers
    "sq_cap": 240.0, "sq_near": 60.0, "sq_hit_s": 6.0, "sq_close_v": 3.0, "sq_close_d": 180.0,
    "sq_weak_frac": 0.40, "sq_weak_d": 160.0, "sq_aim_rad": 0.22, "sq_aim_d": 190.0, "sq_stick_s": 2.0,
    # squad: target score
    "sq_hp_w": 0.2, "sq_low_frac": 0.30, "sq_low_bonus": 100.0, "sq_stick_bonus": 120.0, "sq_hit_bonus": 40.0,
    "sq_close_bonus": 15.0,
    # squad: fight distances
    "sq_r_laser": 120.0, "sq_r_lightning": 28.0, "sq_lt_energy": 12.0, "sq_r_gun": 85.0, "sq_r_vs_lightning": 64.0,
    "sq_r_vs_mortar": 80.0, "sq_r_npc": 58.0, "sq_r_finish": 14.0, "sq_finish_tgt": 0.38, "sq_finish_me": 0.42,
    "sq_finish_view": 2,
    # anti-linear 3.0-4.8s irregular boost-weave strafe
    "sq_ring": 6.0, "sq_w_range": 1.35, "sq_w_exposed": 30.0, "sq_exposed_d": 170.0, "sq_w_los": 45.0,
    "sq_w_cube": 1.5, "sq_cube_pull_d": 40.0, "sq_w_strafe": 4.0, "sq_strafe_s": (3.0, 4.8), "sq_others": 3,
    # crossfire break-off
    "sq_lose_hold_s": 1.5, "sq_lose_d": 160.0, "sq_lose2_hp": 0.56, "sq_lose3_hp": 0.80, "sq_lose_hp": 0.35,
    "sq_lose_gap": 0.10, "sq_press_frac": 0.25, "sq_press_near": 2, "sq_trapped_d": 20.0, "sq_cover_s": 0.5,
    "sq_crossfire_s": 4.0, "sq_drop_ban_s": 6.0,
    # fast cube/upgrade snowball + drop contest
    "sq_cube_r": 35.0, "sq_cube_r_hurt": 14.0, "sq_cube_hurt_hp": 0.45,
    "sq_laser_seek": 240.0, "sq_pickup_seek": 150.0,
    "sq_cube_seek": 115.0, "sq_cube_seek_early": 140.0, "sq_cube_seek_cap": 30.0,
    "sq_farm_r": 150.0, "sq_farm_r_early": 165.0, "sq_farm_hp": 0.50, "sq_farm_hp_early": 0.45,
    "sq_early_t": 120.0, "sq_farm_give_up_s": 30.0,
    "sq_farm_ban_s": 60.0, "sq_crate_shoot_d": 28.0, "sq_kill_margin": 0.8,
    "sq_scout_grid": 70.0, "sq_roof_r": 150.0, "sq_roof_r_fight": 45.0, "sq_roof_max_h": 30.0,
    "sq_drop_max_d": 185.0, "sq_drop_hp": 0.50,
    # zone
    "zone_margin": 30.0, "zone_margin_frac": 0.4, "zone_slack_s": 10.0,
    # health
    "retreat_frac": 0.4, "resume_frac": 0.85, "retreat_gives_up_dist": 25.0, "retreat_memory_s": 20.0,
    "retreat_quiet_fire_s": 2.0,
    "threat_memory_s": 6.0, "face_threat_s": 10.0, "alert_s": 3.0,
    # engagement
    "engage_range": 130.0, "fire_range": 210.0, "fight_min_dist": 40.0, "fight_max_dist": 110.0,
    "aggro_close_dist": 18.0, "aggro_min_dist": 8.0, "aggro_boost_energy": 25.0, "aggro_direct_dist": 60.0,
    "lock_bonus": 30.0, "lock_memory_s": 2.0, "attacker_bonus": 15.0, "weak_w": 50.0, "strafe_s": (1.0, 2.4), "aim_tol_min": 0.012, "aim_at_me_rad": 0.18,
    "max_los_checks": 5,
    # loot and drops
    "loot_detour": 35.0, "cube_detour": 30.0, "weapon_detour_empty": 80.0, "fight_loot_r": 18.0,
    "fight_weapon_frac": 0.6, "fight_weapon_cap": 80.0, "fight_weapon_blind": 40.0, "fight_loot_give_up_s": 2.5, "quiet_s": 12.0, "loot_enemy_clear": 80.0,
    "crate_r": 30.0, "crate_min_target": 90.0, "crate_min_hp": 0.6,
    "drop_max_dist": 170.0, "drop_enemy_clear": 60.0,
    "farm_range": 75.0, "farm_min_hp": 0.75, "farm_until_t": 200.0,
    "weapon_pref": ["lightning", "laser", "mortar"],
    "lt_stop_energy": 8.0, "lt_resume_energy": 32.0, "lt_boost_energy": 45.0, "lt_boost_dist": 70.0,
    "upgrade_pref": ["gun_mag", "lightning_efficiency", "plating", "armor", "gun_damage", "gun_rate",
                     "laser_damage", "laser_charge", "nanites", "gyros", "lightning_range", "lightning_chain",
                     "mortar_reload", "sensors", "servos", "gun_velocity", "reactor", "capacitor", "jets",
                     "mortar_splash", "mortar_velocity", "laser_pierce"],
    # reflexes
    "shell_margin": 5.0, "laser_dodge_rad": 0.12, "stuck_speed": 1.5, "stuck_decisions": 6, "stuck_jet_s": 0.9,
    # hold spots
    "spot_search_r": 140.0, "spot_w_cover": 8.0, "spot_w_center": 0.35, "spot_w_enemy": 200.0, "spot_reach": 5.0,
    "scan_amp": 0.9, "scan_rate": 0.5,
    # roof
    "roof_min_h": 6.0, "roof_max_h": 42.0, "roof_min_area": 150.0, "roof_max_dist": 280.0, "roof_inset": 8.0,
    "roof_energy_per_m": 1.8, "roof_max_tries": 3,
    # timing
    "bfs_budget_s": 0.010,
}

STRATEGIES = PARAMS["strategies"]
STRATEGY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy.txt")


def read_strategy():
    try:
        with open(STRATEGY_FILE) as f:
            name = f.read().strip().lower()
    except OSError:
        name = ""
    return name if name in STRATEGIES else PARAMS["default_strategy"]


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
        self.profile = STRATEGIES[self.strategy]
        m = hello.get("map", {})
        self.world = World(m.get("dir", "/arena"), m.get("files", []), self.cfg["mech"])
        self.state = "HOLD"
        self.state_since = 0.0
        self.threat = None
        self.alert = None
        self.last_enemy_t = -1e9
        self.last_damage_t = -1e9
        self.last_dealt_t = -1e9
        self.spot = None
        self.spot_key = None
        self.roof = None
        self.roof_fail = {}
        self.climb_marked = False
        self.strafe_sign = 1 if (self.slot % 2 == 0) else -1
        self.strafe_until = 0.0
        self.slow_count = 0
        self.stuck_until = -1.0
        self.stuck_turn = 1
        self.shell_threat = None
        self.laser_threat = None
        self.lock = None
        self.lock_tick = None
        self.lock_pick = None
        self.fight_loot = None
        self.loot_skip = {}
        self.lt_dry = False
        self.crate_id = None
        self.hit_by = {}
        self.lose_until = -1e9
        self.cover = None
        self.seen_items = {}
        self.job = None
        self.sq_crate = None
        self.sq_spot_log = -1e9
        self.sq_los_t = -1e9
        self.rooftake_tick = self.rooftake_pick = None
        self.scouted = set()
        self.seek_tick = self.seek_pick = None
        self.farm_tick = self.farm_pick = None
        self.drop_tick = self.drop_pick = None
        self.drop_ban_until = -1e9
        self.handlers = {s: getattr(self, "do_" + s.lower()) for s in self.profile["states"]}
        self.wants = {s: getattr(self, "want_" + s.lower()) for s in self.profile["states"]}
        self.timings = []
        gc.collect()
        gc.freeze()
        log(f"[{self.name}] strategy={self.strategy} map={self.world.nw}x{self.world.nh} roofs={len(self.world.roofs())} "
            f"map_ms={self.world.load_ms:.0f} init_ms={(time.perf_counter() - t0) * 1000:.0f}")

    def _is_squad(self):
        return self.strategy in ("squad", "cand_tempo_drop")

    def next_cube_gap(self, c):
        """Cubes remaining to hit next early upgrade breakpoint [1, 3, 6, 10]."""
        cubes = c.me.get("cubes", 0)
        thresholds = self.cfg.get("cubes", {}).get("upgrade_thresholds", [1, 3, 6, 10, 15, 21, 28, 36, 45])
        for th in thresholds:
            if th > 10:
                break
            if cubes < th:
                return th - cubes
        return 99

    # ================= main =================
    def act(self, obs):
        t0 = time.perf_counter()
        c = Ctx(self, obs)
        self.remember(c)
        self.reflex_scan(c)
        for s in self.profile["states"]:
            if self.wants[s](c):
                if s != self.state:
                    log(f"[{self.name}] t={c.t:.1f} {self.state} -> {s} hp={c.hp_frac:.2f}")
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
        mf.draw.label(f"{self.strategy}:{self.state} hp{c.hp_frac:.2f} c{c.me.get('cubes', 0)}")
        dt = (time.perf_counter() - t0) * 1000
        self.timings.append(dt)
        if len(self.timings) % 600 == 0:
            ts = sorted(self.timings)
            log(f"[{self.name}] act ms p50={ts[len(ts) // 2]:.2f} p99={ts[int(len(ts) * 0.99)]:.2f} max={ts[-1]:.2f}")
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
                self.hit_by[by] = c.t
            elif et == "hit":
                self.last_dealt_t = c.t
            elif et in ("pickup", "upgrade", "took", "kill", "drop"):
                log(f"[{self.name}] t={c.t:.1f} {et} {ev.get('kind', ev.get('cubes', ev.get('id', ev.get('target'))))}")
            elif et == "take_failed":
                log(f"[{self.name}] take failed {ev.get('reason')}")
        for s in c.obs.get("sounds", []):
            if s.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet") \
                    and s.get("dist") in ("near", "mid"):
                self.alert = (s["bearing"], c.t)
        if c.enemies:
            self.last_enemy_t = c.t
        if self._is_squad():
            for cr in c.obs.get("crates", []):
                self.seen_items[cr["id"]] = ("crate", cr["pos"])
            for p in c.obs.get("pickups", []):
                self.seen_items[p["id"]] = (p["kind"], p["pos"])
            listed = {it["id"] for it in c.obs.get("crates", [])} | {it["id"] for it in c.obs.get("pickups", [])}
            for iid in [i for i, (_k, pos) in self.seen_items.items()
                        if i not in listed and math.hypot(pos[0] - c.x, pos[1] - c.y) < 20.0]:
                del self.seen_items[iid]

    def threat_recent(self, c, s):
        return self.threat is not None and c.t - self.threat[1] < s

    def attacker(self, c):
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
        w = self.world
        if c.z - c.ground > 3.0:
            return unit(x - c.x, y - c.y)
        w.request_field(x, y, r)
        d = None
        if w.field_key == (int(x // w.cs), int(y // w.cs), int(r // w.cs)):
            d = w.field_dir(c.x, c.y)
            if d == (0.0, 0.0):
                return unit(x - c.x, y - c.y) if math.hypot(x - c.x, y - c.y) > r else (0.0, 0.0)
            if d is None:
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

    def _zone_time_left(self, c):
        se = c.zone.get("shrink_end")
        if not isinstance(se, (int, float)) or not math.isfinite(se):
            return 1e9
        return max(0.0, se - c.t)

    def zone_urgent(self, c):
        if not c.in_zone:
            return True
        d = c.dist_region()
        if d <= 0:
            return False
        speed = max(1.0, c.stats.get("walk_speed", 11.7))
        time_left = self._zone_time_left(c)
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
            return False
        if self.state == "RETREAT":
            return c.hp_frac < P["resume_frac"]
        return c.hp_frac < P["retreat_frac"]

    def aggro_target(self, c):
        if self.lock_tick == c.obs["tick"]:
            return self.lock_pick
        P = PARAMS
        att = self.attacker(c)
        lock_id = self.lock[0] if self.lock else None
        best, bs = None, 1e18
        cands = list(c.enemies)
        if att and all(att[1]["id"] != m["id"] for _d, m in cands):
            cands.append(att)
        for d, m in cands:
            hp, mhp = m.get("hp"), m.get("max_hp")
            frac = hp / mhp if isinstance(hp, (int, float)) and isinstance(mhp, (int, float)) and mhp > 0 else 1.0
            s = d + P["weak_w"] * min(1.0, max(0.0, frac))
            if m["id"] == lock_id:
                s -= P["lock_bonus"]
            if att and m["id"] == att[1]["id"]:
                s -= P["attacker_bonus"]
            if s < bs:
                best, bs = (d, m), s
        if best is not None:
            self.lock = (best[1]["id"], best[1], c.t)
        elif self.lock and c.t - self.lock[2] < P["lock_memory_s"]:
            m = self.lock[1]
            best = (math.hypot(m["pos"][0] - c.x, m["pos"][1] - c.y), m)
        else:
            self.lock = None
        self.lock_tick, self.lock_pick = c.obs["tick"], best
        return best

    def fight_target(self, c):
        P = PARAMS
        if self.strategy == "aggro":
            return self.aggro_target(c)
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

    def _in_crossfire(self, c):
        P = PARAMS
        recent_atts = sum(1 for mid, ht in self.hit_by.items() if mid < 100 and c.t - ht < P["sq_crossfire_s"])
        near = sum(1 for ed, _m in c.enemies if ed < P["sq_lose_d"])
        return recent_atts >= 2 or (near >= 2 and c.hp_frac < 0.62)

    def sq_drop_target(self, c):
        """Supply drop prediction & contest: track lands_in/capture_s/progress/holder."""
        if self.drop_tick == c.obs["tick"]:
            return self.drop_pick
        P = PARAMS
        pick = None
        if c.t >= self.drop_ban_until and c.hp_frac > P["sq_drop_hp"] and c.zone["radius"] > 35.0 and not self._in_crossfire(c):
            walk_speed = max(1.0, c.stats.get("walk_speed", 11.7))
            best, bs = None, 1e18
            for d in c.obs.get("drops", []):
                x, y, _z = d["pos"]
                dist = math.hypot(x - c.x, y - c.y)
                if dist > P["sq_drop_max_d"] or not self.sq_item_ok(c, d["id"], d["pos"]):
                    continue
                lands_in = d.get("lands_in", 0.0)
                walk_eta = dist / walk_speed
                boost_eta = dist / 14.0
                if lands_in > walk_eta + 5.0:
                    continue
                # Don't run past a close duel enemy unless we're already inside the drop ring about to cap
                if any(ed < 45.0 for ed, _m in c.enemies) and not (dist <= d.get("radius", 8.0) + 4.0 and lands_in <= 1.5):
                    continue
                holder = d.get("holder")
                cap_s = d.get("capture_s", 4.0)
                prog = d.get("progress", 0.0)
                if holder not in (None, c.me["id"]) and (cap_s - prog) <= boost_eta + 0.8:
                    continue
                near_drop = sum(1 for _ed, m in c.enemies if math.hypot(m["pos"][0] - x, m["pos"][1] - y) < 55.0)
                if near_drop >= 2 and dist > 25.0 and c.hp_frac < 0.78:
                    continue
                s = boost_eta + 0.5 * lands_in - (25.0 if holder == c.me["id"] else 0.0) + 12.0 * near_drop
                if s < bs:
                    best, bs = d, s
            pick = best
        self.drop_tick, self.drop_pick = c.obs["tick"], pick
        return pick

    def drop_target(self, c):
        if self._is_squad():
            return self.sq_drop_target(c)
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
        if self._is_squad() and c.t < self.drop_ban_until:
            return False
        return self.drop_target(c) is not None

    def wanted_weapon(self, kind, c):
        pref = self.profile.get("weapon_pref", PARAMS["weapon_pref"])
        if kind not in pref:
            return False
        held = [w["kind"] for w in c.me["weapons"] if w]
        if kind in held:
            return False
        s1 = c.me["weapons"][1] if len(c.me["weapons"]) > 1 else None
        return s1 is None or (s1["kind"] in pref and pref.index(kind) < pref.index(s1["kind"]))

    def loot_target(self, c):
        P = PARAMS
        if any(d < P["loot_enemy_clear"] for d, _m in c.enemies):
            return None
        best, bd = None, 1e18
        for p in c.obs.get("pickups", []):
            if not self.wanted_weapon(p["kind"], c):
                continue
            d = math.hypot(p["pos"][0] - c.x, p["pos"][1] - c.y)
            r = P["weapon_detour_empty"] if self.slot1_empty(c) and p["kind"] != "mortar" else P["loot_detour"]
            if d < r and abs(p["pos"][2] - c.z) < 4 and d < bd:
                best, bd = p["pos"], d
        for cu in c.obs.get("cubes", []):
            d = math.hypot(cu["pos"][0] - c.x, cu["pos"][1] - c.y)
            if d < P["cube_detour"] and abs(cu["pos"][2] - c.z) < 4 and d < bd:
                best, bd = cu["pos"], d
        return best

    def toward_region(self, c, x, y):
        return c.dist_region(x, y) <= max(0.0, c.dist_region()) + 5.0

    def slot1_empty(self, c):
        w = c.me["weapons"]
        return len(w) < 2 or w[1] is None

    def fight_loot_target(self, c, d_tgt):
        P = PARAMS
        best, bd = None, 1e18

        def ok(it):
            x, y, z = it["pos"]
            return abs(z - c.z) < 4 and self.loot_skip.get(it["id"], -1.0) < c.t and self.toward_region(c, x, y)

        cube_r = P["cube_detour"] if d_tgt > P["crate_min_target"] else P["fight_loot_r"]
        for cu in c.obs.get("cubes", []):
            d = math.hypot(cu["pos"][0] - c.x, cu["pos"][1] - c.y)
            if d < cube_r and d < bd and ok(cu):
                best, bd = cu, d
        empty = self.slot1_empty(c)
        for p in c.obs.get("pickups", []):
            if p["kind"] == "mortar" or not self.wanted_weapon(p["kind"], c):
                continue
            d = math.hypot(p["pos"][0] - c.x, p["pos"][1] - c.y)
            r = P["fight_loot_r"]
            if empty:
                r = max(r, min(P["fight_weapon_cap"], P["fight_weapon_frac"] * d_tgt))
            if d < r and d < bd and ok(p) and (d <= P["fight_weapon_blind"] or self.world.clear_ray(
                    c.x, c.y, (p["pos"][0] - c.x) / d, (p["pos"][1] - c.y) / d, d)):
                best, bd = p, d
        return best, bd

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

    # ================= state behaviors =================
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
        weave = 0.45 * math.sin(c.t * 2.6 + self.slot * 1.3)
        wx, wy = ax - ay * weave, ay + ax * weave
        w = self.world

        def score(dx, dy):
            nc = w.ncell(c.x + dx * 12, c.y + dy * 12)
            cov = w.cover[nc] if nc >= 0 else 15
            return dx * wx + dy * wy + 0.08 * (3 - min(cov, 6)) - self.region_penalty(c, dx, dy)

        mv = self.best_heading(c, score)
        return mv, c.me["energy"] > 25, False, b

    def do_fight(self, c):
        P = PARAMS
        tgt = self.fight_target(c)
        d, m = tgt
        bx, by = unit(m["pos"][0] - c.x, m["pos"][1] - c.y)
        if c.t > self.strafe_until:
            self.strafe_sign = -self.strafe_sign
            self.strafe_until = c.t + self.rng.uniform(*P["strafe_s"])
        px, py = -by * self.strafe_sign, bx * self.strafe_sign
        radial = 0.0
        lo, hi = P["fight_min_dist"], P["fight_max_dist"]
        if "lightning" in (m.get("weapons") or []):
            lo = max(lo, 55.0)
        if self.strategy == "aggro":
            lo, hi = P["aggro_min_dist"], P["aggro_close_dist"]
        if d < lo:
            radial = -1.0
        elif d > hi:
            radial = 1.0 if self.strategy == "aggro" else 0.0
        gx, gy = px + radial * bx, py + radial * by
        mv = self.best_heading(c, lambda dx, dy: dx * gx + dy * gy - self.region_penalty(c, dx, dy), length=8.0)
        return mv, False, False, math.atan2(by, bx)

    def boost_floor(self, c, d):
        P = PARAMS
        if d < P["lt_boost_dist"] and any(w and w["kind"] == "lightning" for w in c.me["weapons"]):
            return P["lt_boost_energy"]
        return P["aggro_boost_energy"]

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
            mv = (mv[0] * 0.3, mv[1] * 0.3)
        return mv, False, False, self.threat_look(c, self.watch_yaw(c, watch))

    def do_drop(self, c):
        d = self.drop_target(c)
        x, y = d["pos"][0], d["pos"][1]
        dist = math.hypot(x - c.x, y - c.y)
        if not self._is_squad():
            mv = (0.0, 0.0) if dist < d.get("radius", 8) * 0.5 else self.go(c, x, y, 2.0)
            return mv, dist > 40 and c.me["energy"] > 60, False, self.threat_look(c, self.watch_yaw(c, c.me["aim"][0]))

        r = d.get("radius", 8.0)
        lands_in = d.get("lands_in", 0.0)
        # Present front armor toward nearest threat while capturing
        if c.enemies and c.enemies[0][0] < 160.0:
            m0 = c.enemies[0][1]
            look = math.atan2(m0["pos"][1] - c.y, m0["pos"][0] - c.x)
        else:
            look = self.threat_look(c, self.watch_yaw(c, math.atan2(y - c.y, x - c.x) if dist > 2.0 else c.me["aim"][0]))

        if dist > r * 1.05:
            mv = self.sq_walk(c, x, y, r * 0.45)
            boost = dist > 30.0 and lands_in <= dist / 13.0 + 2.5 and c.me["energy"] > 45.0
            return mv, boost, False, look

        # Inside/at drop ring: pick cover spot inside capture radius to minimize LOS exposure
        w = self.world
        threats = [(m["pos"][0], m["pos"][1], m["pos"][2] + 8.0) for ed, m in c.enemies[:4] if ed < 180.0]
        ring_r = min(r * 0.60, max(1.5, r - 2.0))
        best_pt, bs = (x, y), 1e18
        for k in range(-1, 8):
            if k < 0:
                sx, sy = x, y
            else:
                a = math.pi * k / 4
                sx, sy = x + math.cos(a) * ring_r, y + math.sin(a) * ring_r
            nc = w.ncell(sx, sy)
            if nc < 0 or w.blocked[nc]:
                continue
            gz = w.ground(sx, sy)
            body = (sx, sy, gz + 4.5)
            ex = sum(1 for o in threats if w.los(o, body, 3.0))
            cov = w.cover[nc]
            s = 35.0 * ex + 1.5 * cov + 0.3 * math.hypot(sx - c.x, sy - c.y)
            if s < bs:
                best_pt, bs = (sx, sy), s
        sd = math.hypot(best_pt[0] - c.x, best_pt[1] - c.y)
        mv = (0.0, 0.0) if sd < 1.2 else unit(best_pt[0] - c.x, best_pt[1] - c.y)
        if sd < 2.5:
            mv = (mv[0] * 0.3, mv[1] * 0.3)
        return mv, False, False, look

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
        if self.strategy == "aggro":
            return self.do_push(c)
        return self.do_rotate(c)

    def do_push(self, c):
        cx, cy, r = c.region
        dist = math.hypot(cx - c.x, cy - c.y)
        mv = self.go(c, cx, cy, max(10.0, min(40.0, r * 0.5)))
        boost = self.zone_urgent(c) or (c.me["energy"] > 70 and dist > 60)
        heading = math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else math.atan2(cy - c.y, cx - c.x)
        return mv, boost, False, self.threat_look(c, self.watch_yaw(c, heading))

    def do_climb(self, c):
        rf, ap = self.roof["rf"], self.roof["ap"]
        qx, qy, ex, ey = ap
        tx, ty = rf["c"]
        need = (rf["top"] - c.ground) * PARAMS["roof_energy_per_m"] + 10
        d_ap = math.hypot(qx - c.x, qy - c.y)
        airborne = not c.me.get("grounded", True)
        near_wall = math.hypot(ex - c.x, ey - c.y) < 9.0
        if airborne and c.z > c.ground + 1.0:
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

    # ================= squad / cand_tempo_drop =================
    def mech_frac(self, m):
        hp, mhp = m.get("hp"), m.get("max_hp")
        if isinstance(hp, (int, float)) and isinstance(mhp, (int, float)) and mhp > 0:
            return min(1.0, max(0.0, hp / mhp))
        return 1.0

    def holding(self, c, kind):
        return any(w and w["kind"] == kind for w in c.me["weapons"])

    def sq_target(self, c):
        if self.lock_tick == c.obs["tick"]:
            return self.lock_pick
        P = PARAMS
        lock_id = self.lock[0] if self.lock and c.t - self.lock[2] < P["sq_stick_s"] else None
        best, bs = None, 1e18
        for d, m in c.enemies:
            if d > P["sq_cap"]:
                break
            mid = m["id"]
            frac = self.mech_frac(m)
            rx, ry = c.x - m["pos"][0], c.y - m["pos"][1]
            hit = c.t - self.hit_by.get(mid, -1e9) < P["sq_hit_s"]
            closing = (m["vel"][0] * rx + m["vel"][1] * ry) / max(d, 1.0) > P["sq_close_v"]
            aiming = abs(wrap(m["aim"][0] - math.atan2(ry, rx))) < P["sq_aim_rad"]
            trig = d < P["sq_near"] or hit or (closing and d < P["sq_close_d"]) \
                or (frac < P["sq_weak_frac"] and d < P["sq_weak_d"]) or (aiming and d < P["sq_aim_d"])
            if not trig and mid != lock_id:
                continue
            hp = m.get("hp")
            if not isinstance(hp, (int, float)):
                hp = finite(m.get("max_hp"), 1400.0)
            s = d + P["sq_hp_w"] * hp
            if frac <= P["sq_low_frac"]:
                s -= P["sq_low_bonus"]
            if mid == lock_id:
                s -= P["sq_stick_bonus"]
            if hit:
                s -= P["sq_hit_bonus"]
            if closing:
                s -= P["sq_close_bonus"]
            if s < bs:
                best, bs = (d, m, trig), s
        if best is not None:
            d, m, trig = best
            self.lock = (m["id"], m, c.t if trig else self.lock[2])
            best = (d, m)
        else:
            if lock_id is None:
                self.lock = None
            att = self.attacker(c)
            if att and att[1].get("kind") == "npc" and att[0] < 100.0:
                best = att
        self.lock_tick, self.lock_pick = c.obs["tick"], best
        return best

    def sq_range(self, c, m):
        P = PARAMS
        if m.get("kind") != "mech":
            return P["sq_r_npc"]
        if self.mech_frac(m) < P["sq_finish_tgt"] and c.hp_frac > P["sq_finish_me"] \
                and len(c.enemies) <= P["sq_finish_view"]:
            return P["sq_r_finish"]
        theirs = m.get("weapons") or []
        arc = self.holding(c, "lightning")
        if self.holding(c, "laser"):
            r = P["sq_r_laser"]
        elif arc and c.me["energy"] > P["sq_lt_energy"] and not self.lt_dry:
            r = P["sq_r_lightning"]
        else:
            r = P["sq_r_gun"]
        if "lightning" in theirs and not arc:
            r = max(r, P["sq_r_vs_lightning"])
        if "mortar" in theirs:
            r = max(r, P["sq_r_vs_mortar"])
        return r

    def sq_move(self, c, tx, ty, tz, pref, others):
        """3.0-4.8s irregular boost-weave strafe with deterministic reversal & radial modulation."""
        P = PARAMS
        w = self.world
        if c.t > self.strafe_until:
            self.strafe_sign = -self.strafe_sign
            self.strafe_until = c.t + self.rng.uniform(*P["sq_strafe_s"])
        bx, by = unit(tx - c.x, ty - c.y)
        px, py = -by * self.strafe_sign, bx * self.strafe_sign
        eff_pref = max(14.0, pref + 8.5 * math.sin(c.t * 2.3 + self.slot * 1.7) + 4.0 * math.cos(c.t * 4.1))
        w_strafe = P["sq_w_strafe"] * (0.75 + 0.5 * abs(math.sin(c.t * 1.9 + self.slot)))
        eh = c.eye[2] - c.z
        airborne = c.z - c.ground > 3.0
        ring = P["sq_ring"]
        cube, cd = None, P["sq_cube_pull_d"]
        for cu in c.obs.get("cubes", []):
            dd = math.hypot(cu["pos"][0] - c.x, cu["pos"][1] - c.y)
            if dd < cd and abs(cu["pos"][2] - c.z) < 4:
                cube, cd = cu["pos"], dd
        tc = (tx, ty, tz)
        best, bs = (0.0, 0.0), 1e18
        for k in range(-1, 16):
            if k < 0:
                dx = dy = 0.0
                sx, sy, gz = c.x, c.y, c.z
            else:
                a = math.pi * k / 8
                dx, dy = math.cos(a), math.sin(a)
                sx, sy = c.x + dx * ring, c.y + dy * ring
                if airborne:
                    gz = c.z
                else:
                    if not w.clear_ray(c.x, c.y, dx, dy, ring + 2.0):
                        continue
                    gz = w.ground(sx, sy)
            s = P["sq_w_range"] * abs(math.hypot(tx - sx, ty - sy) - eff_pref)
            if w.los((sx, sy, gz + eh), tc, 2.0):
                s -= P["sq_w_los"]
            body = (sx, sy, gz + 4.5)
            for o in others:
                if w.los(o, body, 3.0):
                    s += P["sq_w_exposed"]
            if cube is not None:
                s += P["sq_w_cube"] * math.hypot(cube[0] - sx, cube[1] - sy)
            s -= w_strafe * (dx * px + dy * py)
            s += 30.0 * self.region_penalty(c, dx, dy, ahead=ring)
            if s < bs:
                best, bs = (dx, dy), s
        return best

    def sq_others(self, c, m):
        P = PARAMS
        out = []
        for d, o in c.enemies:
            if d > P["sq_exposed_d"] or len(out) >= P["sq_others"]:
                break
            if o is not m:
                out.append((o["pos"][0], o["pos"][1], o["pos"][2] + 8.0))
        return out

    def sq_item_ok(self, c, iid, pos):
        x, y, z = pos
        if self.loot_skip.get(iid, -1.0) >= c.t:
            return False
        if not self.toward_region(c, x, y):
            zc, zr = c.zone["center"], c.zone["radius"]
            if math.hypot(x - zc[0], y - zc[1]) > zr - 15.0:
                return False
            left = self._zone_time_left(c)
            trip = math.hypot(x - c.x, y - c.y) + max(0.0, c.dist_region(x, y))
            if trip / max(1.0, c.stats.get("walk_speed", 11.7)) + PARAMS["zone_slack_s"] + 10.0 > left:
                return False
        return abs(z - c.z) < 4 or z - self.world.ground(x, y) < 3.0

    def sq_work(self, c, iid):
        P = PARAMS
        if self.job is None or self.job[0] != iid:
            self.job = (iid, c.t)
        if c.t - self.job[1] > P["sq_farm_give_up_s"]:
            self.loot_skip[iid] = c.t + P["sq_farm_ban_s"]
            self.job = None
            return False
        return True

    def sq_walk(self, c, x, y, r):
        d = math.hypot(x - c.x, y - c.y)
        if d < 1e-6:
            return (0.0, 0.0)
        ux, uy = (x - c.x) / d, (y - c.y) / d
        if d < 40.0 and (c.z - c.ground > 3.0 or self.world.clear_ray(c.x, c.y, ux, uy, d)):
            return (ux, uy) if d > r else (0.0, 0.0)
        return self.go(c, x, y, r)

    # ----- breaking off -----
    def sq_losing(self, c):
        P = PARAMS
        near = 0
        for d, _m in c.enemies:
            if d < P["sq_lose_d"]:
                near += 1
        tgt = self.sq_target(c)
        tf = self.mech_frac(tgt[1]) if tgt and tgt[1].get("kind") == "mech" else None
        recent_atts = sum(1 for mid, ht in self.hit_by.items() if mid < 100 and c.t - ht < P["sq_crossfire_s"])
        if recent_atts >= 2 and (self.state == "DROP" or c.hp_frac < 0.60):
            return True
        if tf is not None and tf < P["sq_press_frac"] and near <= P["sq_press_near"]:
            return False
        hp = c.hp_frac
        return (near >= 2 and hp < P["sq_lose2_hp"]) or (near >= 3 and hp < P["sq_lose3_hp"]) \
            or (hp < P["sq_lose_hp"] and tf is not None and tf > hp + P["sq_lose_gap"])

    def _open_flee_spot(self, c):
        P = PARAMS
        w = self.world
        pts = [
            (m["pos"][0], m["pos"][1])
            for d, m in c.enemies
            if d < P["sq_lose_d"] or c.t - self.hit_by.get(m["id"], -1e9) < P["sq_crossfire_s"]
        ]
        if not pts:
            if c.enemies:
                pts = [(c.enemies[0][1]["pos"][0], c.enemies[0][1]["pos"][1])]
            elif self.threat is not None and c.t - self.threat[1] < P["sq_crossfire_s"]:
                b = self.threat[0]
                pts = [(c.x + math.cos(b) * 50.0, c.y + math.sin(b) * 50.0)]
            else:
                return None
        ax = sum(p[0] for p in pts) / len(pts)
        ay = sum(p[1] for p in pts) / len(pts)
        comp = w.component(c.x, c.y)
        best, bs = None, -1e18
        for k in range(12):
            a = math.pi * k / 6
            dx, dy = math.cos(a), math.sin(a)
            for r in (36.0, 22.0):
                nc = w.nearest_free(c.x + dx * r, c.y + dy * r, 3)
                if nc < 0 or (comp and w.comp[nc] != comp):
                    continue
                tx, ty = w.ncenter(nc)
                td = math.hypot(tx - c.x, ty - c.y)
                if td < 8.0:
                    continue
                ux, uy = (tx - c.x) / td, (ty - c.y) / td
                clear = w.clear_ray(c.x, c.y, ux, uy, min(td, 32.0))
                min_d = min(math.hypot(tx - px, ty - py) for px, py in pts)
                cent_d = math.hypot(tx - ax, ty - ay)
                reg_pen = 3.0 * max(0.0, c.dist_region(tx, ty) + 5.0)
                s = min_d + 0.5 * cent_d + (15.0 if clear else 0.0) - 1.5 * w.cover[nc] - reg_pen
                if s > bs:
                    best, bs = (tx, ty), s
                break
        if best is None:
            ux, uy = unit(c.x - ax, c.y - ay)
            if ux == 0.0 and uy == 0.0:
                ux, uy = 1.0, 0.0
            nc = w.nearest_free(c.x + ux * 24.0, c.y + uy * 24.0, 6)
            if nc >= 0:
                best = w.ncenter(nc)
        return best

    def sq_cover(self, c):
        P = PARAMS
        if self.cover and c.t - self.cover[0] < P["sq_cover_s"]:
            return self.cover[1]
        w = self.world
        threats = [(m["pos"][0], m["pos"][1], m["pos"][2] + 8.0) for d, m in c.enemies[:3] if d < P["sq_lose_d"]]
        pick = None
        if threats and c.enemies[0][0] >= P["sq_trapped_d"] and c.z - c.ground < 3.0:
            comp = w.component(c.x, c.y)
            here = sum(1 for o in threats if w.los(o, (c.x, c.y, c.z + 4.5), 3.0))
            if here == 0:
                pick = (c.x, c.y)
            else:
                best, bs, be = None, 1e18, here
                for k in range(12):
                    a = math.pi * k / 6
                    dx, dy = math.cos(a), math.sin(a)
                    for r in (12.0, 26.0, 44.0):
                        x, y = c.x + dx * r, c.y + dy * r
                        nc = w.ncell(x, y)
                        if nc < 0 or w.blocked[nc] or (comp and w.comp[nc] != comp):
                            continue
                        body = (x, y, w.ground(x, y) + 4.5)
                        ex = sum(1 for o in threats if w.los(o, body, 3.0))
                        s = 0.6 * r + 40.0 * ex + 2.0 * max(0.0, c.dist_region(x, y) + 5.0)
                        for o in threats:
                            s += 0.6 * max(0.0, 50.0 - math.hypot(o[0] - x, o[1] - y))
                        if not w.clear_ray(c.x, c.y, dx, dy, r):
                            s += 8.0
                        if s < bs:
                            best, bs, be = (x, y), s, ex
                if best is not None and be < here:
                    pick = best
        recent_atts = sum(1 for mid, ht in self.hit_by.items() if mid < 100 and c.t - ht < P["sq_crossfire_s"])
        if pick is None and (recent_atts >= 2 or c.hp_frac < 0.60 or self.state in ("DROP", "FLEE")):
            pick = self._open_flee_spot(c)
        self.cover = (c.t, pick)
        return pick

    def want_flee(self, c):
        losing = self.sq_losing(c)
        if losing:
            if self.state == "DROP":
                self.drop_ban_until = max(self.drop_ban_until, c.t + PARAMS["sq_drop_ban_s"])
            if c.t >= self.lose_until:
                self.cover = None
            self.lose_until = c.t + PARAMS["sq_lose_hold_s"]
        return c.t < self.lose_until and self.sq_cover(c) is not None

    def do_flee(self, c):
        x, y = self.sq_cover(c)
        d = math.hypot(x - c.x, y - c.y)
        mv = self.sq_walk(c, x, y, 1.5)
        if d > 10.0 and mv != (0.0, 0.0):
            weave = 0.42 * math.sin(c.t * 2.6 + self.slot * 1.3)
            wx, wy = unit(mv[0] - mv[1] * weave, mv[1] + mv[0] * weave)
            if (c.z - c.ground > 3.0) or self.world.clear_ray(c.x, c.y, wx, wy, 6.0):
                mv = (wx, wy)
        tgt = self.sq_target(c) or (c.enemies[0] if c.enemies else None)
        look = math.atan2(tgt[1]["pos"][1] - c.y, tgt[1]["pos"][0] - c.x) if tgt else c.me["aim"][0]
        return mv, d > 8.0 and c.me["energy"] > 18, False, look

    # ----- fighting -----
    def want_duel(self, c):
        return self.sq_target(c) is not None

    def sq_fight_loot(self, c, d_tgt, tgt_pos):
        """Safe mid-fight cube grabs (<35m when healthy & safe) + weapon seeks."""
        P = PARAMS
        gap = self.next_cube_gap(c)
        healthy = c.hp_frac > P["sq_cube_hurt_hp"]
        cube_lim = P["sq_cube_r"] if healthy else P["sq_cube_r_hurt"]
        best, bd, bs = None, 1e18, 1e18
        for cu in c.obs.get("cubes", []):
            d = math.hypot(cu["pos"][0] - c.x, cu["pos"][1] - c.y)
            if d >= cube_lim or abs(cu["pos"][2] - c.z) >= 4 or not self.sq_item_ok(c, cu["id"], cu["pos"]):
                continue
            # Safe check: nearer to us than enemy, not charging into enemy's face
            if d > d_tgt * (0.65 if healthy else 0.55):
                continue
            d_cu_tgt = math.hypot(cu["pos"][0] - tgt_pos[0], cu["pos"][1] - tgt_pos[1])
            if d_cu_tgt < min(25.0, d_tgt * 0.55):
                continue
            s = d - (12.0 if gap <= 1 else 0.0)
            if s < bs:
                best, bd, bs = cu, d, s
        empty = self.slot1_empty(c)
        for iid, (kind, pos) in self.seen_items.items():
            if kind == "crate" or not self.wanted_weapon(kind, c):
                continue
            d = math.hypot(pos[0] - c.x, pos[1] - c.y)
            lim, bonus = (P["sq_laser_seek"], 90.0) if kind == "laser" else \
                ((P["sq_pickup_seek"], 30.0) if empty else (P["fight_loot_r"], 0.0))
            if d < lim and d - bonus < bs and self.sq_item_ok(c, iid, pos):
                best, bd, bs = {"id": iid, "pos": pos}, d, d - bonus
        if best is None:
            self.fight_loot = None
            return None
        if not self.fight_loot or self.fight_loot[0] != best["id"]:
            self.fight_loot = (best["id"], c.t, 4.0 + bd / 6.0)
        elif c.t - self.fight_loot[1] > self.fight_loot[2]:
            self.loot_skip[best["id"]] = c.t + 20.0
            self.fight_loot = None
            return None
        return self.sq_walk(c, best["pos"][0], best["pos"][1], 0.5), bd

    def do_duel(self, c):
        d, m = self.sq_target(c)
        tx, ty, tz = m["pos"]
        look = math.atan2(ty - c.y, tx - c.x)
        pref = self.sq_range(c, m)
        floor = self.boost_floor(c, d)
        if m.get("kind") == "mech":
            fl = self.sq_fight_loot(c, d, m["pos"])
            if fl is not None:
                return fl[0], fl[1] > 6 and c.me["energy"] > floor, False, look
        center = (tx, ty, tz + (4.5 if m.get("kind") == "mech" else 1.75))
        if self.world.los(c.eye, center):
            self.sq_los_t = c.t
        elif c.t - self.sq_los_t > 1.0:
            mv = self.go(c, round(tx / 20) * 20, round(ty / 20) * 20, 12.0)
            if mv != (0.0, 0.0):
                return mv, d > pref and c.me["energy"] > floor, False, look
        mv = self.sq_move(c, tx, ty, center[2], pref, self.sq_others(c, m))
        # Enemy-aim / laser-charge boost pulses to break 1.0x linear gunnery without draining energy
        threat_aim = False
        charging_threat = False
        for ed, em in c.enemies:
            if ed > 220.0:
                break
            to_me = math.atan2(c.y - em["pos"][1], c.x - em["pos"][0])
            if abs(wrap(em["aim"][0] - to_me)) < 0.25:
                threat_aim = True
                if em.get("charging"):
                    charging_threat = True
                    break
        phase = (c.t + self.slot * 0.37) % 1.4
        pulse = (charging_threat and phase < 0.35) or (threat_aim and phase < 0.28)
        boost = (abs(d - pref) > 30.0 and c.me["energy"] > floor) \
            or (pulse and c.me["energy"] > max(floor + 20.0, 56.0))
        return mv, boost, False, look

    def sq_pick(self, c):
        cands = []
        tgt = self.sq_target(c)
        if tgt:
            cands.append(tgt)
        if self.state == "HARVEST":
            f = self.sq_farm(c)
            if f and f[0] == "npc":
                cands.append((f[2], f[1]))
        for d, m in cands:
            if d > PARAMS["sq_cap"]:
                continue
            center = (m["pos"][0], m["pos"][1], m["pos"][2] + (4.5 if m.get("kind") == "mech" else 1.75))
            if self.world.los(c.muzzle, center):
                return d, m, center
        return None

    def sq_partial_kills(self, c, lw, m):
        st = lw["stats"]
        ch = lw.get("charge", 0)
        hp = m.get("hp")
        if ch < st.get("min_charge", 0.15) or not isinstance(hp, (int, float)):
            return False
        lo = st.get("min_frac", 0.35)
        dmg = st.get("damage", 315.0) * (lo + (1.0 - lo) * min(1.0, ch / max(1e-6, st.get("charge", 0.75))))
        dmg *= self.sq_cube_mult(c)
        if m.get("kind") == "mech" and "mortar" in (m.get("weapons") or []):
            dmg *= 1.0 + st.get("cookoff", 0.6)
        return PARAMS["sq_kill_margin"] * dmg >= hp

    def sq_cube_mult(self, c):
        cu = self.cfg.get("cubes", {})
        return 1.0 + cu.get("damage_per_cube", 0.04) * min(c.me.get("cubes", 0), cu.get("stat_cap", 10))

    # ----- loot and farming (fast 1/3/6/10 cube snowball) -----
    def sq_seek(self, c):
        if self.seek_tick == c.obs["tick"]:
            return self.seek_pick
        P = PARAMS
        cu_cfg = self.cfg.get("cubes", {})
        cubes = c.me.get("cubes", 0)
        capped = cubes >= cu_cfg.get("stat_cap", 10)
        early = c.t < P["sq_early_t"] and not capped
        cr = P["sq_cube_seek_cap"] if capped else (P["sq_cube_seek_early"] if early else P["sq_cube_seek"])
        gap = self.next_cube_gap(c)
        cube_bonus = 45.0 if gap <= 1 else (20.0 if gap == 2 else 0.0)
        best, bs = None, 1e18
        for cu in c.obs.get("cubes", []):
            d = math.hypot(cu["pos"][0] - c.x, cu["pos"][1] - c.y)
            s = d - cube_bonus
            if d < cr and s < bs and self.sq_item_ok(c, cu["id"], cu["pos"]):
                best, bs = (cu["id"], cu["pos"], d), s
        for iid, (kind, pos) in self.seen_items.items():
            if kind == "crate" or not self.wanted_weapon(kind, c):
                continue
            d = math.hypot(pos[0] - c.x, pos[1] - c.y)
            lim, bonus = (P["sq_laser_seek"], 90.0) if kind == "laser" else (P["sq_pickup_seek"], 30.0)
            if d < lim and d - bonus < bs and self.sq_item_ok(c, iid, pos):
                best, bs = (iid, pos, d), d - bonus
        if best is not None and not self.sq_work(c, best[0]):
            best = None
        self.seek_tick, self.seek_pick = c.obs["tick"], best
        return best

    def want_seek(self, c):
        return self.sq_seek(c) is not None

    def do_seek(self, c):
        _iid, pos, d = self.sq_seek(c)
        mv = self.sq_walk(c, pos[0], pos[1], 0.5)
        heading = math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else c.me["aim"][0]
        return mv, d > 45 and c.me["energy"] > 50, False, self.threat_look(c, heading)

    def sq_farm(self, c):
        if self.farm_tick == c.obs["tick"]:
            return self.farm_pick
        P = PARAMS
        cubes = c.me.get("cubes", 0)
        capped = cubes >= self.cfg.get("cubes", {}).get("stat_cap", 10)
        early = c.t < P["sq_early_t"] and not capped
        farm_r = P["sq_farm_r_early"] if early else P["sq_farm_r"]
        farm_hp = P["sq_farm_hp_early"] if early else P["sq_farm_hp"]
        gap = self.next_cube_gap(c)
        best, bs = None, 1e18
        if c.hp_frac > farm_hp:
            cur = self.job[0] if self.job else None
            crate_bonus = 20.0 if (early and gap <= 1) else 0.0
            npc_bonus = 35.0 if (early and gap <= 2) else 0.0
            for iid, (kind, pos) in self.seen_items.items():
                if kind != "crate":
                    continue
                d = math.hypot(pos[0] - c.x, pos[1] - c.y)
                s = d - crate_bonus - (40.0 if iid == cur else 0.0)
                if d < farm_r and s < bs and pos[2] - self.world.ground(pos[0], pos[1]) < 3.0 \
                        and not any(math.hypot(m["pos"][0] - pos[0], m["pos"][1] - pos[1]) < 65.0 for _ed, m in c.enemies) \
                        and self.sq_item_ok(c, iid, pos):
                    best, bs = ("crate", pos, d, iid), s
            if not capped:
                for d, m in c.npcs:
                    s = d - npc_bonus - (40.0 if m["id"] == cur else 0.0)
                    if d < farm_r and s < bs and self.sq_item_ok(c, m["id"], m["pos"]):
                        best, bs = ("npc", m, d, m["id"]), s
            if best is not None and not self.sq_work(c, best[3]):
                best = None
        self.farm_tick, self.farm_pick = c.obs["tick"], best
        return best

    def want_harvest(self, c):
        return self.sq_farm(c) is not None

    def do_harvest(self, c):
        P = PARAMS
        kind, what, d, iid = self.sq_farm(c)
        self.sq_crate = None
        if kind == "npc":
            tx, ty, tz = what["pos"]
            look = math.atan2(ty - c.y, tx - c.x)
            if d > 90.0 and not self.world.los(c.eye, (tx, ty, tz + 1.75)):
                return self.go(c, round(tx / 10) * 10, round(ty / 10) * 10, 30.0), False, False, look
            return self.sq_move(c, tx, ty, tz + 1.75, P["sq_r_npc"], self.sq_others(c, what)), False, False, look
        x, y, z = what
        look = math.atan2(y - c.y, x - c.x)
        seen = self.world.los(c.muzzle, (x, y, z + 1.2))
        if seen and d < 60.0:
            for cr in c.obs.get("crates", []):
                if cr["id"] == iid:
                    self.sq_crate = cr
        if d > P["sq_crate_shoot_d"] or not seen:
            return self.sq_walk(c, x, y, 6.0), False, False, look
        if self.sq_crate is None:
            self.seen_items.pop(iid, None)
        return (0.0, 0.0), False, False, look

    def sq_roof_item(self, c):
        if self.rooftake_tick == c.obs["tick"]:
            return self.rooftake_pick
        P = PARAMS
        w = self.world
        pick = None
        if not self.holding(c, "laser") and c.hp_frac > P["sq_farm_hp"]:
            tgt = self.sq_target(c)
            lim = P["sq_roof_r"] if tgt is None else P["sq_roof_r_fight"]
            best, bd = None, lim
            for iid, (kind, pos) in self.seen_items.items():
                d = math.hypot(pos[0] - c.x, pos[1] - c.y)
                if kind == "laser" and d < bd and pos[2] - w.ground(pos[0], pos[1]) >= 3.0 \
                        and self.loot_skip.get(iid, -1.0) < c.t and c.dist_region(pos[0], pos[1]) <= 0:
                    best, bd = (iid, pos), d
            if best is not None:
                iid, pos = best
                if self.roof is None or self.roof.get("item") != iid:
                    self.roof = None
                    for i, rf in enumerate(w.roofs()):
                        h = rf["top"] - w.ground(rf["c"][0], rf["c"][1])
                        if abs(rf["top"] - pos[2]) < 2.0 and h <= P["sq_roof_max_h"] \
                                and abs(w.roof_top(pos[0], pos[1]) - rf["top"]) < 2.0 \
                                and math.hypot(rf["c"][0] - pos[0], rf["c"][1] - pos[1]) < 60.0 \
                                and self.roof_fail.get(i, 0) < P["roof_max_tries"]:
                            ap = w.approach_point(rf, c.x, c.y)
                            if ap is not None:
                                self.roof = {"i": i, "rf": rf, "ap": ap, "key": c.region, "item": iid}
                                break
                    if self.roof is None:
                        self.loot_skip[iid] = c.t + P["sq_farm_ban_s"]
                if self.roof is not None and self.sq_work(c, iid):
                    pick = best
        self.rooftake_tick, self.rooftake_pick = c.obs["tick"], pick
        return pick

    def want_rooftake(self, c):
        return self.sq_roof_item(c) is not None

    def do_rooftake(self, c):
        _iid, pos = self.sq_roof_item(c)
        if self.on_roof(c):
            mv = unit(pos[0] - c.x, pos[1] - c.y)
            return mv, False, False, self.threat_look(c, math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else c.me["aim"][0])
        return self.do_climb(c)

    def sq_scout(self, c):
        if self.holding(c, "laser"):
            return None
        g = PARAMS["sq_scout_grid"]
        cx, cy, r = c.region
        i0, j0 = int(round(c.x / g)), int(round(c.y / g))
        self.scouted.add((i0, j0))
        best, bd = None, 1e18
        for j in range(j0 - 4, j0 + 5):
            for i in range(i0 - 4, i0 + 5):
                x, y = i * g, j * g
                if (i, j) in self.scouted or math.hypot(x - cx, y - cy) > r:
                    continue
                d = math.hypot(x - c.x, y - c.y)
                if d < bd:
                    if not self.world.free(x, y):
                        self.scouted.add((i, j))
                        continue
                    best, bd = (x, y), d
        return best

    def want_scout(self, c):
        return self.sq_scout(c) is not None

    def do_scout(self, c):
        x, y = self.sq_scout(c)
        mv = self.go(c, x, y, 10.0)
        heading = math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else c.me["aim"][0]
        return mv, False, False, self.threat_look(c, self.watch_yaw(c, heading))

    # ================= gunnery =================
    def pick_target(self, c):
        P = PARAMS
        if self._is_squad():
            return self.sq_pick(c)
        policy = self.profile["fire"]
        attacked = self.threat_recent(c, P["threat_memory_s"])
        if policy == "attacked" and not attacked:
            return None
        if self.state == "RETREAT" and c.t - self.last_damage_t > P["retreat_quiet_fire_s"]:
            return None
        att = self.attacker(c)
        cands = []
        if self.strategy == "aggro":
            ft = self.aggro_target(c)
            if ft and any(ft[1]["id"] == m["id"] for _d, m in c.enemies + c.npcs):
                cands.append(ft)
        if att and all(att[1]["id"] != m["id"] for _d, m in cands):
            cands.append(att)
        cands += [e for e in c.enemies if all(e[1]["id"] != m["id"] for _d, m in cands)]
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
                e = c.me["energy"]
                if e <= P["lt_stop_energy"]:
                    self.lt_dry = True
                elif e >= P["lt_resume_energy"]:
                    self.lt_dry = False
                if d < ls.get("range", 45) * 0.95 and abs(wrap(c.me["aim"][0] - direct)) < math.radians(ls.get("cone_deg", 30)) * 0.4 \
                        and not self.lt_dry and e > ls.get("energy", 3):
                    fire[lt] = True
            if laser_slot is not None:
                lw = weapons[laser_slot]
                direct_yaw = math.atan2(py - my, px - mx)
                direct_pitch = math.atan2(pz - mz, math.hypot(px - mx, py - my))
                charged = lw.get("charge", 0) >= lw["stats"].get("charge", 0.75) * 0.95
                if self._is_squad() and not charged and self.sq_partial_kills(c, lw, m):
                    charged = True
                if lw.get("charge", 0) > 0 or lw["cooldown"] <= 0:
                    aim_yaw, aim_pitch = direct_yaw, direct_pitch
                    fire[laser_slot] = not (charged and self.aimed(c, direct_yaw, direct_pitch, tol))
                    if fire[laser_slot] and gun is not None:
                        fire[gun] = fire[gun] and self.aimed(c, direct_yaw, direct_pitch, tol)
        elif laser_slot is not None and weapons[laser_slot].get("charge", 0) > 0:
            fire[laser_slot] = False
        else:
            cr = self.crate_target(c)
            if cr is not None:
                aim_yaw, aim_pitch, fire = self.shoot_crate(c, cr, weapons)
        mp = c.stats.get("max_pitch", 1.3)
        np_ = c.stats.get("min_pitch", -0.7)
        aim_pitch = min(mp, max(np_, finite(aim_pitch)))
        return [round(wrap(finite(aim_yaw)), 4), round(aim_pitch, 4)], fire

    def crate_target(self, c):
        P = PARAMS
        if self._is_squad():
            return self.sq_crate if self.state == "HARVEST" else None
        if self.strategy != "aggro" or c.hp_frac < P["crate_min_hp"] or self.state in ("ESCAPE", "DODGE") \
                or self.zone_urgent(c):
            return None
        if any(d < P["crate_min_target"] for d, _m in c.enemies):
            return None
        best, bd = None, P["crate_r"]
        for cr in c.obs.get("crates", []):
            x, y, z = cr["pos"]
            d = math.hypot(x - c.x, y - c.y)
            if d < bd and abs(z - c.z) < 4 and self.toward_region(c, x, y):
                best, bd = cr, d
        if best is None:
            return None
        x, y, z = best["pos"]
        if not self.world.los(c.muzzle, (x, y, z + 1.2)):
            return None
        return best

    def shoot_crate(self, c, cr, weapons):
        mx, my, mz = c.muzzle
        x, y, z = cr["pos"]
        z += 1.2
        yaw = math.atan2(y - my, x - mx)
        d = math.hypot(x - mx, y - my)
        pitch = math.atan2(z - mz, d)
        fire = [False, False]
        on = self.aimed(c, yaw, pitch, max(PARAMS["aim_tol_min"], math.atan2(1.0, max(d, 1.0))))
        for i, w in enumerate(weapons):
            if not w or not on:
                continue
            k = w["kind"]
            if k == "gun" and w.get("ammo", 1) > 0 and w.get("reload", 0) <= 0:
                fire[i] = True
            elif k == "lightning" and d < w["stats"].get("range", 45) * 0.9 and not self.lt_dry \
                    and c.me["energy"] > PARAMS["lt_resume_energy"]:
                fire[i] = True
        return yaw, pitch, fire

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
            pref = P["upgrade_pref"]
            if "upgrade_pref_laser" in self.profile and self.holding(c, "laser"):
                pref = self.profile["upgrade_pref_laser"]
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
        gun = weapons[0] if weapons and weapons[0] and weapons[0]["kind"] == "gun" else None
        if gun and not act["fire"][0] and gun.get("reload", 0) <= 0 \
                and gun.get("ammo", 99) < gun["stats"].get("mag", 12) * 0.5 \
                and not any(d < P["crate_min_target"] for d, _m in c.enemies):
            act["reload"] = True


if __name__ == "__main__":
    mf.run(Bot)
