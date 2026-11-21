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
    },
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
    # free-for-all mode: only after mechs died that we did not kill, early in the match
    "ffa_detect_until_t": 45.0, "ffa_hold_t": 45.0,
    # FFA evidence: a mech weapon (beam, or a gun bullet traced to its shooter) seen hitting another mech, confirmed
    # by that mech's HP dropping; and deaths we did not cause, each worth 1 up to ffa_death_full_t, fading to 0 at
    # ffa_death_zero_t (practice house bots die too, but later). Fires on ffa_hits_needed hits, a death score of
    # ffa_death_score, or one hit with ffa_hit_death_score.
    "ffa_hits_needed": 2, "ffa_hit_dedup_s": 1.0,
    "ffa_death_full_t": 18.0, "ffa_death_zero_t": 20.0, "ffa_death_score": 2.0, "ffa_hit_death_score": 1.5,
    "ffa_confirm_s": 0.5, "ffa_min_drop": 15.0, "ffa_gun_min_drop": 30.0, "ffa_body_r": 4.0, "ffa_me_clear": 10.0, "ffa_me_cone": 0.6,
    "ffa_shooter_hit_us_s": 2.0, "ffa_npc_clear": 90.0, "ffa_bullet_reach_s": 0.35,
    # FFA: break line of sight on foot, stand off instead of brawling, hold an unseen spot in the middle ring
    "ffa_seen_r": 250.0, "ffa_seen_max": 4, "ffa_cover_r": (20.0, 40.0, 60.0), "ffa_cover_dirs": 12,
    "ffa_cover_hold_s": 8.0, "ffa_cover_retry_s": 3.0, "ffa_cover_reach": 3.0,
    "ffa_fight_min": 60.0, "ffa_fight_max": 110.0, "ffa_lt_close": 38.0, "ffa_lt_weak": 0.5,
    "ffa_engage_r": 160.0, "ffa_crowd_r": 250.0, "ffa_final_alive": 5,
    "ffa_ring_in": 0.55, "ffa_ring_out": 0.9, "ffa_ring_w": 1.0, "ffa_ring_min_r": 40.0,
    "ffa_spot_seen_cost": 40.0, "ffa_spot_los_top": 8,
    # FFA spacing: hide spots repel mechs within ffa_spot_enemy_r; re-pick when one comes within ffa_respot_r
    "ffa_spot_enemy_r": 150.0, "ffa_respot_r": 120.0, "ffa_respot_s": 5.0,
    # FFA zone: walk to the next circle earlier, stay in ESCAPE until well inside, and weigh the circle hard
    # while it shrinks (the final circle killed a 2nd place that kept fighting at its edge)
    "ffa_zone_slack_s": 8.0, "ffa_zone_slack_dps": 0.03, "ffa_escape_in": 8.0, "ffa_region_w": 0.2,
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


class FfaEvidence:
    """Detects a free-for-all from one observation at a time (no RNG, a few dozen float ops per visible shot).
    Practice opponents never shoot each other, so a mech weapon hitting another mech is the signal. It is attributed
    strictly: beams name their shooter, gun bullets are traced back to a single mech, mortars and NPC fire never
    count, and the hit must show as an HP drop on the target that our own hits do not explain. Shots that may have
    been meant for us are ignored: lines passing near us, lightning arcs with us in the shooter's cone, and anything
    fired by a mech that damaged us moments ago. Deaths we did not cause count only early in the match, when practice
    house bots almost never die; a death is credited once it persists for a second observation (a late kill event
    can still explain it), with the weight of the moment it was first seen."""

    def __init__(self):
        self.hp = {}            # mech id -> hp at the last observation that listed it
        self.pending = []       # (t, shooter, target, kind, target hp then, our damage on it since)
        self.hits = []          # confirmed (t, shooter, target, kind)
        self.hit_us = {}        # mech id -> last t it damaged us
        self.alive0 = None
        self.deaths = 0         # unexplained deaths credited so far
        self.death_score = 0.0
        self.death_seen = None  # (t, count) of an uncredited rise in unexplained deaths
        self.fired = None       # (t, reason) once detected

    @staticmethod
    def _body_near(m, p, r):
        x, y, z = m["pos"]
        return math.hypot(p[0] - x, p[1] - y) <= r and z - 1.0 <= p[2] <= z + 10.0

    @staticmethod
    def _seg_dist(ax, ay, bx, by, px, py, extend=0.0):
        """Horizontal distance from p to segment a->b extended by `extend` beyond b."""
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return math.hypot(px - ax, py - ay)
        ux, uy = dx / L, dy / L
        s = max(0.0, min(L + extend, (px - ax) * ux + (py - ay) * uy))
        return math.hypot(ax + ux * s - px, ay + uy * s - py)

    def update(self, obs, kills):
        """Feed one observation and our kill total so far; returns True once a free-for-all is detected."""
        P = PARAMS
        if self.fired is not None:
            return True
        t = obs.get("t", 0.0)
        me = obs["self"]
        my_id = me.get("id")
        mx, my = me["pos"][0], me["pos"][1]
        mechs, npcs = {}, []
        for m in obs.get("mechs", []):
            if m.get("kind") == "mech":
                if isinstance(m.get("id"), int) and 0 <= m["id"] <= 15 and m["id"] != my_id:
                    mechs[m["id"]] = m
            else:
                npcs.append(m)
        events = obs.get("events", []) + obs.get("skipped_events", [])
        ours = {}
        for ev in events:
            if ev.get("type") == "hit":
                ours[ev.get("target")] = ours.get(ev.get("target"), 0.0) + finite(ev.get("amount"))
            elif ev.get("type") == "damaged" and ev.get("by") is not None:
                self.hit_us[ev["by"]] = t

        def suspect(sid):  # unknown shooter, or one that was just shooting at us
            return not isinstance(sid, int) or sid == my_id or not 0 <= sid <= 15 or \
                t - self.hit_us.get(sid, -1e9) < P["ffa_shooter_hit_us_s"]

        npc_shots = [pr["pos"] for pr in obs.get("projectiles", []) if pr.get("kind") == "npc" and len(pr.get("pos") or ()) >= 2]

        def npc_near(m):  # a drone (or its bullet) that could explain an HP drop
            x, y = m["pos"][0], m["pos"][1]
            return any(math.hypot(n["pos"][0] - x, n["pos"][1] - y) < P["ffa_npc_clear"] for n in npcs) or \
                any(math.hypot(q[0] - x, q[1] - y) < 20.0 for q in npc_shots)

        new = []
        for b in obs.get("beams", []) + obs.get("skipped_beams", []):
            sid, path = b.get("by"), b.get("path") or []
            if suspect(sid) or len(path) < 2 or any(len(q) < 3 for q in path):
                continue
            a, end = path[0], path[1] if b.get("kind") == "lightning" else path[-1]  # an arc's later jumps are chain
            tgt = next((m for i, m in mechs.items() if i != sid and self._body_near(m, end, P["ffa_body_r"])), None)
            if tgt is None:
                continue
            if self._seg_dist(a[0], a[1], end[0], end[1], mx, my, 280.0) < P["ffa_me_clear"]:
                continue  # we were on or near the line
            if b.get("kind") == "lightning":
                ang = abs(wrap(math.atan2(my - a[1], mx - a[0]) - math.atan2(end[1] - a[1], end[0] - a[0])))
                if ang < P["ffa_me_cone"] and math.hypot(mx - a[0], my - a[1]) < 60.0:
                    continue  # we were in its cone: the arc may have been aimed at us
            new.append((sid, tgt, b.get("kind")))
        for pr in obs.get("projectiles", []):
            if pr.get("kind") != "gun" or len(pr.get("pos") or ()) < 3 or len(pr.get("vel") or ()) < 3:
                continue
            (px, py, pz), (vx, vy, vz) = pr["pos"][:3], pr["vel"][:3]
            sp = math.hypot(vx, vy)
            if sp < 50.0:
                continue
            ux, uy = vx / sp, vy / sp
            shooters, tgt, best = [], None, 1e9
            for i, m in mechs.items():
                x, y, z = m["pos"]
                along = (x - px) * ux + (y - py) * uy
                perp = abs((x - px) * uy - (y - py) * ux)
                if -240.0 <= along <= -1.0 and perp < 3.0 + 12.0 * -along / sp:
                    shooters.append(i)
                elif 0.0 <= along <= sp * P["ffa_bullet_reach_s"] and perp < P["ffa_body_r"] and along < best:
                    bz = pz + vz * along / sp
                    if z - 1.0 <= bz <= z + 10.0:
                        tgt, best = m, along
            if len(shooters) != 1 or tgt is None or tgt["id"] == shooters[0] or suspect(shooters[0]):
                continue
            along = (mx - px) * ux + (my - py) * uy
            if -5.0 <= along <= 240.0 and abs((mx - px) * uy - (my - py) * ux) < P["ffa_me_clear"]:
                continue  # also heading our way
            if npc_near(tgt):
                continue  # a drone could explain the HP drop
            new.append((shooters[0], tgt, "gun"))
        for sid, tgt, kind in new:
            if tgt["id"] in self.hp and not any(p[1] == sid and p[2] == tgt["id"] for p in self.pending):
                self.pending.append([t, sid, tgt["id"], kind, self.hp[tgt["id"]], 0.0])
        keep = []
        for p in self.pending:
            p[5] += ours.get(p[2], 0.0)
            m = mechs.get(p[2])
            need = P["ffa_gun_min_drop"] if p[3] == "gun" else P["ffa_min_drop"]  # one drone shot is less than a bullet
            if m is not None and isinstance(m.get("hp"), (int, float)) and p[4] - m["hp"] - p[5] >= need \
                    and not (p[3] == "gun" and npc_near(m)):
                if not any(h[1] == p[1] and h[2] == p[2] and t - h[0] < P["ffa_hit_dedup_s"] for h in self.hits):
                    self.hits.append((t, p[1], p[2], p[3]))
                    log(f"[cbot] t={t:.1f} ffa evidence: {p[3]} {p[1]} -> {p[2]} hp {p[4]:.0f} -> {m['hp']:.0f}")
            elif t - p[0] < P["ffa_confirm_s"]:
                keep.append(p)
        self.pending = keep
        for i, m in mechs.items():
            if isinstance(m.get("hp"), (int, float)):
                self.hp[i] = m["hp"]
        alive = obs.get("alive")
        if isinstance(alive, int):
            if self.alive0 is None:
                self.alive0 = alive
            n = self.alive0 - alive - kills
            if n <= self.deaths:
                self.death_seen = None
            elif self.death_seen is None or n < self.death_seen[1]:
                self.death_seen = (t, n)
            elif t > self.death_seen[0]:
                t0, f0, z0 = self.death_seen[0], P["ffa_death_full_t"], P["ffa_death_zero_t"]
                wt = 1.0 if t0 <= f0 else max(0.0, (z0 - t0) / max(1e-6, z0 - f0))
                self.death_score += wt * (n - self.deaths)
                self.deaths = n
                self.death_seen = None
        if t > P["ffa_detect_until_t"]:
            return False
        h, d = len(self.hits), self.death_score
        if h >= P["ffa_hits_needed"] or d >= P["ffa_death_score"] - 1e-9 or (h >= 1 and d >= P["ffa_hit_death_score"] - 1e-9):
            self.fired = (t, f"hits={h} deaths={self.deaths} score={d:.2f}")
            return True
        return False


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
        self.spot_t = -1e9
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
        self.lock = None            # (id, last seen mech dict, t)
        self.lock_tick = None
        self.lock_pick = None
        self.fight_loot = None      # (item id, start t, give-up s)
        self.loot_skip = {}         # item id -> t until which it is ignored
        self.lt_dry = False         # lightning paused until energy is back to lt_resume_energy
        self.crate_id = None
        self.handlers = {s: getattr(self, "do_" + s.lower()) for s in self.profile["states"]}
        self.wants = {s: getattr(self, "want_" + s.lower()) for s in self.profile["states"]}
        # FFA mode: off until evidence; self.ffa is the only switch and it never turns back off
        self.ffa = False
        self.evidence = FfaEvidence()
        self.kills = 0
        self.npc_ids = set()
        self.alive = None
        self.seen_tick = None       # (tick, mechs that can see us)
        self.cover = None           # (x, y, until t) walk-only hiding point
        self.cover_retry_t = -1e9
        self.ffa_counts = {}        # branch -> decisions, for logs
        self.ffa_tick = {}          # branch -> last tick counted
        early = [s for s in self.profile["states"] if s != "HUNT"]
        early.insert(early.index("DODGE") + 1 if "DODGE" in early else 0, "EVADE")
        if "ROTATE" not in early:
            early.insert(early.index("HOLD") if "HOLD" in early else len(early), "ROTATE")
        nxt = [s for s in ("LOOT", "FARM", "ROTATE") if s in early][0]
        early.insert(early.index(nxt), "COVER")
        self.ffa_states = early     # the whole match once FFA is detected (no HUNT: nobody is chased)
        for s in early:
            self.handlers.setdefault(s, getattr(self, "do_" + s.lower()))
            self.wants.setdefault(s, getattr(self, "want_" + s.lower()))
        self.timings = []
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
        for s in (self.ffa_states if self.ffa else self.profile["states"]):
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
            elif et in ("pickup", "upgrade", "took", "kill"):
                log(f"[cbot] t={c.t:.1f} {et} {ev.get('kind', ev.get('cubes', ev.get('id', ev.get('target'))))}")
            elif et == "take_failed":
                log(f"[cbot] take failed {ev.get('reason')}")
        self.kills += sum(1 for ev in c.obs.get("events", []) if ev.get("type") == "kill")
        self.kills += c.obs.get("skipped_kills", 0)
        self.detect_ffa(c)
        for _d, m in c.npcs:
            self.npc_ids.add(m["id"])
        for s in c.obs.get("sounds", []):
            if s.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet") \
                    and s.get("dist") in ("near", "mid"):
                self.alert = (s["bearing"], c.t)
        if c.enemies:
            self.last_enemy_t = c.t

    def detect_ffa(self, c):
        """FFA mode switch: see FfaEvidence. Only observes until it fires, so play before detection is unchanged."""
        alive = c.obs.get("alive")
        if isinstance(alive, int):
            self.alive = alive
        if not self.ffa and self.evidence.update(c.obs, self.kills):
            self.ffa = True
            log(f"[cbot] FFA detected t={c.t:.1f} {self.evidence.fired[1]} alive={alive}")

    def ffa_early(self, c):
        return self.ffa and c.t < PARAMS["ffa_hold_t"]

    def mech_attacker(self, c):
        """FFA early: the attacker if it is (or may be) a mech, else None."""
        if not self.threat_recent(c, PARAMS["threat_memory_s"]) or self.threat[2] in self.npc_ids:
            return None
        return self.attacker(c) or (None, None)

    def ffa_note(self, c, branch):
        """Count an FFA branch once per decision; log its first use and every 50th decision after."""
        if self.ffa_tick.get(branch) == c.obs["tick"]:
            return
        self.ffa_tick[branch] = c.obs["tick"]
        n = self.ffa_counts.get(branch, 0) + 1
        self.ffa_counts[branch] = n
        if n == 1 or n % 50 == 0:
            log(f"[cbot] t={c.t:.1f} ffa {branch} x{n}")

    def ffa_final(self):
        return self.alive is not None and self.alive <= PARAMS["ffa_final_alive"]

    def watchers(self, c):
        """Visible mechs near enough to matter, nearest first (capped: each costs a LOS march)."""
        P = PARAMS
        return [m for d, m in c.enemies if d < P["ffa_seen_r"]][:P["ffa_seen_max"]]

    @staticmethod
    def mech_eye(m):
        return (m["pos"][0], m["pos"][1], m["pos"][2] + 8.0)

    def seen_by(self, c):
        """Mechs that have line of sight to us now. Cached per decision."""
        if self.seen_tick is not None and self.seen_tick[0] == c.obs["tick"]:
            return self.seen_tick[1]
        me = (c.x, c.y, c.z + 6.0)
        seen = [m for m in self.watchers(c) if self.world.los(self.mech_eye(m), me)]
        self.seen_tick = (c.obs["tick"], seen)
        return seen

    def hidden_at(self, x, y, mechs):
        p = (x, y, self.world.ground(x, y) + 6.0)
        return all(not self.world.los(self.mech_eye(m), p) for m in mechs)

    def find_cover(self, c, seen):
        """Nearest walkable point within reach that none of the watchers can see, preferring sideways or away moves.
        Only points no farther from the next circle than we are. Bounded: dirs x radii samples, early out."""
        P = PARAMS
        w = self.world
        ax = sum(m["pos"][0] - c.x for m in seen)
        ay = sum(m["pos"][1] - c.y for m in seen)
        ax, ay = unit(ax, ay)
        cands = []
        n = P["ffa_cover_dirs"]
        for r in P["ffa_cover_r"]:
            for k in range(n):
                a = 2 * math.pi * k / n
                dx, dy = math.cos(a), math.sin(a)
                x, y = c.x + dx * r, c.y + dy * r
                if not w.free(x, y) or not self.toward_region(c, x, y):
                    continue
                toward = dx * ax + dy * ay  # 1 = straight at them
                cands.append((r * (1.0 + 0.5 * max(0.0, toward)), x, y))
        cands.sort()
        my_comp = w.component(c.x, c.y)
        for _s, x, y in cands:
            if my_comp and w.component(x, y) != my_comp:
                continue
            if self.hidden_at(x, y, seen):
                return x, y
        return None

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

    def region_penalty(self, c, dx, dy, ahead=12.0, fight=False):
        cx, cy, r = c.region
        out = math.hypot(c.x + dx * ahead - cx, c.y + dy * ahead - cy) - max(r, 10.0)
        zr = c.zone["radius"]
        out_now = math.hypot(c.x + dx * ahead - c.zone["center"][0], c.y + dy * ahead - c.zone["center"][1]) - zr
        w = PARAMS["ffa_region_w"] if fight and self.ffa and (c.t >= c.zone["shrink_start"] or self.ffa_final()) else 0.03
        return w * max(0.0, out) + 0.2 * max(0.0, out_now + 15.0)

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
        slack = PARAMS["zone_slack_s"]
        if self.ffa:  # a harsher circle leaves less room for a late fight or a slow, charging walk
            slack += PARAMS["ffa_zone_slack_s"] + PARAMS["ffa_zone_slack_dps"] * finite(c.zone.get("dps"), 20.0)
        return d / speed + slack > time_left

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
        ring = self.ffa and r >= P["ffa_ring_min_r"]
        rep_r = P["ffa_spot_enemy_r"] if self.ffa else 100.0
        pool = []
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
                if ring:  # middle ring: no pull to the centre inside ffa_ring_out, a push off it inside ffa_ring_in
                    s = math.hypot(x - c.x, y - c.y) + P["spot_w_cover"] * w.cs * (cov - 1) ** 2 * 0.5 + \
                        P["spot_w_center"] * max(0.0, dc - P["ffa_ring_out"] * r) + P["ffa_ring_w"] * max(0.0, P["ffa_ring_in"] * r - dc)
                else:
                    s = math.hypot(x - c.x, y - c.y) + P["spot_w_cover"] * w.cs * (cov - 1) ** 2 * 0.5 + P["spot_w_center"] * dc
                if corner:
                    walls = sum(w.blocked[cc + o] for o in (1, -1, w.nw, -w.nw, w.nw + 1, w.nw - 1, -w.nw + 1, -w.nw - 1))
                    s -= 15.0 * min(walls, 4)
                for tx, ty, _tz in threats:
                    dd = math.hypot(x - tx, y - ty)
                    if dd < rep_r:
                        s += P["spot_w_enemy"] * (1 - dd / rep_r)
                if s < bs:
                    best, bs = (x, y), s
                if self.ffa:
                    pool.append((s, x, y))
        if self.ffa and pool:  # fewest watchers: line of sight only for the best few cells
            watch = [m for _d, m in c.enemies[:P["ffa_seen_max"]]]
            pool.sort()
            scored = []
            for s, x, y in pool[:P["ffa_spot_los_top"]]:
                p = (x, y, w.ground(x, y) + 6.0)
                s += P["ffa_spot_seen_cost"] * sum(1 for m in watch if math.hypot(m["pos"][0] - x, m["pos"][1] - y)
                                                   < P["ffa_seen_r"] and w.los(self.mech_eye(m), p))
                scored.append((s, x, y))
            _s, x, y = min(scored)
            best = (x, y)
            self.ffa_note(c, "ring-spot" if ring else "spot")
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
        self.spot_t = c.t
        log(f"[cbot] t={c.t:.1f} spot ({x:.0f},{y:.0f}) cover={w.cover[cc]} region=({cx:.0f},{cy:.0f},{r:.0f})")

    def ensure_spot(self, c):
        cx, cy, r = c.region
        if self.spot is None or self.spot_key != (round(cx), round(cy), round(r)) \
                or math.hypot(self.spot[0] - cx, self.spot[1] - cy) > max(r, 8.0):
            self.choose_spot(c)
        elif self.ffa and c.t - self.spot_t > PARAMS["ffa_respot_s"] and any(
                math.hypot(m["pos"][0] - self.spot[0], m["pos"][1] - self.spot[1]) < PARAMS["ffa_respot_r"]
                for _d, m in c.enemies[:4]):
            self.ffa_note(c, "respot")
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
        if self.ffa and self.state == "ESCAPE" and not self.zone_urgent(c):
            # finish the walk in (to well inside the next circle, or to the spot), or a fight or a pickup at the edge
            # drags us straight back out
            at_spot = self.spot is not None and math.hypot(self.spot[0] - c.x, self.spot[1] - c.y) < PARAMS["spot_reach"]
            return not at_spot and c.dist_region() > -min(PARAMS["ffa_escape_in"], 0.3 * c.region[2])
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

    def aggro_target(self, c):
        """Kill-focus target: stick to the locked mech, else the nearest weak one. Cached per decision."""
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
        if self.ffa_early(c):  # no closing on mechs early: fight back only when cornered
            att = self.mech_attacker(c)
            return att if att and att[1] is not None and att[0] < P["retreat_gives_up_dist"] else None
        if self.ffa:
            return self.ffa_target(c)
        if self.strategy == "aggro":
            return self.aggro_target(c)
        att = self.attacker(c)
        if att and att[0] < P["fire_range"]:
            return att
        for d, m in c.enemies:
            if d < P["engage_range"] or self.strategy == "aggro":
                return d, m
        return None

    def ffa_target(self, c):
        """FFA after the hold. A mech hitting us is fought, unless others are near and it is not on top of us
        (then EVADE breaks line of sight). Otherwise engage only a lone mech: a duel with others watching
        invites a third party. In the final circle the crowd rule lapses."""
        P = PARAMS
        att = self.mech_attacker(c)
        if att is not None:
            if att[1] is None:
                return None
            crowd = any(d < P["ffa_crowd_r"] and m["id"] != att[1]["id"] for d, m in c.enemies)
            if att[0] < P["retreat_gives_up_dist"] or not crowd or self.ffa_final():
                return att
            self.ffa_note(c, "evade-crowd")
            return None
        near = [e for e in c.enemies if e[0] < P["ffa_crowd_r"]]
        if not near or near[0][0] > P["ffa_engage_r"]:
            return None
        if len(near) == 1 or self.ffa_final():
            return near[0]
        self.ffa_note(c, "decline-crowd")
        return None

    def want_fight(self, c):
        return self.fight_target(c) is not None

    def want_evade(self, c):
        return self.ffa and self.mech_attacker(c) is not None and self.fight_target(c) is None

    def want_cover(self, c):
        """FFA: when a mech can see us, walk to a nearby point none of them can see, and stay there a while.
        Bounded: one search per ffa_cover_retry_s, only points toward the next circle, never in the final circle
        after the hold, and ESCAPE/DODGE/EVADE/FIGHT all take priority."""
        P = PARAMS
        if not self.ffa or (self.ffa_final() and not self.ffa_early(c)):
            self.cover = None
            return False
        seen = self.seen_by(c)
        if self.cover is not None:
            x, y, until = self.cover
            if c.t >= until or not self.toward_region(c, x, y) or \
                    (seen and math.hypot(x - c.x, y - c.y) < P["ffa_cover_reach"]):
                self.cover = None
        if self.cover is None and seen and c.t >= self.cover_retry_t:
            self.cover_retry_t = c.t + P["ffa_cover_retry_s"]
            p = self.find_cover(c, seen)
            if p is None:
                self.ffa_note(c, "cover-none")
            else:
                x, y = p
                self.cover = (x, y, c.t + P["ffa_cover_hold_s"])
                self.ffa_note(c, "cover")
                log(f"[cbot] t={c.t:.1f} ffa cover ({x:.0f},{y:.0f}) {math.hypot(x - c.x, y - c.y):.0f} m, seen by {len(seen)}")
                cx, cy, r = c.region
                if math.hypot(x - cx, y - cy) <= max(r, 8.0):  # hold here instead of walking back into view
                    m = seen[0]
                    self.spot = (x, y, math.atan2(m["pos"][1] - y, m["pos"][0] - x))
                    self.spot_key = (round(cx), round(cy), round(r))
                    self.spot_t = c.t
        return self.cover is not None

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
        pref = PARAMS["weapon_pref"]
        if kind not in pref:
            return False
        held = [w["kind"] for w in c.me["weapons"] if w]
        if kind in held or (self.ffa and kind == "mortar"):
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
        """Inside the next circle, or no farther from it than we already are (early, before the first shrink)."""
        return c.dist_region(x, y) <= max(0.0, c.dist_region()) + 5.0

    def slot1_empty(self, c):
        w = c.me["weapons"]
        return len(w) < 2 or w[1] is None

    def fight_loot_target(self, c, d_tgt):
        """Loot worth grabbing mid-fight: cubes at our feet (kill drops), or a first real weapon nearer than the target."""
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
                # the first real weapon is worth more than closing on a far target: reach scales with its distance
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
            lo, hi = P["aggro_min_dist"], P["aggro_close_dist"]
        if self.ffa:  # stand off; close in only with lightning on a target already losing
            lo, hi = P["ffa_fight_min"], P["ffa_fight_max"]
            hp, mhp = m.get("hp"), m.get("max_hp")
            frac = hp / mhp if isinstance(hp, (int, float)) and isinstance(mhp, (int, float)) and mhp > 0 else 1.0
            if frac < P["ffa_lt_weak"] and any(w and w["kind"] == "lightning" for w in c.me["weapons"]):
                lo, hi = P["aggro_min_dist"], P["ffa_lt_close"]
            self.ffa_note(c, "fight-back" if self.mech_attacker(c) else "engage")
        if d < lo:
            radial = -1.0
        elif d > hi:
            radial = 1.0 if self.strategy == "aggro" else 0.0
        if self.strategy == "aggro":
            it, ld = self.fight_loot_target(c, d)
            if it is not None and self.fight_loot and self.fight_loot[0] == it["id"] \
                    and c.t - self.fight_loot[1] > self.fight_loot[2]:
                log(f"[cbot] t={c.t:.1f} fight loot give up {ld:.0f} m")
                self.loot_skip[it["id"]] = c.t + 20.0
                self.fight_loot = None
                it, ld = None, 0.0
            if it is not None:
                if not self.fight_loot or self.fight_loot[0] != it["id"]:
                    log(f"[cbot] t={c.t:.1f} fight loot {ld:.0f} m, target {d:.0f} m")
                    self.fight_loot = (it["id"], c.t, P["fight_loot_give_up_s"] + ld / 12.0)
                x, y = it["pos"][0], it["pos"][1]
                lx, ly = unit(x - c.x, y - c.y)
                # local heading only: a flow field per item would churn the shared path search mid-fight
                mv = (lx, ly) if ld < 5 else self.best_heading(c, lambda dx, dy: dx * lx + dy * ly, length=min(10.0, ld))
                return mv, ld > 6 and c.me["energy"] > self.boost_floor(c, d), False, math.atan2(by, bx)
            self.fight_loot = None
        if self.strategy == "aggro" and d > hi and not self.ffa:
            boost = c.me["energy"] > self.boost_floor(c, d)
            if d < P["aggro_direct_dist"] and c.z - c.ground < 3.0 and self.world.clear_ray(c.x, c.y, bx, by, d - hi * 0.5):
                mv = self.best_heading(c, lambda dx, dy: dx * bx + dy * by - self.region_penalty(c, dx, dy), length=8.0)
            else:
                mv = self.go(c, round(m["pos"][0] / 20) * 20, round(m["pos"][1] / 20) * 20, hi * 0.8)
            return mv, boost, False, math.atan2(by, bx)
        gx, gy = px + radial * bx, py + radial * by
        mv = self.best_heading(c, lambda dx, dy: dx * gx + dy * gy - self.region_penalty(c, dx, dy, fight=True), length=8.0)
        return mv, False, False, math.atan2(by, bx)

    def boost_floor(self, c, d):
        """Energy kept for lightning once the target is near: boosting would empty the arc's magazine."""
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
        if self.strategy == "aggro":
            return self.do_push(c)
        return self.do_rotate(c)

    def do_push(self, c):
        """No one in sight: walk into the middle of the next circle, where the others converge."""
        cx, cy, r = c.region
        dist = math.hypot(cx - c.x, cy - c.y)
        mv = self.go(c, cx, cy, max(10.0, min(40.0, r * 0.5)))
        boost = self.zone_urgent(c) or (c.me["energy"] > 70 and dist > 60)
        heading = math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else math.atan2(cy - c.y, cx - c.x)
        return mv, boost, False, self.threat_look(c, self.watch_yaw(c, heading))

    def do_evade(self, c):
        """FFA and shot by a mech we should not duel: back off toward cover to break line of sight."""
        self.ffa_note(c, "evade")
        return self.do_retreat(c)

    def do_cover(self, c):
        x, y, _until = self.cover
        d = math.hypot(x - c.x, y - c.y)
        mv = (0.0, 0.0) if d < 2.0 else self.go(c, x, y, 2.0)
        if d < 4.0:
            mv = (mv[0] * 0.3, mv[1] * 0.3)
        look = math.atan2(c.enemies[0][1]["pos"][1] - c.y, c.enemies[0][1]["pos"][0] - c.x) if c.enemies else \
            (math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else c.me["aim"][0])
        return mv, False, False, self.threat_look(c, look)

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
    def pick_target(self, c):
        P = PARAMS
        policy = self.profile["fire"]
        attacked = self.threat_recent(c, P["threat_memory_s"])
        if policy == "attacked" and not attacked:
            return None
        if self.state == "RETREAT" and c.t - self.last_damage_t > P["retreat_quiet_fire_s"]:
            return None  # once the hits stop, dealing damage would only reset the regen clock
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
        if self.ffa_early(c):  # no unprovoked hit on a mech: only return fire while being hit
            hit_back = att is not None and att[1].get("kind") == "mech" and self.threat[2] not in self.npc_ids \
                and c.t - self.last_damage_t < P["retreat_quiet_fire_s"]
            cands = [e for e in cands if e[1].get("kind") != "mech" or (hit_back and e[1]["id"] == att[1]["id"])]
        elif self.ffa:  # after the hold: a mech only if it is our fight target or hitting us
            ft = self.fight_target(c)
            ok = {e[1]["id"] for e in (ft, att) if e and e[1] is not None}
            cands = [e for e in cands if e[1].get("kind") != "mech" or e[1]["id"] in ok]
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
                if lw.get("charge", 0) > 0 or lw["cooldown"] <= 0:
                    aim_yaw, aim_pitch = direct_yaw, direct_pitch
                    fire[laser_slot] = not (charged and self.aimed(c, direct_yaw, direct_pitch, tol))
                    if fire[laser_slot] and gun is not None:
                        fire[gun] = fire[gun] and self.aimed(c, direct_yaw, direct_pitch, tol)
        elif laser_slot is not None and weapons[laser_slot].get("charge", 0) > 0:
            fire[laser_slot] = False
            if self.ffa_early(c):  # a released charge fires along the aim: raise it into the sky first
                aim_pitch = c.stats.get("max_pitch", 1.3)
                fire[laser_slot] = c.me["aim"][1] < aim_pitch - 0.15
        else:
            cr = self.crate_target(c)
            if cr is not None:
                aim_yaw, aim_pitch, fire = self.shoot_crate(c, cr, weapons)
        mp = c.stats.get("max_pitch", 1.3)
        np_ = c.stats.get("min_pitch", -0.7)
        aim_pitch = min(mp, max(np_, finite(aim_pitch)))
        return [round(wrap(finite(aim_yaw)), 4), round(aim_pitch, 4)], fire

    def crate_target(self, c):
        """A crate worth a few shots: nothing else to shoot, near, on our level, healthy enough. One LOS check."""
        P = PARAMS
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
        if self.crate_id != best["id"]:
            self.crate_id = best["id"]
            log(f"[cbot] t={c.t:.1f} crate {best['id']} {bd:.0f} m hp={best.get('hp')}")
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
        if self.ffa and "take" not in act:  # FFA: lasers cook off a carried mortar
            slot = next((i for i, w in enumerate(weapons) if w and w["kind"] == "mortar"), None)
            if slot is not None:
                act["drop"] = slot
        gun = weapons[0] if weapons and weapons[0] and weapons[0]["kind"] == "gun" else None
        if gun and not act["fire"][0] and gun.get("reload", 0) <= 0 \
                and gun.get("ammo", 99) < gun["stats"].get("mag", 12) * 0.5 \
                and not any(d < P["crate_min_target"] for d, _m in c.enemies):
            act["reload"] = True


if __name__ == "__main__":
    mf.run(Bot)
