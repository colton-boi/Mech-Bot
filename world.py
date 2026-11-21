"""Static map model: terrain heights, rasterized building prisms, line of sight, a walkability grid,
cover distances, a time-sliced flow field for navigation, and roof candidates for climbing."""
import json
import math
import os
import sys
import time
from array import array
from collections import deque

NONE_TOP = -1e9
SQRT2 = math.sqrt(2.0)


class World:
    def __init__(self, map_dir, files, mech_cfg, raster=2.0, cell=4.0):
        t0 = time.perf_counter()
        world_rel = next((f for f in files if f.endswith("world.json")), "v2/world.json")
        world_path = os.path.join(map_dir, world_rel)
        with open(world_path) as f:
            w = json.load(f)
        tr = w["terrain"]
        self.tx0, self.ty0, self.tstep = tr["minX"], tr["minY"], tr["step"]
        self.tnx, self.tny = tr["nx"], tr["ny"]
        raw = array("h")
        with open(os.path.join(os.path.dirname(world_path), tr.get("file", "terrain.i16")), "rb") as f:
            raw.frombytes(f.read())
        if sys.byteorder == "big":
            raw.byteswap()
        scale = tr.get("scale", 0.01)
        self.h = array("f", [v * scale for v in raw])
        del raw

        self.step_up = mech_cfg.get("step_up", 1.2)
        self.head = mech_cfg.get("height", 9.0)
        self._rasterize(w["prisms"], raster)
        self._roofs = self._roof_candidates(w["prisms"])
        del w
        self._nav_grid(cell, mech_cfg.get("max_slope_deg", 50))
        self.load_ms = (time.perf_counter() - t0) * 1000

    # ---------- terrain ----------
    def ground(self, x, y):
        fx = (x - self.tx0) / self.tstep
        fy = (y - self.ty0) / self.tstep
        nx, ny = self.tnx, self.tny
        i = min(max(int(fx), 0), nx - 2)
        j = min(max(int(fy), 0), ny - 2)
        ax = min(max(fx - i, 0.0), 1.0)
        ay = min(max(fy - j, 0.0), 1.0)
        h = self.h
        k = j * nx + i
        top = h[k] + (h[k + 1] - h[k]) * ax
        bot = h[k + nx] + (h[k + nx + 1] - h[k + nx]) * ax
        return top + (bot - top) * ay

    # ---------- prisms ----------
    def _rasterize(self, prisms, rs):
        self.rs = rs
        self.rx0, self.ry0 = self.tx0, self.ty0
        self.rw = int(math.ceil(self.tnx * self.tstep / rs))
        self.rh = int(math.ceil(self.tny * self.tstep / rs))
        n = self.rw * self.rh
        self.rtop = array("f", [NONE_TOP]) * n
        self.rbase = array("f", [1e9]) * n
        self.rwall = bytearray(n)
        x0, y0, rw, rh = self.rx0, self.ry0, self.rw, self.rh
        for p in prisms:
            ring = p["ring"]
            edges = []
            for poly in [ring] + list(p.get("holes") or []):
                m = len(poly) // 2
                for k in range(m):
                    ax, ay = poly[2 * k], poly[2 * k + 1]
                    bx, by = poly[(2 * k + 2) % (2 * m)], poly[(2 * k + 3) % (2 * m)]
                    if ay != by:
                        edges.append((ax, ay, bx, by))
            ys = ring[1::2]
            j0 = max(0, int(math.ceil((min(ys) - y0) / rs - 0.5)))
            j1 = min(rh - 1, int(math.floor((max(ys) - y0) / rs - 0.5)))
            base, top = p["base"], p["top"]
            bridge = p["kind"] == "bridge"
            for j in range(j0, j1 + 1):
                yc = y0 + (j + 0.5) * rs
                xs = sorted(ax + (yc - ay) * (bx - ax) / (by - ay)
                            for ax, ay, bx, by in edges if (ay <= yc) != (by <= yc))
                for k in range(0, len(xs) - 1, 2):
                    i0 = max(0, int(math.ceil((xs[k] - x0) / rs - 0.5)))
                    i1 = min(rw - 1, int(math.floor((xs[k + 1] - x0) / rs - 0.5)))
                    for i in range(i0, i1 + 1):
                        c = j * rw + i
                        if top > self.rtop[c]:
                            self.rtop[c] = top
                        if base < self.rbase[c]:
                            self.rbase[c] = base
                        if not bridge:
                            g = self.ground(x0 + (i + 0.5) * rs, yc)
                            if top - g > self.step_up and base - g < self.head:
                                self.rwall[c] = 1

    def rcell(self, x, y):
        i = int((x - self.rx0) / self.rs)
        j = int((y - self.ry0) / self.rs)
        if 0 <= i < self.rw and 0 <= j < self.rh:
            return j * self.rw + i
        return -1

    def roof_top(self, x, y):
        c = self.rcell(x, y)
        return self.rtop[c] if c >= 0 else NONE_TOP

    def los(self, a, b, step=1.5):
        """True if the segment a->b (3D) clears every prism and the terrain."""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < 1e-6:
            return True
        n = int(d / step)
        if n <= 1:
            return True
        rtop, rbase, rs, x0, y0, rw, rh = self.rtop, self.rbase, self.rs, self.rx0, self.ry0, self.rw, self.rh
        h, tx0, ty0, ts, tnx, tny = self.h, self.tx0, self.ty0, self.tstep, self.tnx, self.tny
        for k in range(1, n):
            f = k / n
            x, y, z = a[0] + dx * f, a[1] + dy * f, a[2] + dz * f
            i, j = int((x - x0) / rs), int((y - y0) / rs)
            if 0 <= i < rw and 0 <= j < rh:
                c = j * rw + i
                if rtop[c] > NONE_TOP and rbase[c] <= z <= rtop[c]:
                    return False
            ti, tj = int((x - tx0) / ts + 0.5), int((y - ty0) / ts + 0.5)
            if 0 <= ti < tnx and 0 <= tj < tny and z < h[tj * tnx + ti] - 0.3:
                return False
        return True

    # ---------- walkability ----------
    def _nav_grid(self, cell, max_slope_deg):
        self.cs = cell
        k = max(1, int(round(cell / self.rs)))
        self.nw = self.rw // k
        self.nh = self.rh // k
        nw, nh = self.nw, self.nh
        n = nw * nh
        blocked = bytearray(n)
        gh = array("f", [0.0]) * n
        rwall, rw = self.rwall, self.rw
        for j in range(nh):
            for i in range(nw):
                c = j * nw + i
                gh[c] = self.ground(*self.ncenter(c))
                if i == 0 or j == 0 or i == nw - 1 or j == nh - 1:
                    blocked[c] = 1
                    continue
                for dj in range(k):
                    base = (j * k + dj) * rw + i * k
                    if any(rwall[base:base + k]):
                        blocked[c] = 1
                        break
        lim = cell * math.tan(math.radians(max_slope_deg))
        for j in range(1, nh - 1):
            for i in range(1, nw - 1):
                c = j * nw + i
                g = gh[c]
                if (abs(gh[c + 1] - g) > lim or abs(gh[c - 1] - g) > lim
                        or abs(gh[c + nw] - g) > lim or abs(gh[c - nw] - g) > lim):
                    blocked[c] = 1
        self.blocked = blocked
        self.nbr = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                    (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]
        # cover[c]: steps to the nearest wall cell, capped at 15
        cover = bytearray([15]) * n
        q = deque()
        for c in range(n):
            if blocked[c]:
                cover[c] = 0
                q.append(c)
        while q:
            c = q.popleft()
            d = cover[c] + 1
            if d >= 15:
                continue
            i, j = c % nw, c // nw
            for di, dj, _ in self.nbr:
                ii, jj = i + di, j + dj
                if 0 <= ii < nw and 0 <= jj < nh:
                    cc = jj * nw + ii
                    if cover[cc] > d:
                        cover[cc] = d
                        q.append(cc)
        self.cover = cover
        # comp[c]: 4-connected walkable component id (0 = blocked); spots in another component need a jet
        comp = array("i", [0]) * n
        label = 0
        for s in range(n):
            if blocked[s] or comp[s]:
                continue
            label += 1
            comp[s] = label
            q.append(s)
            while q:
                c = q.popleft()
                for o in (1, -1, nw, -nw):
                    cc = c + o
                    if 0 <= cc < n and not blocked[cc] and not comp[cc]:
                        comp[cc] = label
                        q.append(cc)
        self.comp = comp
        self.field = None
        self.field_key = None
        self._fields = {}
        self._job = None

    def ncell(self, x, y):
        i = int((x - self.rx0) / self.cs)
        j = int((y - self.ry0) / self.cs)
        if 0 <= i < self.nw and 0 <= j < self.nh:
            return j * self.nw + i
        return -1

    def ncenter(self, c):
        return (self.rx0 + (c % self.nw + 0.5) * self.cs, self.ry0 + (c // self.nw + 0.5) * self.cs)

    def free(self, x, y):
        c = self.ncell(x, y)
        return c >= 0 and not self.blocked[c]

    def component(self, x, y):
        c = self.nearest_free(x, y, 3)
        return self.comp[c] if c >= 0 else 0

    def clear_ray(self, x, y, dx, dy, length, step=2.0):
        d = step
        while d <= length:
            if not self.free(x + dx * d, y + dy * d):
                return False
            d += step
        return True

    def nearest_free(self, x, y, rings=4):
        c = self.ncell(x, y)
        if c >= 0 and not self.blocked[c]:
            return c
        best, bd = -1, 1e18
        i0, j0 = int((x - self.rx0) / self.cs), int((y - self.ry0) / self.cs)
        for dj in range(-rings, rings + 1):
            for di in range(-rings, rings + 1):
                i, j = i0 + di, j0 + dj
                if 0 <= i < self.nw and 0 <= j < self.nh:
                    cc = j * self.nw + i
                    if not self.blocked[cc]:
                        cx, cy = self.ncenter(cc)
                        d = (cx - x) ** 2 + (cy - y) ** 2
                        if d < bd:
                            best, bd = cc, d
        return best

    # ---------- flow field ----------
    def request_field(self, x, y, r):
        """Ask for a flow field toward the disc (x, y, r). Work happens in pump()."""
        key = (int(x // self.cs), int(y // self.cs), int(r // self.cs))
        if key == self.field_key or (self._job and self._job[0] == key):
            return
        if key in self._fields:
            self.field_key, self.field = key, self._fields[key]
            return
        n = self.nw * self.nh
        dist = array("f", [-1.0]) * n
        q = deque()
        rr = max(r, self.cs * 0.75)
        ci, cj = int((x - self.rx0) / self.cs), int((y - self.ry0) / self.cs)
        span = int(rr / self.cs) + 1
        for j in range(max(0, cj - span), min(self.nh, cj + span + 1)):
            for i in range(max(0, ci - span), min(self.nw, ci + span + 1)):
                c = j * self.nw + i
                if self.blocked[c]:
                    continue
                cx, cy = self.ncenter(c)
                if (cx - x) ** 2 + (cy - y) ** 2 <= rr * rr:
                    dist[c] = 0.0
                    q.append(c)
        if not q:
            c = self.nearest_free(x, y, 8)
            if c < 0:
                return
            dist[c] = 0.0
            q.append(c)
        self._job = (key, dist, q)

    def pump(self, budget_s):
        """Advance the pending flow-field search (BFS with octile step costs, label-correcting)."""
        if not self._job:
            return
        key, dist, q = self._job
        blocked, nw, nh, nbr = self.blocked, self.nw, self.nh, self.nbr
        deadline = time.perf_counter() + budget_s
        count = 0
        while q:
            count += 1
            if count & 255 == 0 and time.perf_counter() > deadline:
                return
            c = q.popleft()
            d = dist[c]
            i, j = c % nw, c // nw
            for di, dj, w in nbr:
                ii, jj = i + di, j + dj
                if ii < 0 or jj < 0 or ii >= nw or jj >= nh:
                    continue
                cc = jj * nw + ii
                if blocked[cc]:
                    continue
                if di and dj and (blocked[j * nw + ii] or blocked[jj * nw + i]):
                    continue
                nd = d + w
                old = dist[cc]
                if old < 0 or nd < old - 1e-3:
                    dist[cc] = nd
                    q.append(cc)
        self.field, self.field_key = dist, key
        self._fields[key] = dist
        while len(self._fields) > 4:
            self._fields.pop(next(iter(self._fields)))
        self._job = None

    def field_dir(self, x, y, lookahead=6):
        """Unit (dx, dy) along the flow field, or None if there is no usable field here."""
        if self.field is None:
            return None
        f = self.field
        c = self.nearest_free(x, y, 2)
        if c < 0 or f[c] < 0:
            return None
        if f[c] == 0:
            return (0.0, 0.0)
        nw, nh = self.nw, self.nh
        path = [c]
        cur = c
        for _ in range(lookahead):
            i, j = cur % nw, cur // nw
            best, bd = -1, f[cur]
            for di, dj, _w in self.nbr:
                ii, jj = i + di, j + dj
                if 0 <= ii < nw and 0 <= jj < nh:
                    cc = jj * nw + ii
                    if f[cc] >= 0 and f[cc] < bd - 1e-3:
                        if di and dj and (self.blocked[j * nw + ii] or self.blocked[jj * nw + i]):
                            continue
                        best, bd = cc, f[cc]
            if best < 0:
                break
            cur = best
            path.append(cur)
        for target in reversed(path[1:]):
            tx, ty = self.ncenter(target)
            dx, dy = tx - x, ty - y
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            if target == path[1] or self.clear_ray(x, y, dx / L, dy / L, L, 1.5):
                return (dx / L, dy / L)
        tx, ty = self.ncenter(c)
        dx, dy = tx - x, ty - y
        L = math.hypot(dx, dy)
        return (dx / L, dy / L) if L > 1e-6 else None

    # ---------- roofs ----------
    def _roof_candidates(self, prisms):
        out = []
        for p in prisms:
            if p["kind"] != "building":
                continue
            ring = p["ring"]
            m = len(ring) // 2
            a = cx = cy = 0.0
            for k in range(m):
                x1, y1 = ring[2 * k], ring[2 * k + 1]
                x2, y2 = ring[(2 * k + 2) % (2 * m)], ring[(2 * k + 3) % (2 * m)]
                cr = x1 * y2 - x2 * y1
                a += cr
                cx += (x1 + x2) * cr
                cy += (y1 + y2) * cr
            if abs(a) < 1e-6:
                continue
            cx, cy, area = cx / (3 * a), cy / (3 * a), abs(a) / 2
            # the standing point must be on this roof: the raster top there equals this prism's top
            if abs(self.roof_top(cx, cy) - p["top"]) > 0.5:
                continue
            verts = [(ring[2 * k], ring[2 * k + 1]) for k in range(m)]
            out.append({"c": (cx, cy), "top": p["top"], "area": area, "verts": verts})
        return out

    def roofs(self):
        return self._roofs

    def approach_point(self, roof, fx, fy, offset=5.0):
        """A free ground point just outside the roof's footprint, nearest to (fx, fy)."""
        cx, cy = roof["c"]
        best, bd = None, 1e18
        vs = roof["verts"]
        for k in range(len(vs)):
            ax, ay = vs[k]
            bx, by = vs[(k + 1) % len(vs)]
            for t in (0.25, 0.5, 0.75):
                px, py = ax + (bx - ax) * t, ay + (by - ay) * t
                ex, ey = bx - ax, by - ay
                L = math.hypot(ex, ey)
                if L < 3:
                    continue
                nx, ny = ey / L, -ex / L
                if (px - cx) * nx + (py - cy) * ny < 0:
                    nx, ny = -nx, -ny
                qx, qy = px + nx * offset, py + ny * offset
                if not self.free(qx, qy) or self.roof_top(qx, qy) > NONE_TOP:
                    continue
                d = (qx - fx) ** 2 + (qy - fy) ** 2
                if d < bd:
                    best, bd = (qx, qy, px, py), d
        return best
