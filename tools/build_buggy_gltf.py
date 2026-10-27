#!/usr/bin/env python3
"""Generate the Silt open-wheel buggy glTF.

Blender is not required. The mesh is a tube-cage sand rail in the solver's
frame: Y up, Z forward, origin at the chassis CG. Wheel spin axis is +X.
Mount numbers are duplicated in src/buggy_gltf.rs; the loader test checks
the extras block against those constants.

    python3 tools/build_buggy_gltf.py
"""

from __future__ import annotations

import array
import math
import os
import struct
import sys
import zlib

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "buggy.glb")

# Solver hardpoints. Keep in lockstep with src/buggy_gltf.rs.
HARD_Y = 0.22
REST_LEN = 0.50
REST_COMPRESSION = 0.08
REST_WHEEL_Y = HARD_Y - REST_LEN + REST_COMPRESSION  # -0.20
WHEEL_X = 0.74
WHEEL_RADIUS = 0.38
FRONT_Z = 1.08
REAR_Z = -1.18

LOWER_IN_X = 0.26
LOWER_IN_Y = -0.04
LOWER_HALF_Z = 0.18
LOWER_OUT_X = 0.55
LOWER_OUT_DY = -0.02

UPPER_IN_X = 0.24
UPPER_IN_Y = 0.34
UPPER_HALF_Z = 0.11
UPPER_OUT_X = 0.52
UPPER_OUT_DY = 0.20

TOWER_X = 0.30
TOWER_Y = 0.52
PERCH_DY = 0.12  # above the lower knuckle

PANEL_CAGE = 0.0
PANEL_NOSE = 1.0
PANEL_LEFT = 2.0
PANEL_RIGHT = 3.0
PANEL_REAR = 4.0
PANEL_BELLY = 5.0
PANEL_ENGINE = 10.0
PANEL_SEAT = 11.0
PANEL_LIGHT = 16.0
PANEL_TAIL = 17.0

TAU = math.tau


def clamp(x, a, b):
    return a if x < a else b if x > b else x


def smoothstep(e0, e1, x):
    if e0 == e1:
        return 0.0
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vmul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vlen(a):
    return math.sqrt(vdot(a, a))


def vnorm(a):
    length = vlen(a)
    if length < 1.0e-8:
        return (0.0, 1.0, 0.0)
    return vmul(a, 1.0 / length)


def hash3(p):
    x = p[0] * 12.9898 + p[1] * 78.233 + p[2] * 37.719
    return abs(math.sin(x) * 43758.5453) % 1.0


class Mat:
    def __init__(self, color, rough, metal, mud, panel, emit=0.0, mode="height", vary=True):
        self.color = color
        self.rough = rough
        self.metal = metal
        self.mud = mud
        self.panel = panel
        self.emit = emit
        self.mode = mode
        self.vary = vary


def resolve(mat, p):
    mud = mat.mud
    if mat.mode == "height":
        h = smoothstep(0.36, -0.06, p[1])
        n = hash3(p)
        mud = clamp(max(mat.mud, h * (0.70 + 0.30 * n)), 0.0, 1.0)
    elif mat.mode == "arm":
        mud = clamp(0.10 + 0.78 * smoothstep(0.02, 0.26, p[0]), 0.0, 1.0)
    elif mat.mode == "coil":
        mud = clamp(smoothstep(0.18, 0.92, p[1]) * 0.86, 0.0, 1.0)
    elif mat.mode == "upright":
        mud = clamp(smoothstep(0.80, 0.05, p[1]) * 0.62, 0.0, 1.0)
    col = mat.color
    if mat.vary:
        k = 0.94 + hash3(vadd(p, (2.7, 0.4, 1.1))) * 0.10
        col = (col[0] * k, col[1] * k, col[2] * k)
    return (col, mat.rough, mat.metal, mud, mat.panel, mat.emit)


def orange(panel, rough=0.30, mud=0.0, mode="height"):
    return Mat((0.86, 0.28, 0.045), rough, 0.05, mud, panel, 0.0, mode)


def steel(panel, rough=0.16, metal=0.90, mud=0.0, mode="height", color=(0.66, 0.68, 0.70)):
    return Mat(color, rough, metal, mud, panel, 0.0, mode)


def rubber(mud, rough=0.90):
    return Mat((0.028, 0.025, 0.023), rough, 0.0, mud, 0.0, 0.0, "fixed", vary=False)


class Mesh:
    def __init__(self, name):
        self.name = name
        self.pos = []
        self.nrm = []
        self.uv = []
        self.col = []
        self.metal = []
        self.mud = []
        self.panel = []
        self.emit = []
        self.idx = []
        self.tan = []
        self.joints = {}
        self.extras = {}

    def push(self, p, n, uv, mat):
        col, rough, metal, mud, panel, emit = resolve(mat, p)
        self.pos.append(p)
        self.nrm.append(vnorm(n))
        self.uv.append(uv)
        self.col.append((col[0], col[1], col[2], rough))
        self.metal.append(metal)
        self.mud.append(mud)
        self.panel.append(panel)
        self.emit.append(emit)
        return len(self.pos) - 1

    def tri(self, i0, i1, i2, n_hint):
        p0, p1, p2 = self.pos[i0], self.pos[i1], self.pos[i2]
        face = vcross(vsub(p1, p0), vsub(p2, p0))
        if vdot(face, n_hint) < 0.0:
            self.idx.extend((i0, i2, i1))
        else:
            self.idx.extend((i0, i1, i2))

    def compute_tangents(self):
        n = len(self.pos)
        t1 = [(0.0, 0.0, 0.0)] * n
        t2 = [(0.0, 0.0, 0.0)] * n
        # lists of tuples are shared if multiplied — rebuild properly
        t1 = [(0.0, 0.0, 0.0) for _ in range(n)]
        t2 = [(0.0, 0.0, 0.0) for _ in range(n)]
        for k in range(0, len(self.idx), 3):
            i0, i1, i2 = self.idx[k], self.idx[k + 1], self.idx[k + 2]
            p0, p1, p2 = self.pos[i0], self.pos[i1], self.pos[i2]
            w0, w1, w2 = self.uv[i0], self.uv[i1], self.uv[i2]
            dp1 = vsub(p1, p0)
            dp2 = vsub(p2, p0)
            du1 = w1[0] - w0[0]
            dv1 = w1[1] - w0[1]
            du2 = w2[0] - w0[0]
            dv2 = w2[1] - w0[1]
            det = du1 * dv2 - dv1 * du2
            if abs(det) < 1.0e-8:
                continue
            r = 1.0 / det
            sdir = vmul(vsub(vmul(dp1, dv2), vmul(dp2, dv1)), r)
            tdir = vmul(vsub(vmul(dp2, du1), vmul(dp1, du2)), r)
            for i in (i0, i1, i2):
                t1[i] = vadd(t1[i], sdir)
                t2[i] = vadd(t2[i], tdir)
        self.tan = []
        for i in range(n):
            normal = self.nrm[i]
            t = vsub(t1[i], vmul(normal, vdot(normal, t1[i])))
            if vlen(t) < 1.0e-6:
                t = (1.0, 0.0, 0.0)
            else:
                t = vnorm(t)
            w = 1.0 if vdot(vcross(normal, t), t2[i]) >= 0.0 else -1.0
            self.tan.append((t[0], t[1], t[2], w))


def add_tube(mesh, a, b, radius, mat, segs=12, caps=True, weld=True):
    a = tuple(a)
    b = tuple(b)
    axis = vsub(b, a)
    length = vlen(axis)
    if length < 1.0e-5:
        return
    direction = vmul(axis, 1.0 / length)
    helper = (0.0, 1.0, 0.0) if abs(direction[1]) < 0.85 else (1.0, 0.0, 0.0)
    u = vnorm(vcross(direction, helper))
    v = vnorm(vcross(direction, u))
    rows = []
    for end_i, origin in enumerate((a, b)):
        row = []
        for i in range(segs + 1):
            ang = i / segs * TAU
            n = vadd(vmul(u, math.cos(ang)), vmul(v, math.sin(ang)))
            p = vadd(origin, vmul(n, radius))
            uv = (i / segs, end_i * max(length * 2.4, 0.25))
            row.append(mesh.push(p, n, uv, mat))
        rows.append(row)
    for i in range(segs):
        n_hint = mesh.nrm[rows[0][i]]
        mesh.tri(rows[0][i], rows[1][i], rows[1][i + 1], n_hint)
        mesh.tri(rows[0][i], rows[1][i + 1], rows[0][i + 1], n_hint)
    if caps:
        ca = mesh.push(a, vmul(direction, -1.0), (0.5, 0.5), mat)
        cb = mesh.push(b, direction, (0.5, 0.5), mat)
        for i in range(segs):
            mesh.tri(ca, rows[0][i + 1], rows[0][i], vmul(direction, -1.0))
            mesh.tri(cb, rows[1][i], rows[1][i + 1], direction)
    if weld:
        for p in (a, b):
            key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
            count, rad = mesh.joints.get(key, (0, 0.0))
            mesh.joints[key] = (count + 1, max(rad, radius))


def add_sphere(mesh, c, radius, mat, segs=10, rings=6):
    c = tuple(c)
    top = mesh.push(vadd(c, (0.0, radius, 0.0)), (0.0, 1.0, 0.0), (0.5, 0.0), mat)
    bot = mesh.push(vadd(c, (0.0, -radius, 0.0)), (0.0, -1.0, 0.0), (0.5, 1.0), mat)
    rows = []
    for i in range(1, rings):
        phi = i / rings * math.pi
        row = []
        for j in range(segs + 1):
            th = j / segs * TAU
            n = (
                math.sin(phi) * math.cos(th),
                math.cos(phi),
                math.sin(phi) * math.sin(th),
            )
            p = vadd(c, vmul(n, radius))
            row.append(mesh.push(p, n, (j / segs, i / rings), mat))
        rows.append(row)
    for j in range(segs):
        mesh.tri(top, rows[0][j], rows[0][j + 1], (0.0, 1.0, 0.0))
        mesh.tri(bot, rows[-1][j + 1], rows[-1][j], (0.0, -1.0, 0.0))
    for i in range(len(rows) - 1):
        for j in range(segs):
            n = mesh.nrm[rows[i][j]]
            mesh.tri(rows[i][j], rows[i + 1][j], rows[i + 1][j + 1], n)
            mesh.tri(rows[i][j], rows[i + 1][j + 1], rows[i][j + 1], n)


def weld_joints(mesh, mat):
    for key, (count, radius) in mesh.joints.items():
        if count >= 2 and radius > 0.0:
            add_sphere(mesh, key, radius * 1.28, mat, segs=8, rings=5)
    mesh.joints = {}


def frames_along(pts):
    frames = []
    u = None
    for i in range(len(pts)):
        if i + 1 < len(pts):
            tang = vnorm(vsub(pts[i + 1], pts[i]))
        else:
            tang = vnorm(vsub(pts[i], pts[i - 1]))
        if u is None:
            helper = (0.0, 1.0, 0.0) if abs(tang[1]) < 0.85 else (1.0, 0.0, 0.0)
            u = vnorm(vcross(tang, helper))
        else:
            u = vsub(u, vmul(tang, vdot(u, tang)))
            if vlen(u) < 1.0e-5:
                helper = (0.0, 1.0, 0.0) if abs(tang[1]) < 0.85 else (1.0, 0.0, 0.0)
                u = vcross(tang, helper)
            u = vnorm(u)
        v = vnorm(vcross(tang, u))
        frames.append((u, v, tang))
    return frames


def add_swept_tube(mesh, pts, radius, mat, sides=8, caps=True):
    if len(pts) < 2:
        return
    frames = frames_along(pts)
    rows = []
    for i, p in enumerate(pts):
        u, v, _tang = frames[i]
        row = []
        for s in range(sides + 1):
            ang = s / sides * TAU
            n = vadd(vmul(u, math.cos(ang)), vmul(v, math.sin(ang)))
            pos = vadd(p, vmul(n, radius))
            row.append(mesh.push(pos, n, (s / sides, i * 0.17), mat))
        rows.append(row)
    for i in range(len(pts) - 1):
        for s in range(sides):
            n_hint = mesh.nrm[rows[i][s]]
            mesh.tri(rows[i][s], rows[i + 1][s], rows[i + 1][s + 1], n_hint)
            mesh.tri(rows[i][s], rows[i + 1][s + 1], rows[i][s + 1], n_hint)
    if caps:
        mesh.tri  # keep linters quiet if caps path is expanded below
        c0 = mesh.push(pts[0], vmul(frames[0][2], -1.0), (0.5, 0.0), mat)
        c1 = mesh.push(pts[-1], frames[-1][2], (0.5, 1.0), mat)
        for s in range(sides):
            mesh.tri(c0, rows[0][s + 1], rows[0][s], vmul(frames[0][2], -1.0))
            mesh.tri(c1, rows[-1][s], rows[-1][s + 1], frames[-1][2])


def add_helix(mesh, y0, y1, coil_r, tube_r, turns, mat, steps_per=8, sides=7):
    n = max(int(turns * steps_per), 4)
    pts = []
    for i in range(n + 1):
        t = i / n
        ang = t * turns * TAU
        y = y0 + (y1 - y0) * t
        pts.append((coil_r * math.cos(ang), y, coil_r * math.sin(ang)))
    add_swept_tube(mesh, pts, tube_r, mat, sides=sides, caps=True)


def add_box(mesh, center, half, mat):
    c = tuple(center)
    hx, hy, hz = half
    faces = [
        # +Y
        ((-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (0.0, 1.0, 0.0)),
        ((-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz), (0.0, -1.0, 0.0)),
        ((hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz), (1.0, 0.0, 0.0)),
        ((-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz), (-1.0, 0.0, 0.0)),
        ((-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz), (0.0, 0.0, 1.0)),
        ((hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz), (0.0, 0.0, -1.0)),
    ]
    for quad in faces:
        n = quad[4]
        ids = []
        for corner in quad[:4]:
            p = vadd(c, corner)
            uv = (corner[0] * 2.0 + 0.5, corner[1] * 2.0 + corner[2])
            ids.append(mesh.push(p, n, uv, mat))
        mesh.tri(ids[0], ids[1], ids[2], n)
        mesh.tri(ids[0], ids[2], ids[3], n)


def add_oriented_box(mesh, center, ax, ay, az, half, mat):
    def corner(sx, sy, sz):
        return vadd(
            center,
            vadd(vadd(vmul(ax, sx * half[0]), vmul(ay, sy * half[1])), vmul(az, sz * half[2])),
        )

    faces = [
        (ax, (1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1)),
        (vmul(ax, -1), (-1, -1, 1), (-1, -1, -1), (-1, 1, -1), (-1, 1, 1)),
        (ay, (-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)),
        (vmul(ay, -1), (-1, -1, 1), (1, -1, 1), (1, -1, -1), (-1, -1, -1)),
        (az, (-1, -1, 1), (-1, 1, 1), (1, 1, 1), (1, -1, 1)),
        (vmul(az, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1)),
    ]
    for n, a, b, c, d in faces:
        ids = []
        for s in (a, b, c, d):
            p = corner(*s)
            ids.append(mesh.push(p, n, (s[0] * 0.5 + 0.5, s[1] * 0.5 + 0.5), mat))
        mesh.tri(ids[0], ids[1], ids[2], n)
        mesh.tri(ids[0], ids[2], ids[3], n)


def add_annulus(mesh, x, r0, r1, half_t, mat, segs=20, sign=1.0):
    """Ring in the YZ plane, thickness along X. Both faces."""
    rows = []
    for end, ox in ((-1.0, x - half_t), (1.0, x + half_t)):
        row = []
        n = (sign * end, 0.0, 0.0) if False else (end, 0.0, 0.0)
        for r in (r0, r1):
            ring = []
            for i in range(segs + 1):
                ang = i / segs * TAU
                radial = (0.0, math.sin(ang), math.cos(ang))
                p = vadd((ox, 0.0, 0.0), vmul(radial, r))
                ring.append(mesh.push(p, n, (i / segs, 0.0 if r == r0 else 1.0), mat))
            row.append(ring)
        rows.append(row)
        # face
        inner, outer = row
        for i in range(segs):
            if end > 0.0:
                mesh.tri(inner[i], outer[i], outer[i + 1], n)
                mesh.tri(inner[i], outer[i + 1], inner[i + 1], n)
            else:
                mesh.tri(inner[i], outer[i + 1], outer[i], n)
                mesh.tri(inner[i], inner[i + 1], outer[i + 1], n)
    # outer wall
    for i in range(segs):
        ang = (i + 0.5) / segs * TAU
        n = (0.0, math.sin(ang), math.cos(ang))
        a0 = rows[0][1][i]
        a1 = rows[0][1][i + 1]
        b0 = rows[1][1][i]
        b1 = rows[1][1][i + 1]
        mesh.tri(a0, b0, b1, n)
        mesh.tri(a0, b1, a1, n)
    # inner wall
    for i in range(segs):
        ang = (i + 0.5) / segs * TAU
        n = (0.0, -math.sin(ang), -math.cos(ang))
        a0 = rows[0][0][i]
        a1 = rows[0][0][i + 1]
        b0 = rows[1][0][i]
        b1 = rows[1][0][i + 1]
        mesh.tri(a0, b1, b0, n)
        mesh.tri(a0, a1, b1, n)


def add_grid(mesh, nu, nv, fn, mat, thick):
    pts = [[fn(i / (nu - 1), j / (nv - 1)) for j in range(nv)] for i in range(nu)]
    nrm = [[(0.0, 1.0, 0.0) for _ in range(nv)] for _ in range(nu)]
    for i in range(nu):
        for j in range(nv):
            i0, i1 = max(i - 1, 0), min(i + 1, nu - 1)
            j0, j1 = max(j - 1, 0), min(j + 1, nv - 1)
            du = vsub(pts[i1][j], pts[i0][j])
            dv = vsub(pts[i][j1], pts[i][j0])
            n = vcross(du, dv)
            nrm[i][j] = vnorm(n) if vlen(n) > 1.0e-8 else (0.0, 1.0, 0.0)
    front = []
    back = []
    for i in range(nu):
        fr = []
        bk = []
        for j in range(nv):
            n = nrm[i][j]
            p = pts[i][j]
            uv = (i / (nu - 1), j / (nv - 1))
            fr.append(mesh.push(vadd(p, vmul(n, thick * 0.5)), n, uv, mat))
            bk.append(mesh.push(vsub(p, vmul(n, thick * 0.5)), vmul(n, -1.0), uv, mat))
        front.append(fr)
        back.append(bk)
    for i in range(nu - 1):
        for j in range(nv - 1):
            n = nrm[i][j]
            mesh.tri(front[i][j], front[i + 1][j], front[i + 1][j + 1], n)
            mesh.tri(front[i][j], front[i + 1][j + 1], front[i][j + 1], n)
            mesh.tri(back[i][j], back[i + 1][j + 1], back[i + 1][j], vmul(n, -1.0))
            mesh.tri(back[i][j], back[i][j + 1], back[i + 1][j + 1], vmul(n, -1.0))


def add_torus(mesh, center, axis, major_r, tube_r, mat, major_segs=16, tube_segs=8):
    axis = vnorm(axis)
    helper = (0.0, 1.0, 0.0) if abs(axis[1]) < 0.85 else (1.0, 0.0, 0.0)
    u = vnorm(vcross(axis, helper))
    v = vnorm(vcross(axis, u))
    rows = []
    for i in range(major_segs + 1):
        ang = i / major_segs * TAU
        spoke = vadd(vmul(u, math.cos(ang)), vmul(v, math.sin(ang)))
        center_ring = vadd(center, vmul(spoke, major_r))
        side = vnorm(vcross(axis, spoke))
        row = []
        for j in range(tube_segs + 1):
            th = j / tube_segs * TAU
            n = vadd(vmul(spoke, math.cos(th)), vmul(axis, math.sin(th)))
            # keep a stable bitangent so the tube doesn't flip; side is unused
            # but retained so the ring has a frame if we need it later.
            _ = side
            p = vadd(center_ring, vmul(n, tube_r))
            row.append(mesh.push(p, n, (i / major_segs, j / tube_segs), mat))
        rows.append(row)
    for i in range(major_segs):
        for j in range(tube_segs):
            n = mesh.nrm[rows[i][j]]
            mesh.tri(rows[i][j], rows[i + 1][j], rows[i + 1][j + 1], n)
            mesh.tri(rows[i][j], rows[i + 1][j + 1], rows[i][j + 1], n)


def corner_zs():
    return (FRONT_Z, REAR_Z)


def build_chassis():
    m = Mesh("chassis")
    cage = orange(PANEL_CAGE, 0.22)
    nose = orange(PANEL_NOSE, 0.24)
    rear = orange(PANEL_REAR, 0.24)
    left = orange(PANEL_LEFT, 0.24)
    right = orange(PANEL_RIGHT, 0.24)
    main_r = 0.034
    sec_r = 0.020
    brace_r = 0.015

    # Lower frame rails.
    add_tube(m, (-0.36, 0.02, 1.28), (-0.36, 0.02, -1.36), main_r, cage)
    add_tube(m, (0.36, 0.02, 1.28), (0.36, 0.02, -1.36), main_r, cage)
    add_tube(m, (-0.36, 0.02, 1.28), (0.36, 0.02, 1.28), main_r, nose)
    add_tube(m, (-0.36, 0.02, -1.36), (0.36, 0.02, -1.36), main_r, rear)
    for z in (0.55, -0.05, -0.62):
        panel_id = PANEL_NOSE if z > 0.2 else PANEL_REAR if z < -0.4 else PANEL_CAGE
        add_tube(m, (-0.36, 0.02, z), (0.36, 0.02, z), sec_r, orange(panel_id, 0.32))

    # Pointed nose.
    add_tube(m, (-0.36, 0.02, 1.28), (0.0, 0.05, 1.70), sec_r, nose)
    add_tube(m, (0.36, 0.02, 1.28), (0.0, 0.05, 1.70), sec_r, nose)
    add_tube(m, (-0.28, 0.18, 1.05), (0.0, 0.12, 1.62), brace_r, nose)
    add_tube(m, (0.28, 0.18, 1.05), (0.0, 0.12, 1.62), brace_r, nose)
    add_tube(m, (0.0, 0.05, 1.70), (0.0, 0.16, 1.42), brace_r, nose)
    add_tube(m, (-0.22, -0.02, 1.22), (0.0, -0.04, 1.66), brace_r, nose)
    add_tube(m, (0.22, -0.02, 1.22), (0.0, -0.04, 1.66), brace_r, nose)

    # Front hoop.
    add_tube(m, (-0.36, 0.04, 0.78), (-0.40, 0.78, 0.30), main_r, left)
    add_tube(m, (0.36, 0.04, 0.78), (0.40, 0.78, 0.30), main_r, right)
    add_tube(m, (-0.40, 0.78, 0.30), (0.40, 0.78, 0.30), main_r, cage)
    add_tube(m, (-0.36, 0.40, 0.62), (0.36, 0.40, 0.62), sec_r, cage)  # dash bar

    # Main hoop.
    add_tube(m, (-0.38, 0.02, -0.42), (-0.44, 0.98, -0.50), main_r, left)
    add_tube(m, (0.38, 0.02, -0.42), (0.44, 0.98, -0.50), main_r, right)
    add_tube(m, (-0.44, 0.98, -0.50), (0.44, 0.98, -0.50), main_r, cage)
    add_tube(m, (-0.40, 0.78, 0.30), (-0.44, 0.98, -0.50), main_r, left)
    add_tube(m, (0.40, 0.78, 0.30), (0.44, 0.98, -0.50), main_r, right)
    add_tube(m, (0.0, 0.86, 0.18), (0.0, 1.00, -0.48), brace_r, cage)

    # Door bars and diagonals.
    add_tube(m, (-0.38, 0.42, 0.58), (-0.42, 0.50, -0.42), sec_r, left)
    add_tube(m, (0.38, 0.42, 0.58), (0.42, 0.50, -0.42), sec_r, right)
    add_tube(m, (-0.36, 0.06, 0.62), (-0.42, 0.72, -0.08), brace_r, left)
    add_tube(m, (0.36, 0.06, 0.62), (0.42, 0.72, -0.08), brace_r, right)
    add_tube(m, (-0.40, 0.55, -0.28), (0.40, 0.55, -0.28), sec_r, cage)  # harness bar

    # Rear stays and bulkhead.
    add_tube(m, (-0.44, 0.98, -0.50), (-0.36, 0.22, -1.36), main_r, rear)
    add_tube(m, (0.44, 0.98, -0.50), (0.36, 0.22, -1.36), main_r, rear)
    add_tube(m, (-0.36, 0.22, -1.36), (0.36, 0.22, -1.36), sec_r, rear)
    add_tube(m, (-0.36, 0.02, -1.36), (-0.36, 0.48, -1.22), sec_r, rear)
    add_tube(m, (0.36, 0.02, -1.36), (0.36, 0.48, -1.22), sec_r, rear)
    add_tube(m, (-0.34, 0.06, -1.20), (0.28, 0.46, -0.78), brace_r, rear)
    add_tube(m, (0.34, 0.06, -1.20), (-0.28, 0.46, -0.78), brace_r, rear)
    # Nerf bars, still inboard of the tire sidewall.
    add_tube(m, (-0.50, 0.10, 0.35), (-0.50, 0.10, -0.70), sec_r, left)
    add_tube(m, (0.50, 0.10, 0.35), (0.50, 0.10, -0.70), sec_r, right)
    add_tube(m, (-0.36, 0.04, 0.35), (-0.50, 0.10, 0.35), brace_r, left)
    add_tube(m, (0.36, 0.04, 0.35), (0.50, 0.10, 0.35), brace_r, right)
    add_tube(m, (-0.38, 0.04, -0.70), (-0.50, 0.10, -0.70), brace_r, left)
    add_tube(m, (0.38, 0.04, -0.70), (0.50, 0.10, -0.70), brace_r, right)

    # Shock towers and inboard pivots. Endpoints are the poser mounts.
    for sign, z in ((-1.0, FRONT_Z), (1.0, FRONT_Z), (-1.0, REAR_Z), (1.0, REAR_Z)):
        side = left if sign < 0.0 else right
        panel = PANEL_NOSE if z > 0.0 else PANEL_REAR
        paint = orange(panel, 0.30)
        tower = (sign * TOWER_X, TOWER_Y, z)
        foot = (sign * 0.32, 0.06, z)
        add_tube(m, foot, tower, sec_r, paint)
        # Short stubs only. The posed A-arms already run from these pivots,
        # and a full chassis copy of each arm reads as a second lattice.
        for dz, iny, inx in (
            (LOWER_HALF_Z, LOWER_IN_Y, LOWER_IN_X),
            (-LOWER_HALF_Z, LOWER_IN_Y, LOWER_IN_X),
        ):
            pivot = (sign * inx, iny, z + dz)
            rail = (sign * 0.36, 0.02, z)
            stub = vadd(pivot, vmul(vnorm(vsub(rail, pivot)), 0.075))
            add_tube(m, pivot, stub, brace_r, side)
        for dz in (UPPER_HALF_Z, -UPPER_HALF_Z):
            pivot = (sign * UPPER_IN_X, UPPER_IN_Y, z + dz)
            stub = vadd(pivot, vmul(vnorm(vsub(tower, pivot)), 0.07))
            add_tube(m, pivot, stub, brace_r, side)

    # Exhaust. Steel, so the wet metal pass has something to catch.
    ex = steel(PANEL_REAR, 0.22, 0.82, 0.15, "height", (0.42, 0.36, 0.32))
    for sign in (-1.0, 1.0):
        pts = [
            (sign * 0.10, 0.16, -0.78),
            (sign * 0.22, 0.10, -1.05),
            (sign * 0.30, 0.08, -1.32),
            (sign * 0.34, 0.14, -1.58),
        ]
        add_swept_tube(m, pts, 0.028, ex, sides=10, caps=True)
        # Muffler.
        add_tube(
            m,
            (sign * 0.22, 0.08, -1.10),
            (sign * 0.32, 0.10, -1.42),
            0.048,
            steel(PANEL_REAR, 0.28, 0.7, 0.2, "height", (0.28, 0.26, 0.25)),
            segs=10,
        )

    weld_joints(m, cage)

    # Belly skid. Always caked.
    belly = Mat((0.10, 0.08, 0.07), 0.62, 0.02, 0.88, PANEL_BELLY, 0.0, "fixed", vary=True)
    add_box(m, (0.0, -0.045, -0.02), (0.30, 0.012, 0.92), belly)

    # Engine, open behind the seats.
    eng = Mat((0.16, 0.16, 0.17), 0.42, 0.45, 0.28, PANEL_ENGINE, 0.0, "fixed")
    add_box(m, (0.0, 0.20, -0.78), (0.18, 0.13, 0.20), eng)
    add_box(m, (-0.10, 0.34, -0.78), (0.06, 0.035, 0.14), steel(PANEL_ENGINE, 0.25, 0.6, 0.1))
    add_box(m, (0.10, 0.34, -0.78), (0.06, 0.035, 0.14), steel(PANEL_ENGINE, 0.25, 0.6, 0.1))
    add_tube(
        m,
        (0.0, 0.40, -0.70),
        (0.0, 0.40, -0.86),
        0.045,
        Mat((0.05, 0.05, 0.05), 0.55, 0.1, 0.15, PANEL_ENGINE, 0.0, "fixed"),
        segs=10,
        weld=False,
    )

    # Seats.
    seat = Mat((0.030, 0.028, 0.027), 0.74, 0.0, 0.14, PANEL_SEAT, 0.0, "fixed", vary=True)
    harness = Mat((0.02, 0.02, 0.02), 0.6, 0.0, 0.05, PANEL_SEAT, 0.0, "fixed", vary=False)
    for x in (-0.175, 0.175):

        def cushion(u, v, x=x):
            px = x + (u - 0.5) * 0.34
            pz = -0.02 + (v - 0.5) * 0.36
            edge = max(abs(u - 0.5) * 2.0, abs(v - 0.5) * 1.6)
            bolster = smoothstep(0.38, 1.0, edge) * 0.07
            dish = (1.0 - min(edge, 1.0)) * 0.016
            return (px, 0.13 + bolster - dish, pz)

        def back(u, v, x=x):
            px = x + (u - 0.5) * 0.32
            eu = abs(u - 0.5) * 2.0
            bolster = smoothstep(0.36, 1.0, eu)
            pz = -0.18 - v * 0.05 + bolster * 0.05
            py = 0.18 + v * 0.46 + bolster * 0.025
            if v > 0.78:
                py += (v - 0.78) * 0.10
                px += (u - 0.5) * (v - 0.78) * 0.08
            return (px, py, pz)

        add_grid(m, 8, 7, cushion, seat, 0.020)
        add_grid(m, 8, 9, back, seat, 0.018)
        add_tube(m, (x - 0.07, 0.48, -0.20), (x - 0.05, 0.16, 0.04), 0.012, harness, segs=6, weld=False)
        add_tube(m, (x + 0.07, 0.48, -0.20), (x + 0.05, 0.16, 0.04), 0.012, harness, segs=6, weld=False)

    # Steering wheel, left of center.
    add_torus(
        m,
        (-0.14, 0.52, 0.50),
        (0.15, 0.35, 1.0),
        0.11,
        0.012,
        Mat((0.04, 0.04, 0.04), 0.45, 0.05, 0.05, PANEL_CAGE, 0.0, "fixed"),
    )
    add_tube(
        m,
        (-0.14, 0.52, 0.50),
        (-0.06, 0.36, 0.62),
        0.012,
        steel(PANEL_CAGE, 0.3, 0.5, 0.0, "fixed"),
        segs=6,
        weld=False,
    )

    # Wing on the main hoop. Endplates stay orange so a side view is not a black slab.
    add_box(m, (0.0, 1.12, -0.76), (0.48, 0.042, 0.18), orange(PANEL_REAR, 0.24, 0.0))
    plate = orange(PANEL_REAR, 0.30, 0.0)
    add_box(m, (-0.49, 1.14, -0.76), (0.016, 0.058, 0.16), plate)
    add_box(m, (0.49, 1.14, -0.76), (0.016, 0.058, 0.16), plate)
    add_tube(m, (-0.22, 0.98, -0.50), (-0.26, 1.07, -0.66), sec_r, rear, weld=False)
    add_tube(m, (0.22, 0.98, -0.50), (0.26, 1.07, -0.66), sec_r, rear, weld=False)
    add_tube(m, (-0.40, 0.96, -0.50), (-0.40, 1.08, -0.66), brace_r, rear, weld=False)
    add_tube(m, (0.40, 0.96, -0.50), (0.40, 1.08, -0.66), brace_r, rear, weld=False)

    # Lamps.
    lamp = Mat((1.0, 0.94, 0.72), 0.12, 0.0, 0.0, PANEL_LIGHT, 1.35, "fixed", vary=False)
    housing = Mat((0.05, 0.05, 0.05), 0.35, 0.3, 0.05, PANEL_NOSE, 0.0, "fixed")
    for x in (-0.12, 0.12):
        add_sphere(m, (x, 0.12, 1.60), 0.045, lamp, segs=8, rings=5)
        add_tube(m, (x, 0.12, 1.52), (x, 0.12, 1.60), 0.05, housing, segs=8, caps=False, weld=False)
    tail = Mat((0.85, 0.03, 0.015), 0.18, 0.0, 0.0, PANEL_TAIL, 0.9, "fixed", vary=False)
    add_sphere(m, (-0.22, 0.24, -1.40), 0.035, tail, segs=8, rings=5)
    add_sphere(m, (0.22, 0.24, -1.40), 0.035, tail, segs=8, rings=5)

    m.extras = {"silt": "chassis"}
    return m


def build_arm(name, length, half_z, radius):
    m = Mesh(name)
    mat = orange(PANEL_CAGE, 0.26, 0.0, "arm")
    mat.metal = 0.08
    front = (0.0, 0.0, half_z)
    rear = (0.0, 0.0, -half_z)
    out = (length, 0.0, 0.0)
    add_tube(m, front, out, radius, mat, segs=8, weld=False)
    add_tube(m, rear, out, radius, mat, segs=8, weld=False)
    add_tube(m, front, rear, radius * 0.75, mat, segs=8, weld=False)
    add_sphere(m, out, radius * 1.7, mat, segs=8, rings=5)
    add_sphere(m, front, radius * 1.35, mat, segs=7, rings=4)
    add_sphere(m, rear, radius * 1.35, mat, segs=7, rings=4)
    m.extras = {"silt": name, "length": length, "half_z": half_z}
    return m


def build_coil():
    m = Mesh("coil")
    body = steel(PANEL_ENGINE, 0.14, 0.92, 0.0, "coil", (0.72, 0.74, 0.76))
    shaft = steel(PANEL_ENGINE, 0.10, 0.95, 0.0, "coil", (0.80, 0.82, 0.84))
    spring = Mat((0.90, 0.32, 0.05), 0.20, 0.18, 0.0, PANEL_ENGINE, 0.0, "coil")
    perch = orange(PANEL_ENGINE, 0.32, 0.0, "coil")
    add_tube(m, (0.0, 0.02, 0.0), (0.0, 0.56, 0.0), 0.028, body, segs=12, weld=False)
    add_tube(m, (0.0, 0.52, 0.0), (0.0, 1.0, 0.0), 0.012, shaft, segs=8, weld=False)
    # Fat wire and a wide coil so the spring reads at the portrait distance.
    # Outer radius stays inside the tire sidewall when the shock is posed.
    add_helix(m, 0.08, 0.82, 0.058, 0.015, 6.5, spring, steps_per=8, sides=8)
    add_tube(m, (0.0, 0.07, 0.0), (0.0, 0.095, 0.0), 0.074, perch, segs=14, weld=False)
    add_tube(m, (0.0, 0.78, 0.0), (0.0, 0.805, 0.0), 0.074, perch, segs=14, weld=False)
    m.extras = {"silt": "coil", "length": 1.0}
    return m


def build_upright():
    m = Mesh("upright")
    mat = steel(PANEL_ENGINE, 0.22, 0.75, 0.05, "upright", (0.18, 0.18, 0.19))
    add_tube(m, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.026, mat, segs=10, weld=False)
    add_sphere(m, (0.0, 0.0, 0.0), 0.032, mat, segs=8, rings=4)
    add_sphere(m, (0.0, 1.0, 0.0), 0.028, mat, segs=8, rings=4)
    m.extras = {"silt": "upright", "length": 1.0}
    return m


def build_wheel():
    m = Mesh("wheel")
    segs = 28

    def tire_mat(x, r):
        tread = smoothstep(0.28, 0.33, r)
        mud = 0.18 + tread * 0.74
        col = (
            0.040 - tread * 0.018,
            0.036 - tread * 0.016,
            0.032 - tread * 0.014,
        )
        return Mat(col, 0.72 + tread * 0.20, 0.0, mud, 0.0, 0.0, "fixed", vary=False)

    profile = [
        (-0.150, 0.150),
        (-0.136, 0.210),
        (-0.116, 0.275),
        (-0.096, 0.318),
        (-0.070, 0.308),
        (-0.042, 0.300),
        (-0.016, 0.312),
        (0.012, 0.300),
        (0.040, 0.308),
        (0.070, 0.320),
        (0.098, 0.332),
        (0.120, 0.275),
        (0.138, 0.205),
        (0.150, 0.148),
    ]
    rows = []
    for pi, (x, r) in enumerate(profile):
        if pi == 0:
            dx = profile[1][0] - x
            dr = profile[1][1] - r
        elif pi == len(profile) - 1:
            dx = x - profile[pi - 1][0]
            dr = r - profile[pi - 1][1]
        else:
            dx = profile[pi + 1][0] - profile[pi - 1][0]
            dr = profile[pi + 1][1] - profile[pi - 1][1]
        nx, nr = dr, -dx
        if nr < 0.0:
            nx, nr = -nx, -nr
        nx, nr = vnorm((nx, nr, 0.0))[0], vnorm((nx, nr, 0.0))[1]
        row = []
        mat = tire_mat(x, r)
        for i in range(segs + 1):
            ang = i / segs * TAU
            radial = (0.0, math.sin(ang), math.cos(ang))
            n = vnorm(vadd((nx, 0.0, 0.0), vmul(radial, nr)))
            p = vadd((x, 0.0, 0.0), vmul(radial, r))
            uv = (i / segs * 4.0, (x + 0.16) * 3.0)
            row.append(m.push(p, n, uv, mat))
        rows.append(row)
    for pi in range(len(profile) - 1):
        for i in range(segs):
            n = m.nrm[rows[pi][i]]
            m.tri(rows[pi][i], rows[pi][i + 1], rows[pi + 1][i + 1], n)
            m.tri(rows[pi][i], rows[pi + 1][i + 1], rows[pi + 1][i], n)

    # Knobs stand proud of a dropped carcass so the voids silhouette.
    lug = rubber(0.94, 0.92)
    half_circ = 0.058
    half_ax = 0.032
    half_rad = 0.034
    outer_center = WHEEL_RADIUS - 0.004
    for i in range(10):
        for row, xoff in enumerate((-0.048, 0.0, 0.048)):
            if row != 1 and (i + row) % 2 == 0:
                continue
            ang = (i + (0.5 if row == 1 else 0.0)) / 10.0 * TAU
            radial = (0.0, math.sin(ang), math.cos(ang))
            tangent = vnorm(vcross((1.0, 0.0, 0.0), radial))
            if row == 1:
                tangent = vnorm(vadd(vmul(tangent, 0.88), (0.35, 0.0, 0.0)))
                tangent = vnorm(vsub(tangent, vmul(radial, vdot(tangent, radial))))
            axle = vnorm(vcross(radial, tangent))
            center = vadd((xoff, 0.0, 0.0), vmul(radial, outer_center - half_rad))
            add_oriented_box(
                m,
                center,
                tangent,
                axle,
                radial,
                (half_circ, half_ax, half_rad),
                lug,
            )

    # Beadlock rings and bolts, both sides so one mesh works on left and right.
    ring = Mat((0.78, 0.80, 0.83), 0.16, 0.90, 0.10, 0.0, 0.0, "fixed", vary=False)
    bolt = steel(0.0, 0.18, 0.92, 0.08, "fixed", (0.82, 0.83, 0.86))
    for x in (-0.132, 0.132):
        add_annulus(m, x, 0.145, 0.228, 0.010, ring, segs=24)
        for b in range(8):
            ang = b / 8.0 * TAU
            p = (x + (0.012 if x > 0 else -0.012), 0.188 * math.sin(ang), 0.188 * math.cos(ang))
            add_sphere(m, p, 0.014, bolt, segs=6, rings=4)
    # Dish and spokes.
    dish = Mat((0.09, 0.09, 0.095), 0.38, 0.55, 0.22, 0.0, 0.0, "fixed")
    for x in (-0.08, 0.08):
        add_annulus(m, x, 0.05, 0.16, 0.008, dish, segs=16)
    spoke = steel(0.0, 0.25, 0.7, 0.18, "fixed", (0.22, 0.22, 0.24))
    for side in (-1.0, 1.0):
        for s in range(6):
            ang = s / 6.0 * TAU + 0.2
            inner = (side * 0.04, 0.045 * math.sin(ang), 0.045 * math.cos(ang))
            outer = (side * 0.09, 0.155 * math.sin(ang), 0.155 * math.cos(ang))
            add_tube(m, inner, outer, 0.012, spoke, segs=6, weld=False)
    hub = steel(0.0, 0.18, 0.85, 0.12, "fixed")
    add_tube(m, (-0.09, 0.0, 0.0), (0.09, 0.0, 0.0), 0.042, hub, segs=10, weld=False)
    m.extras = {"silt": "wheel", "radius": WHEEL_RADIUS}
    return m


def lower_len():
    dx = LOWER_OUT_X - LOWER_IN_X
    dy = (REST_WHEEL_Y + LOWER_OUT_DY) - LOWER_IN_Y
    return math.hypot(dx, dy)


def upper_len():
    dx = UPPER_OUT_X - UPPER_IN_X
    dy = (REST_WHEEL_Y + UPPER_OUT_DY) - UPPER_IN_Y
    return math.hypot(dx, dy)


def basis_arm(sign, z, wheel_y, in_x, in_y, half_z, out_x, out_dy):
    mid = (sign * in_x, in_y, z)
    knuckle = (sign * out_x, wheel_y + out_dy, z)
    front = (sign * in_x, in_y, z + half_z)
    rear = (sign * in_x, in_y, z - half_z)
    x_axis = vnorm(vsub(knuckle, mid))
    z_hint = vsub(front, rear)
    z_axis = vnorm(vsub(z_hint, vmul(x_axis, vdot(z_hint, x_axis))))
    y_axis = vcross(z_axis, x_axis)
    return mid, x_axis, y_axis, z_axis


def mat_columns(origin, x, y, z):
    # glTF matrix is column-major.
    return [
        x[0], x[1], x[2], 0.0,
        y[0], y[1], y[2], 0.0,
        z[0], z[1], z[2], 0.0,
        origin[0], origin[1], origin[2], 1.0,
    ]


def coil_matrix(tower, perch):
    axis = vsub(perch, tower)
    length = vlen(axis)
    direction = vmul(axis, 1.0 / length)
    helper = (0.0, 1.0, 0.0) if abs(direction[1]) < 0.85 else (1.0, 0.0, 0.0)
    x_axis = vnorm(vcross(helper, direction))
    z_axis = vcross(x_axis, direction)
    y_axis = vmul(direction, length)
    return mat_columns(tower, x_axis, y_axis, z_axis)


def rest_nodes(meshes):
    """Assembled rest pose so a glTF viewer shows the whole buggy."""
    nodes = [{"name": "chassis", "mesh": 0}]
    names = {mesh.name: i for i, mesh in enumerate(meshes)}
    corners = ((-1.0, FRONT_Z), (1.0, FRONT_Z), (-1.0, REAR_Z), (1.0, REAR_Z))
    labels = ("fl", "fr", "rl", "rr")
    for (sign, z), label in zip(corners, labels):
        wy = REST_WHEEL_Y
        nodes.append(
            {
                "name": f"wheel_{label}",
                "mesh": names["wheel"],
                "translation": [sign * WHEEL_X, wy, z],
            }
        )
        mid, x, y, zz = basis_arm(
            sign, z, wy, LOWER_IN_X, LOWER_IN_Y, LOWER_HALF_Z, LOWER_OUT_X, LOWER_OUT_DY
        )
        nodes.append({"name": f"arm_{label}", "mesh": names["arm"], "matrix": mat_columns(mid, x, y, zz)})
        mid, x, y, zz = basis_arm(
            sign, z, wy, UPPER_IN_X, UPPER_IN_Y, UPPER_HALF_Z, UPPER_OUT_X, UPPER_OUT_DY
        )
        nodes.append(
            {
                "name": f"arm_upper_{label}",
                "mesh": names["arm_upper"],
                "matrix": mat_columns(mid, x, y, zz),
            }
        )
        tower = (sign * TOWER_X, TOWER_Y, z)
        perch = (sign * LOWER_OUT_X, wy + LOWER_OUT_DY + PERCH_DY, z)
        nodes.append(
            {"name": f"coil_{label}", "mesh": names["coil"], "matrix": coil_matrix(tower, perch)}
        )
        lower = (sign * LOWER_OUT_X, wy + LOWER_OUT_DY, z)
        upper = (sign * UPPER_OUT_X, wy + UPPER_OUT_DY, z)
        nodes.append(
            {
                "name": f"upright_{label}",
                "mesh": names["upright"],
                "matrix": coil_matrix(lower, upper),
            }
        )
    return nodes


def normal_png(w=128, h=128):
    rng_row = []
    # Tileable value noise, then a height-to-normal peel.
    grid = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            grid[y][x] = hash3((x * 0.37, y * 0.37, 4.0))
    for _ in range(3):
        nxt = [[0.0 for _ in range(w)] for _ in range(h)]
        for y in range(h):
            for x in range(w):
                acc = 0.0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        acc += grid[(y + dy) % h][(x + dx) % w]
                nxt[y][x] = acc / 9.0
        grid = nxt
    raw = bytearray(w * h * 4)
    amp = 1.15
    for y in range(h):
        for x in range(w):
            hl = grid[y][(x - 1) % w]
            hr = grid[y][(x + 1) % w]
            hd = grid[(y - 1) % h][x]
            hu = grid[(y + 1) % h][x]
            nx = (hl - hr) * amp
            ny = (hd - hu) * amp
            nz = 1.0
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = nx / length, ny / length, nz / length
            i = (y * w + x) * 4
            raw[i] = int(clamp(nx * 0.5 + 0.5, 0.0, 1.0) * 255.0 + 0.5)
            raw[i + 1] = int(clamp(ny * 0.5 + 0.5, 0.0, 1.0) * 255.0 + 0.5)
            raw[i + 2] = int(clamp(nz * 0.5 + 0.5, 0.0, 1.0) * 255.0 + 0.5)
            raw[i + 3] = 255
    return write_png(w, h, raw), w, h


def write_png(w, h, rgba):
    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(rgba[y * w * 4 : (y + 1) * w * 4]) for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def pack_f32(values):
    buf = array.array("f", values)
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tobytes()


def pack_u32(values):
    buf = array.array("I", values)
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tobytes()


def accessor(view, offset, comp, count, typ, mins=None, maxs=None):
    acc = {
        "bufferView": view,
        "byteOffset": offset,
        "componentType": comp,
        "count": count,
        "type": typ,
    }
    if mins is not None:
        acc["min"] = mins
        acc["max"] = maxs
    return acc


def build_glb(meshes, png, png_w, png_h):
    blob = bytearray()
    views = []
    accs = []

    def align(n=4):
        while len(blob) % n:
            blob.append(0)

    def push_view(data, target=None):
        align(4)
        off = len(blob)
        blob.extend(data)
        view = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        views.append(view)
        return len(views) - 1

    def add_mesh(mesh):
        mesh.compute_tangents()
        n = len(mesh.pos)
        pos = []
        nrm = []
        tan = []
        uv = []
        col = []
        metal = []
        mud = []
        panel = []
        emit = []
        for i in range(n):
            pos.extend(mesh.pos[i])
            nrm.extend(mesh.nrm[i])
            tan.extend(mesh.tan[i])
            uv.extend(mesh.uv[i])
            col.extend(mesh.col[i])
            metal.append(mesh.metal[i])
            mud.append(mesh.mud[i])
            panel.append(mesh.panel[i])
            emit.append(mesh.emit[i])
        chunks = [
            (pack_f32(pos), "VEC3", 5126, False),
            (pack_f32(nrm), "VEC3", 5126, False),
            (pack_f32(tan), "VEC4", 5126, False),
            (pack_f32(uv), "VEC2", 5126, False),
            (pack_f32(col), "VEC4", 5126, False),
            (pack_f32(metal), "SCALAR", 5126, False),
            (pack_f32(mud), "SCALAR", 5126, False),
            (pack_f32(panel), "SCALAR", 5126, False),
            (pack_f32(emit), "SCALAR", 5126, False),
            (pack_u32(mesh.idx), "SCALAR", 5125, True),
        ]
        ids = []
        for data, typ, comp, index in chunks:
            vid = push_view(data, 34963 if index else 34962)
            mins = maxs = None
            if typ == "VEC3" and comp == 5126 and ids == []:
                xs = pos[0::3]
                ys = pos[1::3]
                zs = pos[2::3]
                mins = [min(xs), min(ys), min(zs)]
                maxs = [max(xs), max(ys), max(zs)]
            ids.append(accessor(vid, 0, comp, n if not index else len(mesh.idx), typ, mins, maxs))
            accs.append(ids[-1])
            ids[-1] = len(accs) - 1
        attrs = {
            "POSITION": ids[0],
            "NORMAL": ids[1],
            "TANGENT": ids[2],
            "TEXCOORD_0": ids[3],
            "COLOR_0": ids[4],
            "_METAL": ids[5],
            "_MUD": ids[6],
            "_PANEL": ids[7],
            "_EMIT": ids[8],
        }
        prim = {
            "attributes": attrs,
            "indices": ids[9],
            "material": 0,
            "mode": 4,
        }
        return {"name": mesh.name, "primitives": [prim], "extras": mesh.extras}

    gltf_meshes = [add_mesh(mesh) for mesh in meshes]
    png_view = push_view(png)
    image = {"bufferView": png_view, "mimeType": "image/png"}
    # Pad the BIN chunk.
    while len(blob) % 4:
        blob.append(0)

    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "silt build_buggy_gltf.py",
            "extras": {
                "silt": {
                    "color0": "linear base color rgb, roughness in alpha",
                    "mud": "mask, 1 cakes the surface",
                    "mounts": {
                        "lower_in_x": LOWER_IN_X,
                        "lower_in_y": LOWER_IN_Y,
                        "lower_half_z": LOWER_HALF_Z,
                        "lower_out_x": LOWER_OUT_X,
                        "lower_out_dy": LOWER_OUT_DY,
                        "upper_in_x": UPPER_IN_X,
                        "upper_in_y": UPPER_IN_Y,
                        "upper_half_z": UPPER_HALF_Z,
                        "upper_out_x": UPPER_OUT_X,
                        "upper_out_dy": UPPER_OUT_DY,
                        "tower_x": TOWER_X,
                        "tower_y": TOWER_Y,
                        "perch_dy": PERCH_DY,
                        "wheel_x": WHEEL_X,
                        "front_z": FRONT_Z,
                        "rear_z": REAR_Z,
                        "rest_wheel_y": REST_WHEEL_Y,
                    },
                }
            },
        },
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accs,
        "images": [image],
        "samplers": [
            {
                "magFilter": 9729,
                "minFilter": 9729,
                "wrapS": 10497,
                "wrapT": 10497,
            }
        ],
        "textures": [{"sampler": 0, "source": 0}],
        "materials": [
            {
                "name": "buggy-pbr",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 1.0,
                    "roughnessFactor": 1.0,
                },
                "normalTexture": {"index": 0, "scale": 1.0},
                "extras": {
                    "baseColor": "COLOR_0.rgb",
                    "roughness": "COLOR_0.a",
                    "metal": "_METAL",
                    "mudMask": "_MUD",
                    "normal": "normalTexture",
                },
            }
        ],
        "meshes": gltf_meshes,
        "nodes": rest_nodes(meshes),
        "scenes": [{"name": "buggy", "nodes": list(range(len(rest_nodes(meshes))))}],
        "scene": 0,
    }
    # png size is recorded only so a bad image fails here rather than in the shader.
    _ = (png_w, png_h)
    import json

    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(js) % 4:
        js += b" "
    total = 12 + 8 + len(js) + 8 + len(blob)
    out = struct.pack("<4sII", b"glTF", 2, total)
    out += struct.pack("<I4s", len(js), b"JSON")
    out += js
    out += struct.pack("<I4s", len(blob), b"BIN\0")
    out += blob
    return out


def check(meshes):
    chassis = meshes[0]
    wheel = next(m for m in meshes if m.name == "wheel")
    arm = next(m for m in meshes if m.name == "arm")
    assert chassis.name == "chassis"
    radius = 0.0
    for p in wheel.pos:
        radius = max(radius, math.hypot(p[1], p[2]))
    if abs(radius - WHEEL_RADIUS) > 0.012:
        raise SystemExit(f"wheel radius {radius:.4f} != {WHEEL_RADIUS}")
    low = [v for i, v in enumerate(chassis.mud) if chassis.pos[i][1] < 0.0]
    high = [v for i, v in enumerate(chassis.mud) if chassis.pos[i][1] > 0.72]
    if not low or not high:
        raise SystemExit("chassis mud bands missing")
    if sum(low) / len(low) < sum(high) / len(high) + 0.25:
        raise SystemExit(
            f"mud mask did not settle on the lower chassis ({sum(low)/len(low):.2f} vs {sum(high)/len(high):.2f})"
        )
    # Winding: face normal should agree with the vertex normal.
    bad = 0
    total = 0
    for mesh in meshes:
        for k in range(0, len(mesh.idx), 3):
            i0, i1, i2 = mesh.idx[k : k + 3]
            face = vcross(vsub(mesh.pos[i1], mesh.pos[i0]), vsub(mesh.pos[i2], mesh.pos[i0]))
            if vlen(face) < 1.0e-10:
                continue
            total += 1
            if vdot(vnorm(face), mesh.nrm[i0]) < -0.05:
                bad += 1
    if total and bad / total > 0.02:
        raise SystemExit(f"winding flipped on {bad}/{total} triangles")
    towers = []
    for sign, z in ((-1.0, FRONT_Z), (1.0, FRONT_Z), (-1.0, REAR_Z), (1.0, REAR_Z)):
        target = (sign * TOWER_X, TOWER_Y, z)
        best = min(vlen(vsub(p, target)) for p in chassis.pos)
        towers.append(best)
        if best > 1.0e-3:
            raise SystemExit(f"shock tower {target} is {best:.4f} off the cage")
    arm_reach = max(p[0] for p in arm.pos)
    if arm_reach < lower_len():
        raise SystemExit("lower arm does not reach the knuckle")
    print(
        f"chassis verts {len(chassis.pos)} tris {len(chassis.idx)//3}  "
        f"wheel r {radius:.3f}  arm reach {arm_reach:.3f} (len {lower_len():.3f})  "
        f"upper {upper_len():.3f}  mud {sum(low)/len(low):.2f}/{sum(high)/len(high):.2f}  "
        f"winding bad {bad}/{total}"
    )


def main():
    meshes = [
        build_chassis(),
        build_wheel(),
        build_arm("arm", lower_len(), LOWER_HALF_Z, 0.022),
        build_arm("arm_upper", upper_len(), UPPER_HALF_Z, 0.016),
        build_coil(),
        build_upright(),
    ]
    check(meshes)
    png, w, h = normal_png()
    data = build_glb(meshes, png, w, h)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(data)
    print(f"wrote {OUT} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
