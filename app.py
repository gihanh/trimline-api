import io, os, json, traceback
import numpy as np
import trimesh
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.ops import unary_union
from flask import Flask, request, jsonify

app = Flask(__name__)

def mesh_to_numpy(mesh):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    return V, F

def adjacency_from_faces(nv, F):
    nbr = [set() for _ in range(nv)]
    for a,b,c in F:
        nbr[a].add(b); nbr[a].add(c)
        nbr[b].add(a); nbr[b].add(c)
        nbr[c].add(a); nbr[c].add(b)
    return [list(s) for s in nbr]

def umbrella_curvature(V, nbr):
    C = np.zeros(len(V), dtype=np.float64)
    for i in range(len(V)):
        if not nbr[i]: continue
        m = np.mean(V[nbr[i]], axis=0)
        C[i] = np.linalg.norm(V[i] - m)
    return C

def best_fit_plane(points, sample_cap=20000):
    P = points
    if P.shape[0] > sample_cap:
        P = P[np.random.choice(P.shape[0], sample_cap, replace=False)]
    c = P.mean(axis=0); X = P - c; C = X.T @ X
    w, Vp = np.linalg.eigh(C); order = np.argsort(w)
    n = Vp[:, order[0]]; x = Vp[:, order[2]]; y = Vp[:, order[1]]
    if np.dot(np.cross(x, y), n) < 0: y = -y
    return c, n, x, y

def to_plane_uv(V, o, x, y):
    R = V - o
    return np.stack([R @ x, R @ y], axis=1)

def from_plane_uv(uv, o, x, y):
    return o + uv[:,0,None]*x + uv[:,1,None]*y

def typical_spacing(P):
    if len(P) <= 1: return 1.0
    Q = P if len(P) <= 800 else P[np.random.choice(len(P), 800, replace=False)]
    dmins = []
    for i in range(len(Q)):
        d = np.linalg.norm(Q[i] - Q, axis=1)
        d[i] = 1e9
        dmins.append(np.min(d))
    return float(np.median(dmins))

def morph_concave_outline(points2d):
    P = np.unique(np.round(points2d, 4), axis=0)
    if P.shape[0] < 4: return P
    from shapely.geometry import Point
    from shapely.ops import unary_union
    base = max(typical_spacing(P), 0.05)
    for f in [1.0, 1.5, 2, 3, 4, 6, 8, 10]:
        r = base * f
        discs = [Point(p[0], p[1]).buffer(r, resolution=8) for p in P]
        union = unary_union(discs).buffer(0)
        if union.is_empty: continue
        if union.geom_type == "Polygon":
            poly = union
        else:
            polys = [g for g in union.geoms if g.geom_type == "Polygon"]
            if not polys: continue
            poly = max(polys, key=lambda g: g.area)
        coords = np.asarray(poly.exterior.coords)[:, :2]
        if len(coords) >= 12 and poly.length >= 10.0:
            return coords
    hull = MultiPoint([tuple(p) for p in P]).convex_hull
    if hull.is_empty: return P
    if hull.geom_type == "Polygon":
        return np.asarray(hull.exterior.coords)[:, :2]
    try:
        return np.asarray(hull.coords)[:, :2]
    except:
        return P

def resample_polyline_2d(poly_uv, step):
    if len(poly_uv) < 2: return poly_uv
    close = np.linalg.norm(poly_uv[0]-poly_uv[-1]) < 1e-9
    poly = poly_uv if close else np.vstack([poly_uv, poly_uv[0]])
    segs = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    L = float(np.sum(segs))
    if L < 1e-9: return poly_uv
    n = max(int(round(L / max(step,1e-6))), 8)
    t = np.linspace(0.0, L, n, endpoint=False)
    out = []; acc = 0.0; i = 0
    for target in t:
        while i < len(segs) and acc + segs[i] < target:
            acc += segs[i]; i += 1
        if i >= len(segs):
            out.append(poly[-1]); continue
        d = target - acc
        p = poly[i] + (poly[i+1]-poly[i]) * (d / max(segs[i],1e-9))
        out.append(p)
    return np.asarray(out)

def smooth_polyline_2d(poly_uv, passes=1):
    if passes <= 0 or len(poly_uv) < 5: return poly_uv
    P = poly_uv.copy()
    closed = np.linalg.norm(P[0]-P[-1]) < 1e-9
    for _ in range(passes):
        Q = P.copy()
        for i in range(1, len(P)-1):
            Q[i] = 0.25*P[i-1] + 0.5*P[i] + 0.25*P[i+1]
        if closed:
            Q[0]  = 0.25*P[-2] + 0.5*P[0] + 0.25*P[1]
            Q[-1] = Q[0]
        P = Q
    return P

def compute_trimline(V, F, offset_mm=0.7, curv_percentile=80,
                     resample_step=0.20, smooth_passes=2, min_arc_len=20.0):
    nbr = adjacency_from_faces(len(V), F)
    curv = umbrella_curvature(V, nbr)
    thresh = np.percentile(curv, curv_percentile)
    band_idx = np.where(curv >= thresh)[0]
    if band_idx.size < 200: return None
    bandV = V[band_idx]
    o, n, x, y = best_fit_plane(V)
    uv = to_plane_uv(bandV, o, x, y)
    if uv.shape[0] > 6000:
        uv = uv[np.random.choice(uv.shape[0], 6000, replace=False)]
    outline = morph_concave_outline(uv)
    if outline.shape[0] < 8: return None
    ring = Polygon(outline)
    if (not ring.is_valid) or (ring.length < min_arc_len): return None
    inner = ring.buffer(-offset_mm).buffer(0)
    if inner.is_empty:
        inner = ring.simplify(0.05).buffer(-max(0.2, 0.5*offset_mm)).buffer(0)
    if inner.is_empty: return None
    uv_poly = np.asarray(inner.exterior.coords)[:, :2]
    uv_poly = smooth_polyline_2d(uv_poly, passes=smooth_passes)
    uv_poly = resample_polyline_2d(uv_poly, step=resample_step)
    xyz_poly = from_plane_uv(uv_poly, o, x, y)
    return xyz_poly

def stl_to_pts_text(stl_bytes, offset=0.7, curv_pct=80, resample=0.20, smooth=2, min_arc=20.0):
    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type='stl')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()
    V, F = mesh_to_numpy(mesh)
    xyz = compute_trimline(V, F, offset_mm=offset, curv_percentile=curv_pct,
                           resample_step=resample, smooth_passes=smooth, min_arc_len=min_arc)
    if xyz is None: return None
    return "\n".join("{:.6f} {:.6f} {:.6f}".format(p[0],p[1],p[2]) for p in xyz)

@app.get("/")
def health():
    return jsonify(ok=True)

@app.post("/trimline")
def process():
    try:
        api_key = os.environ.get("API_KEY")
        if api_key and request.headers.get("X-API-Key") != api_key:
            return jsonify(ok=False, error="Unauthorized"), 401
        if "file" not in request.files:
            return jsonify(ok=False, error="No file"), 400

        f = request.files["file"]
        name = f.filename or "input.stl"
        stl_bytes = f.read()

        def getfloat(k, d):
            try: return float(request.form.get(k, d))
            except: return d
        offset   = getfloat("offset", 0.7)
        curv_pct = getfloat("curv_pct", 80.0)
        resample = getfloat("resample", 0.20)
        smooth   = int(getfloat("smooth", 2))
        min_arc  = getfloat("min_arc", 20.0)

        pts_text = stl_to_pts_text(stl_bytes, offset, curv_pct, resample, smooth, min_arc)
        if pts_text is None:
            return jsonify(ok=False, error="Trimline could not be computed (try lower curv_pct / min_arc).")
        return jsonify(ok=True,
                       filename=name,
                       pts_filename=name.rsplit(".",1)[0] + "_trimline.pts",
                       pts_text=pts_text,
                       count=len(pts_text.splitlines()))
    except Exception as e:
        return jsonify(ok=False, error=str(e), trace=traceback.format_exc()), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
