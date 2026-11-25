# Fault (shear band) extraction with Fatbox (Gayrin et al., 2026, Solid Earth 17, 555; https://github.com/PaulineGayrin/Fatbox),
# following its numerical-model tutorial (tuto_num1_extraction_v2.ipynb, extract_faults): relative threshold of the plastic strain
# rate, Guo-Hall skeleton (OpenCV ximgproc), connected components, pixel graph (edges between pixels closer than minimum_distance),
# remove self edges and triangles, split triple junctions by orientation (dos = 8, 'minimum': the two most aligned branches stay
# connected), remove short components, cut the nodes where the local orientation changes by more than max_bend (60°), remove short
# components again.
# Input: every <name>_meta.txt / <name>_ldot.bin in the folder (fatbox_export.jl: Tet<θ>; ts_export.jl: timeseries/Tet<θ>_s<step>).
# Output: <name>_nodes.csv (node, component, i, j: 1-based grid indices) and <name>_edges.csv, and (default folder only) a check
# figure <name>_fatbox.png.
# Usage: <Fatbox>/.venv/bin/python fatbox_extract.py [frac = 0.1] [min_len_mm = 1.0] [folder = this folder]
import sys, os
import numpy as np
import networkx as nx
import cv2
from scipy.spatial import distance_matrix
FATBOX = os.path.expanduser("~/REPO/Fatbox/modules")
sys.path.insert(0, FATBOX)
import metrics, edits                                   # Fatbox
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
min_len_mm = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
folder = os.path.join(here, sys.argv[3]) if len(sys.argv) > 3 else here   # relative to this folder (Fatbox modules change the cwd)
checks = folder == here
names = sorted(f[:-len('_meta.txt')] for f in os.listdir(folder) if f.endswith('_meta.txt'))
minimum_distance, dos, max_bend = 1.5, 8, 60

def orientation(G, dof):                                # local line angle [deg, 0-180) over dof graph steps
    for node in G:
        nb = sorted(nx.single_source_shortest_path_length(G, node, cutoff=dof).items())
        (x1, y1), (x2, y2) = G.nodes[nb[0][0]]['pos'], G.nodes[nb[-1][0]]['pos']
        G.nodes[node]['ang'] = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
    return G

for name in names:
    nx_, ny_, x1, y1, dx = open(os.path.join(folder, f"{name}_meta.txt")).read().split()
    nx_, ny_, dx = int(nx_), int(ny_), float(dx)
    F = np.fromfile(os.path.join(folder, f"{name}_ldot.bin"), dtype="<f8").reshape((ny_, nx_))   # F[j, i]
    if np.nanmax(F) <= 0:                               # no plastic flow yet: no segment
        open(os.path.join(folder, f"{name}_nodes.csv"), "w").write("node,component,i,j\n")
        open(os.path.join(folder, f"{name}_edges.csv"), "w").write("a,b\n")
        print(f"{name}: no plastic flow"); continue
    thr = ((F > 0) & (F >= frac * np.nanmax(F))).astype(np.uint8)
    skel = cv2.ximgproc.thinning(thr * 255, thinningType=cv2.ximgproc.THINNING_GUOHALL) // 255
    ret, markers = cv2.connectedComponents(skel.astype(np.uint8))
    G = nx.Graph(); node = 0
    for comp in range(1, ret):
        for p in np.transpose(np.vstack(np.where(markers == comp))):
            G.add_node(node, pos=(int(p[1]), int(p[0])), x=int(p[1]), y=int(p[0]), z=0, component=comp); node += 1
    # edges between pixels closer than minimum_distance (1.5 px = the 8 neighbours), as the tutorial's distance_matrix per component,
    # but by neighbour lookup (linear memory: the n² distance matrix runs out of memory when a component covers the whole gouge)
    where = {G.nodes[n]['pos']: n for n in G}
    for n in list(G):
        x, y = G.nodes[n]['pos']
        for dx_ in (-1, 0, 1):
            for dy_ in (-1, 0, 1):
                m = where.get((x + dx_, y + dy_))
                if m is not None and m != n and G.nodes[m]['component'] == G.nodes[n]['component'] and np.hypot(dx_, dy_) < minimum_distance:
                    G.add_edge(n, m)
    G = metrics.compute_edge_length(G)
    G = edits.remove_self_edge(G)
    G = edits.label_components(G)
    G = edits.remove_triangles(G)
    G = edits.split_triple_junctions(G, dos, split='minimum', threshold=20, plot=False)
    rsc = round(min_len_mm / dx)
    G = edits.remove_small_components(G, minimum_size=rsc)
    G = edits.label_components(G)
    G = orientation(G, 5)
    rem = []
    for n in G:                                         # bends: largest orientation change within 5 graph steps
        nb = nx.single_source_shortest_path_length(G, n, cutoff=5)
        a = np.array([G.nodes[m]['ang'] for m in nb])
        d = np.abs((a[:, None] - a[None, :] + 90) % 180 - 90)
        if d.max() > max_bend: rem.append(n)
    G.remove_nodes_from(rem)
    G = edits.remove_small_components(G, minimum_size=rsc)
    G = edits.label_components(G)
    with open(os.path.join(folder, f"{name}_nodes.csv"), "w") as f:
        f.write("node,component,i,j\n")
        for n in G: f.write(f"{n},{G.nodes[n]['component']},{G.nodes[n]['x'] + 1},{G.nodes[n]['y'] + 1}\n")
    with open(os.path.join(folder, f"{name}_edges.csv"), "w") as f:
        f.write("a,b\n")
        for a, b in G.edges: f.write(f"{a},{b}\n")
    ncomp = len(set(nx.get_node_attributes(G, 'component').values()))
    print(f"{name}: threshold {frac:g} x max, {thr.sum()} band pixels, skeleton {skel.sum()} px, {G.number_of_nodes()} nodes, {ncomp} segments")
    if not checks: continue
    # check figure
    J, I = np.where(F > 0); pad = 10
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.imshow(np.log10(np.maximum(F, 1e-7)), cmap='gray_r', vmin=-6, vmax=-3, origin='lower')
    cols = plt.cm.tab20(np.linspace(0, 1, 20))
    for n in G:
        ax.plot(G.nodes[n]['x'], G.nodes[n]['y'], '.', ms=1.5, color=cols[G.nodes[n]['component'] % 20])
    ax.set_xlim(I.min() - pad, I.max() + pad); ax.set_ylim(J.min() - pad, J.max() + pad); ax.set_aspect('equal')
    fig.savefig(os.path.join(folder, f"{name}_fatbox.png"), dpi=200); plt.close(fig)
