# IIM step 1 — interface geometry

This is part of the plan in [IIM_plan.md](IIM_plan.md). All the code is in
[src/StokesIIM/Geometry.jl](../src/StokesIIM/Geometry.jl), which is exported by the
`StagFDTools.StokesIIM` submodule.

## Convention

Γ is the zero level of φ.
- φ < 0: inside the inclusion.
- φ > 0: the matrix.

[`side(φ)`](../src/StokesIIM/Geometry.jl#L14) returns −1 or +1. There is no 0 case, so every node
belongs to exactly one phase.

## Implementation steps

1. **Sample φ on the staggered grids.** [`LevelSet(φfun, X)`](../src/StokesIIM/Geometry.jl#L22)
   evaluates φ on the extended grids of `GenerateGrid`, which include ghost nodes. It returns
   `(x, y, c, v)` for the Vx, Vy, centroid and vertex arrays. Nodes with |φ| < 1e−10·Δ lie on Γ
   and are moved to the + side. Without this, a node that lies exactly on Γ (e.g. vertex (0, 2)
   when r_c = 2 and Δ = 0.5) creates a zero-width sliver.
2. **Merge the grids into one fine lattice.**
   [`FineLattice(φ, X, Δ)`](../src/StokesIIM/Geometry.jl#L38) combines the four arrays into one
   lattice `Φ` with spacing Δ/2:

   | (I, J) parity | Node |
   |---|---|
   | odd / odd | vertex |
   | even / even | centroid |
   | odd / even | Vx |
   | even / odd | Vy |

   Every interface point used by the IIM stencils lies on an edge of this lattice.
3. **Find the crossings.** [`InterfaceCrossings(fl)`](../src/StokesIIM/Geometry.jl#L109) finds
   every lattice edge where `side` changes. On each one, it locates the root of the cubic
   through 4 samples along the grid line by bisection:
   [`crossing_fraction`](../src/StokesIIM/Geometry.jl#L79) and
   [`lagrange`](../src/StokesIIM/Geometry.jl#L61). Near the edge of the lattice, where only 3
   samples are available, it uses a quadratic.
   - Output: `tx` and `ty`, the fraction along each horizontal or vertical edge (`NaN` if the
     edge isn't crossed), and a list `points` with the position x_Γ, unit normal n and edge
     index of each crossing.
4. **Compute the normals.** [`∇Φ`](../src/StokesIIM/Geometry.jl#L93) takes central differences
   of Φ at the two ends of the edge, interpolates them linearly to x_Γ, then normalises. n
   points towards φ > 0.
5. **Flag the irregular equations.** [`IrregularNodes(φ, nc)`](../src/StokesIIM/Geometry.jl#L145)
   flags a Vx, Vy or Pt equation when any member of its standard stencil lies on the other side
   of Γ. The index ranges match `Stokes.jl`:

   | Equation | Stencil members checked |
   |---|---|
   | Vx[i,j] | Vx[i±1,j], Vx[i,j±1], Pt[i−1:i, j−1], Vy[i:i+1, j−1:j] |
   | Vy[i,j] | Vy[i±1,j], Vy[i,j±1], Pt[i−1, j−1:j], Vx[i−1:i, j:j+1] |
   | Pt[i,j] | Vx[i:i+1, j+1], Vy[i+1, j:j+1] |

   These are the equations that will receive the IIM corrections in step 4.
6. **Check for thin features.** [`MultipleCrossings(Γ)`](../src/StokesIIM/Geometry.jl#L177)
   counts lattice segments of length Δ that Γ crosses twice. These are features thinner than
   about Δ, which the first-order IIM doesn't handle. It returns 0 for the inclusion test.

Tests are in [test/test_iim_geometry.jl](../test/test_iim_geometry.jl):
- the fine lattice reproduces φ at every node
- crossings and normals are exact for a straight inclined interface
- the irregular columns for a vertical interface match a hand check
- nodes on Γ are moved to the + side

## Figures

### Discretisation sketch: `figures/iim_mesh.png`

Produced by [figures/iim_mesh.jl](figures/iim_mesh.jl).
- **(a)** The MAC unknowns (z = ηv on the faces, p at the centroids), coloured by phase. Black
  outlines mark irregular nodes. Gold diamonds are the crossings of Γ with the half-grid lines,
  where the jumps are evaluated. These points are not unknowns.
- **(b)** One irregular x-momentum stencil. It shows the interface points the stencil uses (star
  for ∂xx/∂yy, hexagon for a cut corner), the normal n, the two fit centres x_Γ ± L n, and the
  one-sided 3-node fits (solid: zx, dashed: zy) that give v_Γ and ∂ₜv_Γ.

### Step 1 result: `../examples/_Stokes_IIM/IIM_Step1_Geometry.png`

Produced by [IIM_Step1_Geometry.jl](../examples/_Stokes_IIM/IIM_Step1_Geometry.jl) for the
Schmid & Podladchikov (2003) inclusion: L = 10, r_c = 2, φ = r − r_c.
- **Left (nc = 20):** nodes coloured by phase (circle: Pt, ▶: Vx, ▲: Vy). Outlined nodes are the
  irregular equations: 48 Vx, 48 Vy and 24 Pt. Gold diamonds are the 60 crossings of Γ with the
  lattice. The irregular nodes form a band one stencil wide around Γ.
- **Middle:** a zoom on the first quadrant showing the computed normals at each crossing. They
  are perpendicular to the circle and point outward.
- **Right:** the maximum error against h (nc = 10 to 320).
  - Crossing positions, measured as | |x_Γ| − r_c |: O(h⁴).
  - Normals, measured as |n − x_Γ/|x_Γ||: O(h²).
  - Both orders hold for the exact distance φ = r − r_c and for φ = tanh(r − r_c). The second
    has the same zero level but isn't a distance function, so the geometry doesn't require an
    exact distance function (useful later for marker-based level sets).

The PNG files are gitignored. Run the two scripts to regenerate them.
