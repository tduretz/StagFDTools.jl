# Immersed Interface Method (IIM) for discontinuous-viscosity Stokes — implementation plan

Branch: `iim-discontinuous-viscosity`

Reference implementation: W. Landry, `discontinuous_stokes` (C++, 2011–2013),
<https://gitlab.com/wlandry/discontinuous_stokes>.
- HEAD `19e9e30`: first-order version.
- `fdae05a^`: last second-order version.

![mesh](figures/iim_mesh.png)

The figure is drawn by `figures/iim_mesh.jl`.
- **Panel (a):** MAC unknowns coloured by phase. Black outlines mark irregular nodes, whose
  stencil crosses Γ. Gold diamonds mark where Γ crosses half-grid lines; this is where jumps
  are evaluated.
- **Panel (b):** one irregular x-momentum stencil, its interface points, and the one-sided
  3-node fits that give v_Γ and ∂ₜv_Γ.

---

## 1. Method summary (what we port)

### Unknowns and operators
- **Unknowns:** standard MAC layout, `z = η v` on the faces and `p` at the cell centres.
  There are no extra global DOFs.
- **Viscosity:** piecewise constant, with value η₋ inside Γ and η₊ outside.
- **Regular equations:** inside one phase, the equations use η-free operators:
  - momentum: `2∂xx zx + ∂yy zx + ∂xy zy − ∂x p = fx`
  - continuity: `∂x zx + ∂y zy = 0`
- **Irregular equations:** when a stencil crosses Γ, the standard stencil gets a correction
  `C`, built from Taylor-expanded jumps. Writing `[·] = this side − other side`:
  - `∂xx`-type terms: `±([z] + [∇z]·δ)/h²`
  - pressure gradient: `±[p]/h`
  - mixed derivatives: corner or side terms
  - continuity: `Cp`
- **Level set φ:** sampled at the vx, vy, centre and vertex positions. Spacing along each grid
  line is then h/2, so a crossing is the root of a 3-point quadratic through φ.

### Jump conditions (n/t frame, Δη = η_this − η_other)

| Condition | Jump |
|---|---|
| [v] = 0 | [z] = Δη v_Γ |
| [∂ₜv] = 0 | [∂ₜz] = Δη ∂ₜv_Γ |
| incompressibility on both sides | [∂ₙzₙ] = −Δη ∂ₜvₜ |
| shear traction continuous | [∂ₙzₜ] = −Δη ∂ₜvₙ |
| normal traction continuous | [p] = −2Δη ∂ₜvₜ |

Every jump depends only on four interface quantities: v_Γ (2 components) and ∂ₜv_Γ
(2 components). This is why z = ηv is the right unknown. If v were the unknown, [∂ₙvₜ] would
need a one-sided traction, which would mean an extra unknown.

### Closing v_Γ and ∂ₜv_Γ (eliminated, not solved for)
- **Fit centres:** one per side, at x_Γ ± L n with L = h / max|nᵢ|.
- **One-sided values:** on each side, fit a linear function to the 3 nearest same-side nodes,
  which must not all lie on one grid line. Extrapolate the fit to x_Γ to get v± (v = z/η).
- **Weighted average:** v_Γ = (η₋v₋ + η₊v₊)/(η₋ + η₊). Landry's history says this
  stiff-side weighting is what makes contrasts up to 1e10 work.
- **Tangential derivative:** ∂ₜv_Γ uses the nearest one-sided finite difference on each side,
  with the same η-weighting.
- **Consequence:** C is linear in z. Interface rows therefore reach up to ±5 nodes away, and
  the matrix is **non-symmetric**.

### First vs second order
- **First order (Landry HEAD):** jumps up to [∇z] and [p].
- **Second order:** additionally needs ∂ₜₜv_Γ, obtained from a 6-point quadratic fit. This
  gives [∇∇z] and [∇p], and adds a ½[∂²z]δ² term. Landry left a TODO saying curvature and
  density-jump terms were missing even there.

---

## 2. Design decisions for StagFDTools

1. **New submodule `StagFDTools.StokesIIM`**, so the production `Stokes` module is not touched.
   It reuses the following from `Stokes`:
   - `Fields`, `Ranges`, `Numbering!`, `set_boundaries_template!`
   - `PatternBlock`, `pattern_storage`, `set_sparsity_pattern!`, `fixed_pattern`,
     `zero_values!`, `assembly_loop`, `_target` (all from `StokesAssembly.jl`)
2. **Unknown z = ηv inside the IIM system.**
   - Boundary values are converted with z = η_bc v_bc. Phase 1 requires Γ to stay at least
     6h from the domain boundary, so every boundary node belongs to one phase.
   - After the solve, recover v = z/η node by node, so downstream code (plots, markers) sees v.
3. **Linear-viscous, piecewise-constant η only in phases 1–4.** VE/VEP and power law are out of
   scope until phase 6.
4. **Assembly = regular part + correction part, both written through the existing `K[a][b][row, col]` interface.**
   - **Regular part:** a compact, AD-local kernel in the same style as
     `SMomentum_x_Generic`, but in z-form. Do not reuse `SMomentum_x_Generic`: its
     deviatoric strain rate adds `−⅔∂x(∇·z)`, and ∇·z ≠ 0 discretely next to the interface.
   - **Correction part:** for each irregular row, gather an index list (the cloud), build an
     `SVector` of cloud values, and write the correction as a plain Julia function
     `C(cloud_vals)`. Its weights are `ad_gradient(C, cloud_vals)`. This replaces Landry's
     unit-vector probing and keeps the AD style.
   - `PatternBlock` records arbitrary (row, col) pairs, so the wide rows enter the fixed
     pattern with no change to `StokesAssembly.jl`.
5. **Solvers.**
   - **Verification:** direct `lu` on the full block system, with pressure fixed by a null-mean
     row or one pinned cell.
   - **Production:** GCR on the IIM operator, preconditioned with the existing symmetric
     averaged-viscosity operator. `Solvers.jl` already factorises `Hermitian(Kuusc)` with
     Cholesky, which is valid only as a preconditioner here. This is the existing `M` /
     `M_PC` split.
6. **Geometry from an analytic φ first.** Marker-based φ (JustPIC) comes in phase 6.

---

## 3. Work plan

### Phase 0 — baseline (≈1 day)
- [ ] Revive `examples/_Stokes_VEP_SSG/untested/ExampleADStokes_ViscousInclusionSchmid2003.jl`
      as `examples/_Stokes_IIM/Baseline_Schmid2003.jl`, using the current averaging approach.
- [ ] Convergence table (L1, L∞ of vx, vy, p) for nc = 32, 64, 128, 256 and
      η_inc/η_mat = 1e−3, 1e3, 1e6. Save it to `docs/IIM_baseline.md`.
      Every later phase is compared against this table.

### Phase 1 — geometry (`src/StokesIIM/Geometry.jl`) — done
Example: `examples/_Stokes_IIM/IIM_Step1_Geometry.jl`.

- [x] `LevelSet(φfun, X)`: a NamedTuple `(x, y, c, v)` on the extended grids of `GenerateGrid`.
- [x] Nodes with |φ| < 1e−10·Δ are moved to the + side. Without this, r_c = 2, h = 0.5 put
      vertex (0, 2) on Γ, and round-off created a zero-width sliver.
- [x] `FineLattice`: the four grids merged into one lattice of spacing Δ/2. Every IIM
      interface point lies on one of its edges.
- [x] `InterfaceCrossings`: crossings on every fine edge (`tx`, `ty`, `points`). Root of the
      cubic through 4 samples, located by bisection (quadratic at the lattice ends).
      Measured order O(h⁴); the plan expected a quadratic with O(h³).
- [x] Normals: central-difference ∇φ on the fine lattice, linearly interpolated along the
      edge. Measured order O(h²).
- [x] `IrregularNodes`: flags Vx, Vy and Pt equations whose standard stencil crosses Γ.
- [x] `MultipleCrossings`: counts length-Δ segments crossed twice. Error handling and
      fallback are still to do (phase 5).
- [ ] Per-equation lists of interface points `(kind, x_Γ, n, Δη, sgn)`. These are deferred to
      phase 4, where they become lookups on the fine edges around each node:
  - `:dd`: (I±1|I±2, J), (I, J±1|J±2)
  - `:dp`: (I±1, J)
  - `:corner` / `:side`: box edges at I±1 and J±1
  - `:div`: the 4 edges around a centroid
- [x] Tests in `test/test_iim_geometry.jl` (also included in `runtests.jl`):
  - fine-lattice mapping
  - exact crossings and normals for a straight interface
  - hand-checked irregular columns
  - nodes on Γ moved to the + side

### Phase 2 — interface trace operator (`src/StokesIIM/Trace.jl`)
- [ ] `trace_weights(x_Γ, n, comp, φ, η, num)`. It returns sparse weights (indices + values)
      for `v_Γ[comp]` and `∂ₜv_Γ[comp]` as linear functions of z:
  - cloud selection as in `compute_dv_dtt.cxx`, using only `:in` nodes
  - 3×3 Vandermonde solve with `StaticArrays`
  - η-weighted average
- [ ] Dirichlet nodes inside a cloud add an affine term. Return it separately as `(w, b)`.
- [ ] Tests:
  - exact for fields that are piecewise linear and continuous across a straight Γ
  - weights sum correctly (Σw = 1 for values, Σw = 0 for derivatives)
  - no cloud crosses Γ

### Phase 3 — jumps (`src/StokesIIM/Jumps.jl`)
- [ ] `jumps(Δη, v_Γ, ∂ₜv_Γ, n) -> ([z], [∇z], [p])`: a direct port of `compute_jumps.cxx`,
      in SVector/SMatrix form.
- [ ] Test: insert the exact Schmid & Podladchikov (2003) solution
      (`ExactFieldSolutions.Stokes2D_Schmid2003`) at points on the circle and compare with
      the jumps of that solution measured by finite differences on each side.

### Phase 4 — corrected operator and solve (`src/StokesIIM/Assembly.jl`)
- [ ] z-form local residuals `IIM_Momentum_x`, `IIM_Momentum_y` and `IIM_Continuity`, each
      the regular stencil plus `C`. Corrections, with x-momentum shown (Landry's file
      `compute_Cxyz.cxx`):
  - `:dd`: `+f·([z]·e_d + [∇z]·δ)/h²`, with f = 2 for d = dd and 1 otherwise
  - `:side`: `−sgn·[∂z]/h`
  - `:dp`: `+sgn·[p]/h`
  - `:corner`: `±([z] + [∇z]·δ)/h²`
- [ ] `Residual*_IIM!` and `Assemble*_IIM!`, with the same signatures as in `Stokes.jl`. The
      extra columns are written through `K[a][b]`.
- [ ] Driver `examples/_Stokes_IIM/IIM_Schmid2003.jl`: pattern → assemble → `lu` → recover v.
- [ ] Ensure the residual is zero for z = 0, p = 0 (no affine leak), and check the Jacobian
      against finite differences on a 16×16 grid (`test/test_iim_assembly.jl`).

### Phase 5 — verification and solver (`examples/_Stokes_IIM/IIM_convergence.jl`)
- [ ] **Schmid 2003 circle:** η contrast 1e−6…1e6 in both directions, nc 32–512, and the same
      table as phase 0.
  - Expected: no pressure spikes at Γ.
  - Expected: L∞(p) converges, which the averaging approach does not do.
  - Expected: velocity converges at O(h) or better.
- [ ] **SolCx** (straight vertical interface): rotated by θ = 0, 0.5 rad, and so on, following
      Landry's `initial.cxx`. This tests inclined interfaces.
- [ ] **Robustness:** sweep the circle centre over sub-cell offsets. Landry's `middle`
      constant was used to probe the cases where Γ lands exactly on nodes.
- [ ] **Iterative solver:** GCR with the averaged-viscosity preconditioner `M_PC`. Record the
      iteration count against η contrast and nc.

### Phase 6 — extensions (after phase 5 passes)
- [ ] **Second order:** 6-point quadratic fits giving ∂ₜₜv_Γ, plus [∇∇z] and [∇p], including
      curvature terms (Landry never derived these) and density-jump terms in [∂ₙp].
      References: Li & Ito (2006) *The Immersed Interface Method*, SIAM; Li, Ito & Lai (2007)
      augmented IIM for Stokes with discontinuous viscosity; Tan, Le, Lim & Khoo (2009).
- [ ] **Moving interface:** φ from JustPIC markers (signed-distance reconstruction), with the
      pattern rebuilt every step, since clouds change.
- [ ] **Visco-elasticity:** use η_eff = (1/η + 1/(GΔt))⁻¹ per phase, and add affine jump terms
      from [τ⁰] to the traction conditions.
- [ ] **Stress BCs** (`:normal_stress`, `:tangential_stress`) when Γ is near the boundary.

---

## 4. Files

```
src/StokesIIM/StokesIIM.jl     module, exports; included from src/StagFDTools.jl
src/StokesIIM/Geometry.jl      LevelSet, crossings, normals, IrregularInfo
src/StokesIIM/Trace.jl         trace_weights (v_Γ, ∂ₜv_Γ)
src/StokesIIM/Jumps.jl         jumps(...)
src/StokesIIM/Assembly.jl      z-form kernels + corrections, Residual/Assemble
test/test_iim_geometry.jl, test/test_iim_trace.jl, test/test_iim_assembly.jl
examples/_Stokes_IIM/Baseline_Schmid2003.jl, IIM_Schmid2003.jl, IIM_SolCx.jl, IIM_convergence.jl
docs/IIM_plan.md (this file), docs/IIM_baseline.md, docs/figures/
```

## 5. Risks and open questions
- **Non-symmetric operator.** The current Cholesky-based solvers become preconditioners only.
  Check that GCR stays robust at contrasts of 1e6 and above.
- **Biased weighting.** The η-weighted v_Γ favours the stiff side by design. Check that the
  soft-inclusion case (η_inc ≪ η_mat) converges as well as the stiff-inclusion case.
- **Thin features.** Layers under ~2h produce anticorners and double crossings, which Landry
  never implemented. A fallback strategy is needed before using IIM in layered or shear-band
  models.
- **Cost.** The pattern gains up to ~11×11-wide rows, but only along Γ, so the extra nnz scale
  with the interface length (≈ O(nc)).
- **Coupling with the existing VEP machinery** (𝐷 tangent, markers) is deliberately deferred.
  Decide after phase 5 whether IIM stays a separate linear-viscous solver or becomes a
  correction operator added on top of `Stokes`.
