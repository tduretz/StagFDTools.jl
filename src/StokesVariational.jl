# Variational free surface for single-phase Stokes on the staggered grid
#
# Larionov et al. (2017), Gray et al. (2026, GMD): the volume integrals of the variational form are approximated
# control volume by control volume, multiplying each term by the fluid fraction φ of its control volume:
#
#   [ 2Dᵀ M φτ D    G φP ] [v]   [φv ρg]
#   [ φP Gᵀ         0    ] [p] = [  0  ]
#
# i.e. σxx, σyy (and P) are weighted by φc (centroids), τxy by φv (vertices), gravity by φVx/φVy. The discrete
# operators are unchanged, the weighting is done in SMomentum_x_Generic/SMomentum_y_Generic (keyword φ).
# Unknowns whose equations are empty (no fluid in any control volume they are connected to) are removed: their
# type is set to :air, they are not numbered and stay zero.
#
# The fluid fractions are computed from a JustPIC MarkerChain (topography y = h(x), fluid below).
#
# Grid layout (StagFDTools, with ghosts):  φ.c ~ Pt (nc.x+2, nc.y+2),  φ.v ~ vertices (nc.x+3, nc.y+3),
#                                          φ.Vx ~ V.x (nc.x+3, nc.y+4),  φ.Vy ~ V.y (nc.x+4, nc.y+3)

"""
    VariationalWeights(nc)

Fluid fractions of the control volumes of centroids (`c`), vertices (`v`), Vx (`Vx`) and Vy (`Vy`) nodes,
stored on the ghosted StagFDTools arrays. Initialised to one (no free surface).
"""
struct VariationalWeights{T}
    c::T
    v::T
    Vx::T
    Vy::T
end

function VariationalWeights(nc)
    r = Ranges(nc)
    return VariationalWeights(ones(r.size_c...), ones(r.size_v...), ones(r.size_x...), ones(r.size_y...))
end

"""
    VariationalFreeSurface(a::Allocs, nc; tol=1e-9)

Variational free surface state: fluid fractions `φ`, the boundary-condition template `type0` (copy of `a.type`
before any :air flag), the activation tolerance `tol` (fractions below are set to zero) and the storage of
the (reduced) linear system, rebuilt by `update_free_surface!` whenever the set of active unknowns changes.
`φp` holds the fluid fraction of each pressure unknown (penalty scaling of the GCR preconditioner) and
`gcr` the GCR storage of Solvers.jl (StokesGCRCache, Cholesky factor reused while the flags are unchanged).
"""
mutable struct VariationalFreeSurface{W,T,MT}
    φ::W
    type0::T
    tol::Float64
    M::MT
    M_PC::MT
    dx::Vector{Float64}
    r::Vector{Float64}
    φp::Vector{Float64}
    gcr::StokesGCRCache
    θ_fs::Float64
end

function VariationalFreeSurface(a, nc; tol=1e-9, θ_fs=0.0)
    φ = VariationalWeights(nc)
    type0 = Fields(copy(a.type.Vx), copy(a.type.Vy), copy(a.type.Pt))
    M, dx, r = allocate_system(a.number)
    M_PC, _, _ = allocate_system(a.number)
    return VariationalFreeSurface(φ, type0, tol, M, M_PC, dx, r, ones(maximum(a.number.Pt)), StokesGCRCache(), θ_fs)
end

function allocate_system(number)
    nVx, nVy, nPt = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)
    M, _, _, _, _, dx, r = allocate_matrices(nVx, nVy, nPt)
    return M, dx, r
end

#--------------------------------------------------------------------------------------------------------------#
# Fluid fractions from the marker chain

# Copy the outermost inner layer of A (inner part A[ix, iy]) into the ghost layers
function extend_ghosts!(A, ix, iy)
    @views begin
        for i in 1:(first(ix)-1)
            A[i, iy] .= A[first(ix), iy]
        end
        for i in (last(ix)+1):size(A, 1)
            A[i, iy] .= A[last(ix), iy]
        end
        for j in 1:(first(iy)-1)
            A[:, j] .= A[:, first(iy)]
        end
        for j in (last(iy)+1):size(A, 2)
            A[:, j] .= A[:, last(iy)]
        end
    end
    return A
end

"""
    fluid_fractions!(φ::VariationalWeights, chain::MarkerChain, X, Δ, nc; tol=1e-9)

Fraction of each control volume lying below the marker chain (JustPIC `compute_rock_fraction!`, exact for
the piecewise linear chain), copied to the ghosted arrays of `φ`. Fractions below `tol` are set to zero.
"""
function fluid_fractions!(φ::VariationalWeights, chain, X, Δ, nc; tol=1e-9)
    ratios = (
        center = zeros(nc.x, nc.y),
        vertex = zeros(nc.x + 1, nc.y + 1),
        Vx     = zeros(nc.x + 1, nc.y),
        Vy     = zeros(nc.x, nc.y + 1),
    )
    xvi = (collect(X.v.x), collect(X.v.y))
    JustPIC.compute_rock_fraction!(ratios, chain, xvi, (Δ.x, Δ.y))
    for A in values(ratios)
        @. A = ifelse(A < tol, 0.0, A)
    end

    r = Ranges(nc)
    φ.c[r.inx_c, r.iny_c]   .= ratios.center
    φ.v[r.inx_v, r.iny_v]   .= ratios.vertex
    φ.Vx[r.inx_Vx, r.iny_Vx] .= ratios.Vx
    φ.Vy[r.inx_Vy, r.iny_Vy] .= ratios.Vy
    extend_ghosts!(φ.c, r.inx_c, r.iny_c)
    extend_ghosts!(φ.v, r.inx_v, r.iny_v)
    extend_ghosts!(φ.Vx, r.inx_Vx, r.iny_Vx)
    extend_ghosts!(φ.Vy, r.inx_Vy, r.iny_Vy)
    return φ
end

#--------------------------------------------------------------------------------------------------------------#
# Flagging and numbering

# Weights entering the x-momentum equation of Vx[i, j]: centroids W/E, vertices S/N
@inline φ_loc_x(φ, i, j) = (c=SVector(φ.c[i-1, j-1], φ.c[i, j-1]), v=SVector(φ.v[i, j-1], φ.v[i, j]))
# Weights entering the y-momentum equation of Vy[i, j]: centroids S/N, vertices W/E, Vy control volume (gravity)
@inline φ_loc_y(φ, i, j) = (c=SVector(φ.c[i-1, j-1], φ.c[i-1, j]), v=SVector(φ.v[i-1, j], φ.v[i, j]), Vy=φ.Vy[i, j])

"""
    flag_free_surface!(type, φ, type0, nc)

Reset `type` to the template `type0`, then flag as `:air` every interior unknown (`:in`) whose equation is
empty: pressures with φc = 0, velocities with zero fraction in their own control volume and in all the
stress control volumes of their momentum equation (Gray et al., 2026, Figs. 3-4).
Any stress with a nonzero weight then only involves active velocities.
"""
function flag_free_surface!(type, φ, type0, nc)
    type.Vx .= type0.Vx
    type.Vy .= type0.Vy
    type.Pt .= type0.Pt
    for j in 2:(nc.y+1), i in 2:(nc.x+1)
        type.Pt[i, j] === :in && φ.c[i, j] == 0 && (type.Pt[i, j] = :air)
    end
    for j in 3:(nc.y+2), i in 2:(nc.x+2)
        type.Vx[i, j] === :in || continue
        w = φ_loc_x(φ, i, j)
        (φ.Vx[i, j] > 0 || any(>(0), w.c) || any(>(0), w.v)) || (type.Vx[i, j] = :air)
    end
    for j in 2:(nc.y+2), i in 3:(nc.x+2)
        type.Vy[i, j] === :in || continue
        w = φ_loc_y(φ, i, j)
        (φ.Vy[i, j] > 0 || any(>(0), w.c) || any(>(0), w.v)) || (type.Vy[i, j] = :air)
    end
    return type
end

"""
    numbering_free_surface!(number, type, nc)

Equation numbering (Stokes `Numbering!`) without the `:air` unknowns (compacted; periodic copies follow).
"""
function numbering_free_surface!(number, type, nc)
    fill!(number.Vx, 0); fill!(number.Vy, 0); fill!(number.Pt, 0)
    Numbering!(number, type, nc)
    for (N, T) in ((number.Vx, type.Vx), (number.Vy, type.Vy), (number.Pt, type.Pt))
        keep = trues(maximum(N))
        for I in eachindex(N)
            N[I] > 0 && T[I] === :air && (keep[N[I]] = false)
        end
        new = cumsum(keep)
        for I in eachindex(N)
            N[I] > 0 && (N[I] = keep[N[I]] ? new[N[I]] : 0)
        end
    end
    return number
end

"""
    update_free_surface!(fs::VariationalFreeSurface, a, chain, nc, Δ)

Recompute the fluid fractions from the marker chain, flag the `:air` unknowns and zero the air velocities
and pressures. Only if the flags have changed: renumber, reallocate the linear system and drop the stored
Cholesky factor. Call it whenever the chain has moved.
"""
function update_free_surface!(fs::VariationalFreeSurface, a, chain, nc, Δ)
    fluid_fractions!(fs.φ, chain, a.X, Δ, nc; tol=fs.tol)
    type_old = (copy(a.type.Vx), copy(a.type.Vy), copy(a.type.Pt))
    flag_free_surface!(a.type, fs.φ, fs.type0, nc)
    changed = type_old != (a.type.Vx, a.type.Vy, a.type.Pt) || nnz(fs.M.Vx.Vx) == 0
    if changed
        numbering_free_surface!(a.number, a.type, nc)
        fs.M, fs.dx, fs.r = allocate_system(a.number)
        fs.M_PC, _, _ = allocate_system(a.number)
        fs.gcr = StokesGCRCache()
    end
    # fluid fraction of each pressure unknown
    fs.φp = zeros(maximum(a.number.Pt))
    for I in CartesianIndices(a.number.Pt)
        k = a.number.Pt[I]
        k > 0 && a.type.Pt[I] === :in && (fs.φp[k] = fs.φ.c[I])
    end
    a.V.x[a.type.Vx .=== :air] .= 0.0
    a.V.y[a.type.Vy .=== :air] .= 0.0
    a.Pt[a.type.Pt .=== :air]  .= 0.0
    return fs
end

#--------------------------------------------------------------------------------------------------------------#
# Markers: phases from the chain, air removed from the grid phase ratios

# Topography of the chain at x (linear between the grid vertices xv)
@inline function chain_height(x, xv, h)
    i = clamp(searchsortedlast(xv, x), 1, length(xv) - 1)
    t = (x - xv[i]) / (xv[i+1] - xv[i])
    return (1 - t) * h[i] + t * h[i+1]
end

"""
    set_phases_from_chain!(phases, particles, chain, xv; air_phase, rock_phase=nothing, refill_phase=nothing)

Markers above the chain get `air_phase`. Markers below get `rock_phase`, or keep their phase if
`rock_phase === nothing`; air markers that are now below the chain (rising surface) get `refill_phase`
(if given).
"""
function set_phases_from_chain!(phases, particles, chain, xv; air_phase, rock_phase=nothing, refill_phase=nothing)
    h = Array(chain.h_vertices)
    xv = collect(xv)
    Threads.@threads for j in axes(phases, 2)
        for i in axes(phases, 1)
            for ip in cellaxes(phases)
                @index(particles.index[ip, i, j]) == 0 && continue
                x = @index particles.coords[1][ip, i, j]
                y = @index particles.coords[2][ip, i, j]
                if y > chain_height(x, xv, h)
                    @index phases[ip, i, j] = air_phase
                elseif rock_phase !== nothing
                    @index phases[ip, i, j] = rock_phase
                elseif refill_phase !== nothing && @index(phases[ip, i, j]) == air_phase
                    @index phases[ip, i, j] = refill_phase
                end
            end
        end
    end
    return nothing
end

"""
    remove_air_phase!(phase_ratios, air_phase; fallback_phase=1)

In the variational method the fluid fraction is carried by the weights φ, so the material properties of cut
cells must be those of the rock alone: remove `air_phase` from the grid phase ratios and renormalise.
Points without any rock marker get `fallback_phase` (their weight is zero unless the markers under-resolve
the surface).
"""
function remove_air_phase!(phase_ratios, air_phase; fallback_phase=1)
    for pr in (phase_ratios.c, phase_ratios.v), I in eachindex(pr)
        r = pr[I]
        s = sum(r) - r[air_phase]
        r[air_phase] = 0.0
        if s > 1e-12
            r ./= s
        else
            fill!(r, 0.0)
            r[fallback_phase] = 1.0
        end
    end
    return nothing
end

"""
    set_grid_phases_free_surface!(a, adv, chain; air_phase, fallback_phase=1, refill_phase=nothing)

Markers → grid phase ratios for the variational free surface: air above the chain, phase ratios
interpolated to the grid, air removed from them (see `remove_air_phase!`).
"""
function set_grid_phases_free_surface!(a, adv, chain; air_phase, fallback_phase=1, refill_phase=nothing)
    phases = adv.particle_args[1]
    refill = refill_phase === nothing ? nothing : Float64(refill_phase)
    set_phases_from_chain!(phases, adv.particles, chain, a.X.v.x; air_phase=Float64(air_phase), refill_phase=refill)
    Set_PhaseRatios!(a, adv.phase_ratios, adv.particles, phases)
    remove_air_phase!(a.phase_ratios, air_phase; fallback_phase)
    return nothing
end

"""
    fill_velocity_ghosts!(V, type, BC, Δ)

Write the first layer of tangential ghost velocities (Vx S/N rows, Vy W/E columns) from the boundary
conditions. The solver applies BCs only inside the local stencils, so the ghosts of `V` are otherwise stale
(zero), which biases any grid → marker interpolation within half a cell of a boundary.
"""
function fill_velocity_ghosts!(V, type, BC, Δ)
    nx, ny = size(V.x)
    for i in 1:nx, (jg, jn) in ((2, 3), (ny - 1, ny - 2))
        t, s = type.Vx[i, jg], jg == 2 ? -1 : 1
        t === :Dirichlet_tangent && (V.x[i, jg] = 2BC.Vx[i, jg] - V.x[i, jn])
        t === :Neumann_tangent   && (V.x[i, jg] = V.x[i, jn] + s * Δ.y * BC.Vx[i, jg])
    end
    nx, ny = size(V.y)
    for j in 1:ny, (ig, ii) in ((2, 3), (nx - 1, nx - 2))
        t, s = type.Vy[ig, j], ig == 2 ? -1 : 1
        t === :Dirichlet_tangent && (V.y[ig, j] = 2BC.Vy[ig, j] - V.y[ii, j])
        t === :Neumann_tangent   && (V.y[ig, j] = V.y[ii, j] + s * Δ.x * BC.Vy[ig, j])
    end
    return V
end

"""
    plastic_strain_ToParticles_FS!(a, adv, Δt)

Marker plastic strain (second marker field, `adv.particle_args[2]`): εp += Δt ε̇II_pl, with ε̇II_pl = Σᵢ φᵢ ε̇II_plᵢ the
plastic strain rate of the converged step at the centroids (phase ratios without air), interpolated to the markers.
"""
function plastic_strain_ToParticles_FS!(a, adv, Δt)
    εpm  = adv.particle_args[2]
    rate = zero(a.εp.c)
    ε̇pl  = a.εp_ph.ε̇pl.c
    for I in eachindex(rate)
        pr = a.phase_ratios.c[I]
        rate[I] = sum(pr[p] * ε̇pl[p][I] for p in eachindex(ε̇pl))
    end
    zero_gradient!(rate)
    tmp = init_cell_arrays(adv.particles, Val(1))[1]
    centroid2particle!(tmp, rate, adv.particles)
    foreach_marker(adv.particles) do ip, i, j
        @index εpm[ip, i, j] = @index(εpm[ip, i, j]) + Δt * max(@index(tmp[ip, i, j]), 0.0)
    end
    return nothing
end

"""
    plastic_strain_ToGrid_FS!(a, adv)

Grid plastic strain (centroids and vertices, used by the strain softening) = average of the marker plastic strain.
"""
function plastic_strain_ToGrid_FS!(a, adv)
    εpm = adv.particle_args[2]
    particle2centroid!(a.εp.c, εpm, adv.particles)
    particle2grid!(a.εp.v, εpm, adv.particles)
    zero_gradient!(a.εp.c)
    zero_gradient!(a.εp.v)
    return nothing
end

"""
    advect_free_surface!(a, adv, chain, BC, Δ; air_phase, refill_phase, fallback_phase=1, max_slope_angle=45.0, conserve_mean=true)

One transport step (Δ.t) after a Stokes solve: velocity ghosts from the BCs, marker plastic strain increment (if the
markers carry a second field, εp), semi-Lagrangian (RK2) update of the marker chain, RK4 advection of the material
markers, phase-aware injection (εp of new markers interpolated from the grid), air flagging of the markers from the
new chain (air markers that end below the surface get `refill_phase`), new grid phase ratios and grid εp.
Call `update_free_surface!` afterwards to recompute the fluid fractions and flags.
"""
function advect_free_surface!(a, adv, chain, BC, Δ; air_phase, refill_phase, fallback_phase=1, max_slope_angle=45.0, conserve_mean=true)
    Δt = Δ.t
    with_εp = length(adv.particle_args) ≥ 2
    with_εp && plastic_strain_ToParticles_FS!(a, adv, Δt)
    fill_velocity_ghosts!(a.V, a.type, BC, Δ)
    V = (view(a.V.x, 2:(size(a.V.x, 1)-1), 2:(size(a.V.x, 2)-1)), view(a.V.y, 2:(size(a.V.y, 1)-1), 2:(size(a.V.y, 2)-1)))
    semilagrangian_advection_markerchain!(chain, RungeKutta2(), V, adv.grid_vi, (collect(a.X.v.x), collect(a.X.v.y)), Δt; max_slope_angle, conserve_mean)
    advection!(adv.particles, RungeKutta4(), V, Δt)
    move_particles!(adv.particles, adv.particle_args)
    if with_εp
        inject_particles_phase!(adv.particles, adv.particle_args[1], (adv.particle_args[2],), (a.εp.c,))
    else
        inject_particles_phase!(adv.particles, adv.particle_args[1], (), ())
    end
    set_grid_phases_free_surface!(a, adv, chain; air_phase, fallback_phase, refill_phase)
    with_εp && plastic_strain_ToGrid_FS!(a, adv)
    return nothing
end

#--------------------------------------------------------------------------------------------------------------#
# Weighted residuals

# Local stencils of the x-momentum equation of Vx[i, j] (as in Stokes ResidualMomentum2D_x!)
@inline function local_x(V, P, ΔP, τ0, 𝐷, G, type, BC, i, j)
    Vx_loc = SMatrix{3,3}(V.x[ii, jj] for ii in (i-1):(i+1), jj in (j-1):(j+1))
    Vy_loc = SMatrix{4,4}(V.y[ii, jj] for ii in (i-1):(i+2), jj in (j-2):(j+1))
    bcx_loc = SMatrix{3,3}(BC.Vx[ii, jj] for ii in (i-1):(i+1), jj in (j-1):(j+1))
    bcy_loc = SMatrix{4,4}(BC.Vy[ii, jj] for ii in (i-1):(i+2), jj in (j-2):(j+1))
    typex_loc = SMatrix{3,3}(type.Vx[ii, jj] for ii in (i-1):(i+1), jj in (j-1):(j+1))
    typey_loc = SMatrix{4,4}(type.Vy[ii, jj] for ii in (i-1):(i+2), jj in (j-2):(j+1))
    typep_loc = SMatrix{2,1}(type.Pt[ii, jj] for ii in (i-1):(i-0), jj in (j-1):(j-1))
    P_loc = SMatrix{2,3}(P[ii, jj] for ii in (i-1):i, jj in (j-2):j)
    ΔP_loc = SMatrix{2,1}(ΔP.c[ii, jj] for ii in (i-1):i, jj in (j-1):(j-1))
    τxx0 = SMatrix{2,3}(τ0.xx[ii, jj] for ii in (i-1):i, jj in (j-2):j)
    τyy0 = SMatrix{2,3}(τ0.yy[ii, jj] for ii in (i-1):i, jj in (j-2):j)
    τxy0 = SMatrix{3,2}(τ0.xy[ii, jj] for ii in (i-1):(i+1), jj in (j-1):j)
    Gc_loc = SMatrix{2,1}(G.c[ii, jj] for ii in (i-1):i, jj in (j-1):(j-1))
    Gv_loc = SMatrix{1,2}(G.v[ii, jj] for ii in (i-0):(i-0), jj in (j-1):(j-0))
    Dc = SMatrix{2,1}(𝐷.c[ii, jj] for ii in (i-1):i, jj in (j-1):(j-1))
    Dv = SMatrix{1,2}(𝐷.v[ii, jj] for ii in (i-0):(i-0), jj in (j-1):(j-0))
    return Vx_loc, Vy_loc, P_loc, ΔP_loc, (xx=τxx0, yy=τyy0, xy=τxy0), (c=Gc_loc, v=Gv_loc), (c=Dc, v=Dv),
           (x=typex_loc, y=typey_loc, p=typep_loc), (x=bcx_loc, y=bcy_loc)
end

# Local stencils of the y-momentum equation of Vy[i, j] (as in Stokes ResidualMomentum2D_y!)
@inline function local_y(V, P, ΔP, τ0, 𝐷, G, ρ, type, BC, i, j)
    Vx_loc = SMatrix{4,4}(V.x[ii, jj] for ii in (i-2):(i+1), jj in (j-1):(j+2))
    Vy_loc = SMatrix{3,3}(V.y[ii, jj] for ii in (i-1):(i+1), jj in (j-1):(j+1))
    bcx_loc = SMatrix{4,4}(BC.Vx[ii, jj] for ii in (i-2):(i+1), jj in (j-1):(j+2))
    bcy_loc = SMatrix{3,3}(BC.Vy[ii, jj] for ii in (i-1):(i+1), jj in (j-1):(j+1))
    typex_loc = SMatrix{4,4}(type.Vx[ii, jj] for ii in (i-2):(i+1), jj in (j-1):(j+2))
    typey_loc = SMatrix{3,3}(type.Vy[ii, jj] for ii in (i-1):(i+1), jj in (j-1):(j+1))
    P_loc = SMatrix{3,2}(P[ii, jj] for ii in (i-2):i, jj in (j-1):j)
    ΔP_loc = SMatrix{1,2}(ΔP.c[ii, jj] for ii in (i-1):(i-1), jj in (j-1):j)
    Gc_loc = SMatrix{1,2}(G.c[ii, jj] for ii in (i-1):(i-1), jj in (j-1):j)
    Gv_loc = SMatrix{2,1}(G.v[ii, jj] for ii in (i-1):(i-0), jj in (j-0):(j-0))
    ρ_loc = SMatrix{1,2}(ρ.c[ii, jj] for ii in (i-1):(i-1), jj in (j-1):j)
    τxx0 = SMatrix{3,2}(τ0.xx[ii, jj] for ii in (i-2):i, jj in (j-1):j)
    τyy0 = SMatrix{3,2}(τ0.yy[ii, jj] for ii in (i-2):i, jj in (j-1):j)
    τxy0 = SMatrix{2,3}(τ0.xy[ii, jj] for ii in (i-1):i, jj in (j-1):(j+1))
    Dc = SMatrix{1,2}(𝐷.c[ii, jj] for ii in (i-1):(i-1), jj in (j-1):j)
    Dv = SMatrix{2,1}(𝐷.v[ii, jj] for ii in (i-1):(i-0), jj in (j-0):(j-0))
    return Vx_loc, Vy_loc, P_loc, ΔP_loc, (xx=τxx0, yy=τyy0, xy=τxy0), (c=Gc_loc, v=Gv_loc), ρ_loc, (c=Dc, v=Dv),
           (x=typex_loc, y=typey_loc), (x=bcx_loc, y=bcy_loc)
end

function ResidualMomentum2D_x_FS!(R, a, 𝐷, φ, materials, BC, nc, Δ)
    fill!(R.x, 0.0)
    Threads.@threads for j in 3:(nc.y+2)
        for i in 2:(nc.x+2)
            a.type.Vx[i, j] === :in || continue
            Vx, Vy, P, ΔP, τ0, G, D, t, bc = local_x(a.V, a.Pt, a.ΔPt, a.τ0, 𝐷, a.G, a.type, BC, i, j)
            R.x[i, j] = SMomentum_x_Generic(Vx, Vy, P, ΔP, τ0, G, D, materials, t, bc, Δ; φ=φ_loc_x(φ, i, j))
        end
    end
    return nothing
end

function ResidualMomentum2D_y_FS!(R, a, 𝐷, φ, materials, BC, nc, Δ; θ_fs=0.0)
    fill!(R.y, 0.0)
    Threads.@threads for j in 2:(nc.y+2)
        for i in 3:(nc.x+2)
            a.type.Vy[i, j] === :in || continue
            Vx, Vy, P, ΔP, τ0, G, ρ, D, t, bc = local_y(a.V, a.Pt, a.ΔPt, a.τ0, 𝐷, a.G, a.ρ, a.type, BC, i, j)
            R.y[i, j] = SMomentum_y_Generic(Vx, Vy, P, ΔP, τ0, G, ρ, D, materials, t, bc, Δ; φ=φ_loc_y(φ, i, j), θ_fs)
        end
    end
    return nothing
end

# Continuity weighted by φc (keeps the system symmetric, φP Gᵀ in Gray et al., 2026, Eq. 10)
function ResidualContinuity2D_FS!(R, a, φ, materials, nc, Δ)
    fill!(R.p, 0.0)
    Threads.@threads for j in 2:(nc.y+1)
        for i in 2:(nc.x+1)
            a.type.Pt[i, j] === :in || continue
            Vx = SMatrix{2,3}(a.V.x[ii, jj] for ii in i:(i+1), jj in j:(j+2))
            Vy = SMatrix{3,2}(a.V.y[ii, jj] for ii in i:(i+2), jj in j:(j+1))
            R.p[i, j] = φ.c[i, j] * Continuity(Vx, Vy, a.Pt[i, j], a.Pt0[i, j], (;), a.β.c[i, j], a.ξ.c[i, j], materials, (;), (;), Δ)
        end
    end
    return nothing
end

"""
    residuals_free_surface!(a, fs, materials, BC, nc, Δ)

Weighted Stokes residuals (into `a.R`), with the current tangent operator `a.𝐷`.
"""
function residuals_free_surface!(a, fs, materials, BC, nc, Δ)
    ResidualContinuity2D_FS!(a.R, a, fs.φ, materials, nc, Δ)
    ResidualMomentum2D_x_FS!(a.R, a, a.𝐷, fs.φ, materials, BC, nc, Δ)
    ResidualMomentum2D_y_FS!(a.R, a, a.𝐷, fs.φ, materials, BC, nc, Δ; fs.θ_fs)
    return nothing
end

#--------------------------------------------------------------------------------------------------------------#
# Weighted Jacobian

function AssembleMomentum2D_x_FS!(K, a, 𝐷, φ, materials, BC, nc, Δ)
    num, pattern = a.number, a.pattern
    assembly_loop(K, 3:(nc.y+2)) do j, tid
        @inbounds for i in 2:(nc.x+2)
            a.type.Vx[i, j] === :in || continue
            row = num.Vx[i, j]
            row > 0 || continue
            Vx, Vy, P, ΔP, τ0, G, D, t, bc = local_x(a.V, a.Pt, a.ΔPt, a.τ0, 𝐷, a.G, a.type, BC, i, j)
            w = φ_loc_x(φ, i, j)
            ∂R∂Vx = ad_gradient(x -> SMomentum_x_Generic(x, Vy, P, ΔP, τ0, G, D, materials, t, bc, Δ; φ=w), Vx)
            ∂R∂Vy = ad_gradient(x -> SMomentum_x_Generic(Vx, x, P, ΔP, τ0, G, D, materials, t, bc, Δ; φ=w), Vy)
            ∂R∂Pt = ad_gradient(x -> SMomentum_x_Generic(Vx, Vy, x, ΔP, τ0, G, D, materials, t, bc, Δ; φ=w), P)
            Kt = _target(K, tid)
            Local = SMatrix{3,3}(num.Vx[ii, jj] for ii in (i-1):(i+1), jj in (j-1):(j+1)) .* pattern[1][1]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                Local[ii, jj] > 0 && (Kt[1][1][row, Local[ii, jj]] = ∂R∂Vx[ii, jj])
            end
            Local = SMatrix{4,4}(num.Vy[ii, jj] for ii in (i-1):(i+2), jj in (j-2):(j+1)) .* pattern[1][2]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                Local[ii, jj] > 0 && (Kt[1][2][row, Local[ii, jj]] = ∂R∂Vy[ii, jj])
            end
            Local = SMatrix{2,3}(num.Pt[ii, jj] for ii in (i-1):i, jj in (j-2):j) .* pattern[1][3]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                Local[ii, jj] > 0 && (Kt[1][3][row, Local[ii, jj]] = ∂R∂Pt[ii, jj])
            end
        end
    end
    return nothing
end

function AssembleMomentum2D_y_FS!(K, a, 𝐷, φ, materials, BC, nc, Δ; θ_fs=0.0)
    num, pattern = a.number, a.pattern
    assembly_loop(K, 2:(nc.y+2)) do j, tid
        @inbounds for i in 3:(nc.x+2)
            a.type.Vy[i, j] === :in || continue
            row = num.Vy[i, j]
            row > 0 || continue
            Vx, Vy, P, ΔP, τ0, G, ρ, D, t, bc = local_y(a.V, a.Pt, a.ΔPt, a.τ0, 𝐷, a.G, a.ρ, a.type, BC, i, j)
            w = φ_loc_y(φ, i, j)
            ∂R∂Vx = ad_gradient(x -> SMomentum_y_Generic(x, Vy, P, ΔP, τ0, G, ρ, D, materials, t, bc, Δ; φ=w, θ_fs), Vx)
            ∂R∂Vy = ad_gradient(x -> SMomentum_y_Generic(Vx, x, P, ΔP, τ0, G, ρ, D, materials, t, bc, Δ; φ=w, θ_fs), Vy)
            ∂R∂Pt = ad_gradient(x -> SMomentum_y_Generic(Vx, Vy, x, ΔP, τ0, G, ρ, D, materials, t, bc, Δ; φ=w, θ_fs), P)
            Kt = _target(K, tid)
            Local = SMatrix{4,4}(num.Vx[ii, jj] for ii in (i-2):(i+1), jj in (j-1):(j+2)) .* pattern[2][1]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                Local[ii, jj] > 0 && (Kt[2][1][row, Local[ii, jj]] = ∂R∂Vx[ii, jj])
            end
            Local = SMatrix{3,3}(num.Vy[ii, jj] for ii in (i-1):(i+1), jj in (j-1):(j+1)) .* pattern[2][2]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                Local[ii, jj] > 0 && (Kt[2][2][row, Local[ii, jj]] = ∂R∂Vy[ii, jj])
            end
            Local = SMatrix{3,2}(num.Pt[ii, jj] for ii in (i-2):i, jj in (j-1):j) .* pattern[2][3]
            for jj in axes(Local, 2), ii in axes(Local, 1)
                Local[ii, jj] > 0 && (Kt[2][3][row, Local[ii, jj]] = ∂R∂Pt[ii, jj])
            end
        end
    end
    return nothing
end

function AssembleContinuity2D_FS!(K, a, φ, materials, nc, Δ)
    num = a.number
    assembly_loop(K, 2:(nc.y+1)) do j, tid
        @inbounds for i in 2:(nc.x+1)
            a.type.Pt[i, j] === :in || continue
            row = num.Pt[i, j]
            row > 0 || continue
            Vx = SMatrix{2,3}(a.V.x[ii, jj] for ii in i:(i+1), jj in j:(j+2))
            Vy = SMatrix{3,2}(a.V.y[ii, jj] for ii in i:(i+2), jj in j:(j+1))
            P = SMatrix{1,1}(a.Pt[i, j])
            φc = φ.c[i, j]
            f(vx, vy, p) = φc * Continuity(vx, vy, p, a.Pt0[i, j], (;), a.β.c[i, j], a.ξ.c[i, j], materials, (;), (;), Δ)
            ∂R∂Vx, ∂R∂Vy, ∂R∂P = ad_partial_gradients(f, (Vx, Vy, P))
            Kt = _target(K, tid)
            Local = SMatrix{2,3}(num.Vx[ii, jj] for ii in i:(i+1), jj in j:(j+2))
            for jj in axes(Local, 2), ii in axes(Local, 1)
                Local[ii, jj] > 0 && (Kt[3][1][row, Local[ii, jj]] = ∂R∂Vx[ii, jj])
            end
            Local = SMatrix{3,2}(num.Vy[ii, jj] for ii in i:(i+2), jj in j:(j+1))
            for jj in axes(Local, 2), ii in axes(Local, 1)
                Local[ii, jj] > 0 && (Kt[3][2][row, Local[ii, jj]] = ∂R∂Vy[ii, jj])
            end
            Kt[3][3][row, row] = ∂R∂P[1, 1]
        end
    end
    return nothing
end

function _assemble_blocks_FS!(K, a, 𝐷, φ, materials, BC, nc, Δ; θ_fs=0.0)
    AssembleContinuity2D_FS!(K, a, φ, materials, nc, Δ)
    AssembleMomentum2D_x_FS!(K, a, 𝐷, φ, materials, BC, nc, Δ)
    AssembleMomentum2D_y_FS!(K, a, 𝐷, φ, materials, BC, nc, Δ; θ_fs)
end

"""
    assemble_free_surface!(fs, a, 𝐷, materials, BC, nc, Δ; M=fs.M)

Assemble the weighted Stokes Jacobian into `M` (default `fs.M`; pattern recorded on the first call after
`update_free_surface!`). Pass `a.𝐷_ctl` for Newton, `a.𝐷` for Picard (and `M=fs.M_PC` for the preconditioner).
"""
function assemble_free_surface!(fs, a, 𝐷, materials, BC, nc, Δ; M=fs.M)
    if nnz(M.Vx.Vx) == 0
        K_pat = pattern_storage(M)
        _assemble_blocks_FS!(K_pat, a, 𝐷, fs.φ, materials, BC, nc, Δ; fs.θ_fs)
        set_sparsity_pattern!(M, K_pat)
    end
    zero_values!(M)
    _assemble_blocks_FS!(fixed_pattern(M), a, 𝐷, fs.φ, materials, BC, nc, Δ; fs.θ_fs)
    return M
end

#--------------------------------------------------------------------------------------------------------------#
# Solver

# Tangent operator with a NaN-free consistent tangent: the AD tangent of the rheology is NaN where the strain rate
# vanishes (air cells, rigid blocks); there the Picard tangent is used (in air cells it is multiplied by φ = 0 anyway)
# εp (accumulated plastic strain, strain softening) and εp_ph (per-phase plastic strain rates, for the marker εp) are passed
function tangent_FS!(a, materials, BC, Δ)
    TangentOperator!(a.𝐷, a.𝐷_ctl, a.τ, a.τ0, a.ε̇, a.λ̇, a.η, a.G, a.V, a.Pt, a.Pt0, a.ΔPt, a.type, BC, materials, a.phase_ratios, Δ;
                     εp=a.εp, εp_ph=a.εp_ph)
    for (Dc, D) in ((a.𝐷_ctl.c, a.𝐷.c), (a.𝐷_ctl.v, a.𝐷.v)), I in eachindex(Dc)
        any(isnan, Dc[I]) && (Dc[I] .= D[I])
    end
    return nothing
end

# Residual norm (as in solve_free_surface!) of the current state
function residual_norm_FS!(a, fs, materials, BC, nc, Δ, n)
    tangent_FS!(a, materials, BC, Δ)
    residuals_free_surface!(a, fs, materials, BC, nc, Δ)
    return max(norm(a.R.x) / sqrt(n[1]), norm(a.R.y) / sqrt(n[2]), norm(a.R.p) / sqrt(n[3]))
end

# Apply α fs.dx, halving α (down to α_min) until the residual decreases below ϵ0; returns (α, success). On failure
# the state is restored (accept=false) or the α_min step is kept (accept=true)
function backtrack!(a, fs, materials, BC, nc, Δ, ϵ0, n; α_min=1/64, accept=true)
    Vx0, Vy0, Pt0 = copy(a.V.x), copy(a.V.y), copy(a.Pt)
    α = 1.0
    while true
        a.V.x .= Vx0; a.V.y .= Vy0; a.Pt .= Pt0
        UpdateSolution!(a.V, a.Pt, α .* fs.dx, a.number, a.type, nc)
        residual_norm_FS!(a, fs, materials, BC, nc, Δ, n) < ϵ0 && return α, true
        if α ≤ α_min
            accept || (a.V.x .= Vx0; a.V.y .= Vy0; a.Pt .= Pt0)
            return α, false
        end
        α /= 2
    end
end

# Linear solve Mop fs.dx = -fs.r with the GCR solver of Solvers.jl (StokesGCRCache): one Powell-Hestenes step with
# the Cholesky factor of the Schur complement as preconditioner, penalty scaled by the fluid fraction φp of the pressure
# cells; Kuu from the Picard operator (Picard steps, pc_type = :picard) or symmetrised Newton Kuu (pc_type = :sym_newton)
# ϵ_abs: absolute GCR tolerance (block residual norms, normalised as the nonlinear residual)
function linear_solve_FS!(fs, ϵ_abs, γ, maxit, noisy; newton=false, pc_type=:picard)
    pc_type in (:picard, :sym_newton) || error("pc_type must be :picard or :sym_newton (got :$pc_type)")
    fs.gcr.initialized || setup_gcr!(fs.gcr, fs.M_PC, fs.M_PC; restart=30)   # M and M_PC share the sparsity pattern
    M_N = (newton && pc_type === :sym_newton) ? fs.M : nothing
    update_preconditioner!(fs.gcr, fs.M_PC; ηb=γ, M_N, φp=fs.φp)
    update_operator!(fs.gcr, newton ? fs.M : fs.M_PC)
    fill!(fs.dx, 0.0)
    its = KSP_GCR_Stokes!(fs.dx, fs.gcr, fs.r; ϵ_l=ϵ_abs, maxit, noisy=false)
    noisy && @printf("         GCR: %d iterations\n", its)
    return its
end

"""
    iter, err = solve_free_surface!(a, fs, materials, BC, nc, Δ, nphases; niter=10, ϵ_nl=1e-10, γ=1e5, ϵ_l=1e-10, maxit=500, pc_type=:picard, noisy=true)

Iterations on the weighted Stokes system: Picard first, then Newton once the residual has decreased by `Pic2Newt`
(Inf: Newton from the second iteration). Newton steps that the backtracking cannot make reduce the residual are
replaced by Picard steps. Converged when the residual is below `ϵ_nl` or below `ϵ_rel` × the initial residual.
Linear solver: GCR of Solvers.jl (`linear_solve_FS!`), tolerance `ϵ_l` relative to the current residual, or inexact
Newton (`inexact = true`: `linear_tol` of Solvers.jl, first solve 1/10 of the residual, then tightened as the residual
decreases), `maxit` iterations; preconditioner (Newton steps) from the Picard (`pc_type = :picard`) or symmetrised Newton Kuu (`:sym_newton`).
Returns the number of iterations and the history of the residual norms (x, y, p).
"""
function solve_free_surface!(a, fs, materials, BC, nc, Δ, nphases; niter=10, ϵ_nl=1e-10, γ=1e5, ϵ_l=1e-10, inexact=false, maxit=500, pc_type=:picard, line_search=true, Pic2Newt=Inf, ϵ_rel=0.0, noisy=true)
    compute_grid_fields!(a.G, a.β, a.ρ, a.ξ, materials, a.phase_ratios, nc, nphases)
    nVx, nVy, nPt = maximum(a.number.Vx), maximum(a.number.Vy), maximum(a.number.Pt)
    err = (x=zeros(niter), y=zeros(niter), p=zeros(niter))
    iter, ϵ0 = 0, 1.0
    while iter < niter
        iter += 1
        tangent_FS!(a, materials, BC, Δ)
        residuals_free_surface!(a, fs, materials, BC, nc, Δ)
        err.x[iter] = norm(a.R.x) / sqrt(nVx)
        err.y[iter] = norm(a.R.y) / sqrt(nVy)
        err.p[iter] = norm(a.R.p) / sqrt(nPt)
        ϵ = max(err.x[iter], err.y[iter], err.p[iter])
        noisy && @printf("Iter. %02d: ‖Rx‖ = %1.3e, ‖Ry‖ = %1.3e, ‖Rp‖ = %1.3e\n", iter, err.x[iter], err.y[iter], err.p[iter])
        iter == 1 && (ϵ0 = ϵ)
        (ϵ < ϵ_nl || ϵ < ϵ_rel * ϵ0) && break
        iter == niter && break
        SetRHS!(fs.r, a.R, a.number, a.type, nc)
        # Picard first (the consistent tangent of a power law is NaN at zero strain rate, e.g. V = 0), then Newton
        newton = iter > 1 && ϵ / ϵ0 < Pic2Newt
        assemble_free_surface!(fs, a, a.𝐷, materials, BC, nc, Δ; M=fs.M_PC)
        newton && assemble_free_surface!(fs, a, a.𝐷_ctl, materials, BC, nc, Δ; M=fs.M)
        # GCR tolerance: ϵ_l relative to the current residual, or inexact Newton (linear_tol of Solvers.jl, as the main solver)
        ϵ_abs = inexact ? linear_tol(ϵ, ϵ0, iter; α=50) : ϵ_l * ϵ
        linear_solve_FS!(fs, ϵ_abs, γ, maxit, noisy; newton, pc_type)
        if line_search
            # Backtracking on the residual; a Newton step that does not reduce it is replaced by a Picard step
            α, ok = backtrack!(a, fs, materials, BC, nc, Δ, ϵ, (nVx, nVy, nPt); accept=!newton)
            if !ok && newton
                noisy && @printf("         Newton step rejected → Picard step\n")
                linear_solve_FS!(fs, ϵ_abs, γ, maxit, noisy; newton=false, pc_type)
                α, ok = backtrack!(a, fs, materials, BC, nc, Δ, ϵ, (nVx, nVy, nPt); accept=true)
            end
        else
            α = 1.0
            UpdateSolution!(a.V, a.Pt, fs.dx, a.number, a.type, nc)
        end
        noisy && α < 1 && @printf("         line search: α = %.3f\n", α)
    end
    a.Pt .+= a.ΔPt.c
    return iter, err
end
