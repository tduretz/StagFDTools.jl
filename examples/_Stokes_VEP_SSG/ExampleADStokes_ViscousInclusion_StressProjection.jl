using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, CairoMakie, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
import Random
using DifferentiationInterface
using TimerOutputs

# Grid -> marker stress interpolation that preserves discrete force balance.
#
# After the Stokes solve, stresses are known on the grid (τxx, τyy, P at centroids, τxy at vertices).
# Markers receive them by:
#   (3) standard bilinear interpolation  ->  s̃
#   (4) bilinear interpolation + equilibrium projection:
#       s = s̃ - W⁻¹Bᵀ (B W⁻¹ Bᵀ)⁻¹ (B s̃ - f)
#       where B = D Π, Π is the marker -> grid stress map and D the staggered discrete divergence.
#       s is the closest field to s̃ (in the W-norm) whose marker -> grid projection is exactly in
#       discrete force balance, i.e. it produces no spurious forces when fed back to the solver.

#--------------------------------------------#
# Bilinear interpolation of a field F defined on a regular grid (x0, y0, Δx, Δy) to point (xm, ym)
function bilinear(F, x0, y0, Δx, Δy, xm, ym)
    i  = clamp(floor(Int, (xm - x0) / Δx) + 1, 1, size(F, 1) - 1)
    j  = clamp(floor(Int, (ym - y0) / Δy) + 1, 1, size(F, 2) - 1)
    wx = (xm - (x0 + (i - 1) * Δx)) / Δx
    wy = (ym - (y0 + (j - 1) * Δy)) / Δy
    return (1 - wx) * (1 - wy) * F[i, j] + wx * (1 - wy) * F[i+1, j] +
           (1 - wx) *      wy  * F[i, j+1] + wx *      wy  * F[i+1, j+1]
end

# Marker -> grid map: each node gets the bilinear-weighted average of the markers around it
# Nodes are (x0 + (i-1)Δx, y0 + (j-1)Δy), i = 1:nx, j = 1:ny; nodes with active[i,j] = false get empty rows
function marker_to_grid_matrix(xm, ym, x0, y0, Δx, Δy, nx, ny, active)
    I, J, V = Int[], Int[], Float64[]
    LI = LinearIndices((nx, ny))
    for m in eachindex(xm)
        i0 = floor(Int, (xm[m] - x0) / Δx) + 1
        j0 = floor(Int, (ym[m] - y0) / Δy) + 1
        for j in j0:j0+1, i in i0:i0+1
            (1 <= i <= nx && 1 <= j <= ny && active[i, j]) || continue
            w = (1 - abs(xm[m] - (x0 + (i - 1) * Δx)) / Δx) * (1 - abs(ym[m] - (y0 + (j - 1) * Δy)) / Δy)
            w > 0 || continue
            push!(I, LI[i, j]); push!(J, m); push!(V, w)
        end
    end
    Π = sparse(I, J, V, nx * ny, length(xm))
    wsum = vec(sum(Π, dims=2))
    return spdiagm(1 ./ max.(wsum, eps())) * Π
end

# Staggered discrete divergence of the total stress σ = τ - P I, (∂σxx/∂x + ∂σxy/∂y, ∂σxy/∂x + ∂σyy/∂y)
# on interior Vx and Vy nodes (free slip: boundary normal velocities are prescribed and carry no equation).
# Returns the blocks such that  Rx = Gx_c*σxx + Gy_v*τxy,  Ry = Gy_c*σyy + Gx_v*τxy
function divergence_operators(nc, Δ)
    C, Vt = LinearIndices((nc.x, nc.y)), LinearIndices((nc.x + 1, nc.y + 1))
    nVx, nVy = (nc.x - 1) * nc.y, nc.x * (nc.y - 1)
    # Two-point difference operator assembled from triplets: row k gets +1/h at column ip, -1/h at column im
    diff_op(rows, ip, im, h, nrow, ncol) = sparse([rows; rows], [ip; im], [fill(1 / h, length(rows)); fill(-1 / h, length(rows))], nrow, ncol)
    Ix  = vec(LinearIndices((nc.x - 1, nc.y)))
    ijx = vec([(i, j) for i in 2:nc.x, j in 1:nc.y])
    Gx_c = diff_op(Ix, [C[i, j] for (i, j) in ijx],    [C[i-1, j] for (i, j) in ijx], Δ.x, nVx, nc.x * nc.y)
    Gy_v = diff_op(Ix, [Vt[i, j+1] for (i, j) in ijx], [Vt[i, j] for (i, j) in ijx],  Δ.y, nVx, (nc.x + 1) * (nc.y + 1))
    Iy  = vec(LinearIndices((nc.x, nc.y - 1)))
    ijy = vec([(i, j) for i in 1:nc.x, j in 2:nc.y])
    Gy_c = diff_op(Iy, [C[i, j] for (i, j) in ijy],    [C[i, j-1] for (i, j) in ijy], Δ.y, nVy, nc.x * nc.y)
    Gx_v = diff_op(Iy, [Vt[i+1, j] for (i, j) in ijy], [Vt[i, j] for (i, j) in ijy],  Δ.x, nVy, (nc.x + 1) * (nc.y + 1))
    return Gx_c, Gy_v, Gy_c, Gx_v
end

# Discrete force balance residual of grid stresses
function grid_residual(ops, τxx_c, τyy_c, τxy_v, P_c)
    Gx_c, Gy_v, Gy_c, Gx_v = ops
    Rx = Gx_c * (vec(τxx_c) .- vec(P_c)) .+ Gy_v * vec(τxy_v)
    Ry = Gy_c * (vec(τyy_c) .- vec(P_c)) .+ Gx_v * vec(τxy_v)
    return Rx, Ry
end

@views function marker_stress_comparison(a, nc, Δ, L, rad; nmark=4, w_mixed=1.0)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    X = a.X

    # Grid stresses (interior centroids / all vertices)
    P_c   = a.Pt[inx_c, iny_c] .- mean(a.Pt[inx_c, iny_c])
    τxx_c = a.τ.xx[inx_c, iny_c]
    τyy_c = a.τ.yy[inx_c, iny_c]
    τxy_v = a.τ.xy[inx_v, iny_v]
    τzz_c = .-(τxx_c .+ τyy_c)

    ops = divergence_operators(nc, Δ)
    Rx_grid, Ry_grid = grid_residual(ops, τxx_c, τyy_c, τxy_v, P_c)
    @printf("Grid solution:            max|R| = %.3e  (reference: discrete balance of the solver)\n", max(maximum(abs, Rx_grid), maximum(abs, Ry_grid)))
    @printf("Boundary vertices:        max|τxy| = %.3e\n", max(maximum(abs, τxy_v[1, :]), maximum(abs, τxy_v[end, :]), maximum(abs, τxy_v[:, 1]), maximum(abs, τxy_v[:, end])))

    # (2) Markers: nmark × nmark per cell, randomly perturbed
    Random.seed!(1234)
    xm, ym = Float64[], Float64[]
    for j in 1:nc.y, i in 1:nc.x, jm in 1:nmark, im in 1:nmark
        push!(xm, X.v.x[i] + (im - 0.5 + 0.4 * (rand() - 0.5)) * Δ.x / nmark)
        push!(ym, X.v.y[j] + (jm - 0.5 + 0.4 * (rand() - 0.5)) * Δ.y / nmark)
    end
    Nm    = length(xm)
    phase = [(xm[m]^2 + ym[m]^2 <= rad^2) ? 2 : 1 for m in 1:Nm]

    # (3) Standard bilinear interpolation (centroid fields extended with zero-gradient ghosts)
    ext(F) = (G = zeros(size(F, 1) + 2, size(F, 2) + 2); G[2:end-1, 2:end-1] .= F;
              G[1, :] .= G[2, :]; G[end, :] .= G[end-1, :]; G[:, 1] .= G[:, 2]; G[:, end] .= G[:, end-1]; G)
    τxx_e, τyy_e, P_e = ext(τxx_c), ext(τyy_c), ext(P_c)
    xc0, yc0 = X.c_e.x[1], X.c_e.y[1]
    xv0, yv0 = X.v.x[1], X.v.y[1]
    τxx_b = [bilinear(τxx_e, xc0, yc0, Δ.x, Δ.y, xm[m], ym[m]) for m in 1:Nm]
    τyy_b = [bilinear(τyy_e, xc0, yc0, Δ.x, Δ.y, xm[m], ym[m]) for m in 1:Nm]
    P_b   = [bilinear(P_e,   xc0, yc0, Δ.x, Δ.y, xm[m], ym[m]) for m in 1:Nm]
    τxy_b = [bilinear(τxy_v, xv0, yv0, Δ.x, Δ.y, xm[m], ym[m]) for m in 1:Nm]
    τzz_b = .-(τxx_b .+ τyy_b)

    # Marker -> grid maps; boundary τxy is set by the free slip condition (τxy = 0), so it has no marker contribution
    Πc = marker_to_grid_matrix(xm, ym, X.c.x[1], X.c.y[1], Δ.x, Δ.y, nc.x, nc.y, trues(nc.x, nc.y))
    act_v = trues(nc.x + 1, nc.y + 1); act_v[1, :] .= false; act_v[end, :] .= false; act_v[:, 1] .= false; act_v[:, end] .= false
    Πv = marker_to_grid_matrix(xm, ym, xv0, yv0, Δ.x, Δ.y, nc.x + 1, nc.y + 1, act_v)

    # Constraint B s = f with s = [τxx; τyy; τxy; P] on markers (rows scaled by Δx for conditioning)
    Gx_c, Gy_v, Gy_c, Gx_v = ops
    Z  = spzeros(size(Gx_c, 1), Nm)
    Zy = spzeros(size(Gy_c, 1), Nm)
    B  = Δ.x .* [Gx_c*Πc  Z        Gy_v*Πv  -Gx_c*Πc;
                 Zy       Gy_c*Πc  Gx_v*Πv  -Gy_c*Πc]
    f  = zeros(size(B, 1))   # no body force here (g = 0)

    # (4) Equilibrium projection. W: confidence per marker; markers in cells cut by the interface can be
    # made cheaper to modify (w_mixed < 1) so that the correction goes where bilinear interpolation is least reliable
    cell_of(m) = (clamp(floor(Int, (xm[m] - xv0) / Δ.x) + 1, 1, nc.x), clamp(floor(Int, (ym[m] - yv0) / Δ.y) + 1, 1, nc.y))
    mixed = falses(nc.x, nc.y)
    for m in 1:Nm
        i, j = cell_of(m)
        mixed[i, j] |= phase[m] != phase[1 + (m - 1) - mod(m - 1, nmark^2)]   # differs from first marker of the cell
    end
    w     = [mixed[cell_of(m)...] ? w_mixed : 1.0 for m in 1:Nm]
    Winv  = spdiagm(repeat(1 ./ w, 4))
    s̃     = [τxx_b; τyy_b; τxy_b; P_b]
    S     = B * Winv * B'
    λ     = cholesky(Symmetric(S + 1e-12 * mean(diag(S)) * I)) \ (B * s̃ .- f)
    s     = s̃ .- Winv * (B' * λ)
    τxx_p, τyy_p, τxy_p, P_p = s[1:Nm], s[Nm+1:2Nm], s[2Nm+1:3Nm], s[3Nm+1:4Nm]
    τzz_p = .-(τxx_p .+ τyy_p)

    # Round trip marker -> grid and discrete force balance
    back(τxx, τyy, τxy, P) = (reshape(Πc * τxx, nc.x, nc.y), reshape(Πc * τyy, nc.x, nc.y), reshape(Πv * τxy, nc.x + 1, nc.y + 1), reshape(Πc * P, nc.x, nc.y))
    Rx_b, Ry_b = grid_residual(ops, back(τxx_b, τyy_b, τxy_b, P_b)...)
    Rx_p, Ry_p = grid_residual(ops, back(τxx_p, τyy_p, τxy_p, P_p)...)
    @printf("Bilinear  -> grid:        max|R| = %.3e\n", max(maximum(abs, Rx_b), maximum(abs, Ry_b)))
    @printf("Projected -> grid:        max|R| = %.3e\n", max(maximum(abs, Rx_p), maximum(abs, Ry_p)))
    @printf("Marker change (projected - bilinear): max|ΔP| = %.3e, max|Δτxx| = %.3e, max|Δτxy| = %.3e (max|τxx| = %.3e)\n",
        maximum(abs, P_p .- P_b), maximum(abs, τxx_p .- τxx_b), maximum(abs, τxy_p .- τxy_b), maximum(abs, τxx_b))

    #--------------------------------------------#
    # Figure: grid (FD) solution, markers (bilinear / projected / difference) and residual of the marker -> grid round trip
    Fig = Figure(size=(2000, 1350), fontsize=14)
    θ   = LinRange(0, 2π, 200)
    # Each marker sits in its own sub-cell (nmark × nmark per cell), so marker values are shown as an image, one pixel per marker
    xs  = X.v.x[1] .+ ((1:nmark*nc.x) .- 0.5) .* Δ.x / nmark
    ys  = X.v.y[1] .+ ((1:nmark*nc.y) .- 0.5) .* Δ.y / nmark
    to_image(val) = reshape(permutedims(reshape(val, nmark, nmark, nc.x, nc.y), (1, 3, 2, 4)), nmark * nc.x, nmark * nc.y)
    function marker_panel(pos, title, val; cmap=:redsblues, crange=nothing)
        ax = Axis(Fig[pos...], aspect=DataAspect(), title=title, xlabel="x", ylabel="y")
        cr = isnothing(crange) ? (m = maximum(abs, val); (-m, m)) : crange
        sc = heatmap!(ax, xs, ys, to_image(val), colormap=cmap, colorrange=cr)
        lines!(ax, rad .* cos.(θ), rad .* sin.(θ), color=:black, linewidth=1)
        Colorbar(Fig[pos[1], pos[2]+1], sc)
        return ax
    end
    function grid_panel(pos, title, val; cmap=:redsblues, crange)
        ax = Axis(Fig[pos...], aspect=DataAspect(), title=title, xlabel="x", ylabel="y")
        hm = heatmap!(ax, X.c.x, X.c.y, val, colormap=cmap, colorrange=crange)
        lines!(ax, rad .* cos.(θ), rad .* sin.(θ), color=:black, linewidth=1)
        Colorbar(Fig[pos[1], pos[2]+1], hm)
        return ax
    end
    cr_τ = (-1, 1) .* maximum(abs, τxx_b)
    cr_P = (-1, 1) .* maximum(abs, P_b)
    grid_panel(  (1, 1), "τxx — grid (FD)",     τxx_c; crange=cr_τ)
    marker_panel((1, 3), "τxx — bilinear (3)",  τxx_b; crange=cr_τ)
    marker_panel((1, 5), "τxx — projected (4)", τxx_p; crange=cr_τ)
    marker_panel((1, 7), "τxx: (4) − (3)",      τxx_p .- τxx_b)
    grid_panel(  (2, 1), "P — grid (FD)",       P_c; crange=cr_P)
    marker_panel((2, 3), "P — bilinear (3)",    P_b; crange=cr_P)
    marker_panel((2, 5), "P — projected (4)",   P_p; crange=cr_P)
    marker_panel((2, 7), "P: (4) − (3)",        P_p .- P_b)

    # Residual magnitude at Vx nodes
    xVx, yVx = X.v.x[2:end-1], X.c.y
    lR(R)    = log10.(abs.(reshape(R, nc.x - 1, nc.y)) .+ 1e-16)
    for (col, R, title) in ((1, Rx_grid, "log₁₀|Rx| — grid (FD)"), (3, Rx_b, "log₁₀|Rx| — (3) markers → grid"), (5, Rx_p, "log₁₀|Rx| — (4) markers → grid"))
        ax = Axis(Fig[3, col], aspect=DataAspect(), title=title, xlabel="x", ylabel="y")
        hm = heatmap!(ax, xVx, yVx, lR(R), colormap=:viridis, colorrange=(-16, 2))
        Colorbar(Fig[3, col+1], hm)
    end

    # Profile across the inclusion along y ≈ 0
    ax = Axis(Fig[3, 7:8], title="σxx = τxx − P along |y| < Δy/4", xlabel="x", ylabel="σxx")
    jc   = argmin(abs.(X.c.y))
    strip = findall(abs.(ym .- X.c.y[jc]) .< Δ.y / 4)
    o    = strip[sortperm(xm[strip])]
    lines!(ax, X.c.x, τxx_c[:, jc] .- P_c[:, jc], color=:black, label="grid (centroids)")
    scatter!(ax, xm[o], τxx_b[o] .- P_b[o], markersize=5, color=:steelblue, label="(3) bilinear")
    scatter!(ax, xm[o], τxx_p[o] .- P_p[o], markersize=5, color=:orangered, label="(4) projected")
    vlines!(ax, [-rad, rad], color=:gray, linestyle=:dash)
    axislegend(ax, position=:rb)

    return Fig
end

@views function main(BC_template, D_template)
    #--------------------------------------------#

    # Resolution
    nc = (x = 150, y = 150)

    # Boundary loading type
    config = BC_template
    D_BC   = D_template

    # Materials initialization
    nphases = 2
    materials = initialize_materials(nphases; compressible=false)

    # Parameters
    params_bg = (ρ=1.0, n=1.0, η0=1e0, G=1e6)
    params_in = (ρ=1.0, n=1.0, η0=1e5, G=1e6)

    materials.g .= [0., 0.]
    materials.ρ .= [params_bg.ρ, params_in.ρ]
    materials.n .= [params_bg.n, params_in.n]
    materials.η0 .= [params_bg.η0, params_in.η0]
    materials.G .= [params_bg.G, params_in.G]

    preprocess!(materials)

    # Time steps
    Δt0   = 0.5
    nt    = 1

    # Solver parameters
    iter_params = IterParams(niter=2, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    # Intialise field
    L   = (x=1.0, y=1.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)
    x = (min=-L.x / 2, max=L.x / 2)
    y = (min=-L.y / 2, max=L.y / 2)

    # Allocate all fields and solver structures
    a = Allocs(nc, config, x, y, Δ, nphases)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Initial velocity & pressure field
    @views a.V.x .= D_BC[1,1]*a.X.vx_e.x .+ D_BC[1,2]*a.X.vx_e.y'
    @views a.V.y .= D_BC[2,1]*a.X.vy_e.x .+ D_BC[2,2]*a.X.vy_e.y'
    @views a.Pt[inx_c, iny_c]  .= 10.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (a.type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (a.type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (a.type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[1]  )
        BC.Vx[inx_Vx,  end-1] .= (a.type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[end])
        BC.Vy[inx_Vy,     2 ] .= (a.type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (a.type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (a.type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[1]   .+ D_BC[2,2]*a.X.v.y)
        BC.Vy[ end-1, iny_Vy] .= (a.type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[end] .+ D_BC[2,2]*a.X.v.y)
    end

    # Set material geometry
    rad = 0.1 + 1e-13
    @views a.phases.c[inx_c, iny_c][(a.X.c.x.^2 .+ (a.X.c.y').^2) .<= rad^2] .= 2
    @views a.phases.v[inx_v, iny_v][(a.X.v.x.^2 .+ (a.X.v.y').^2) .<= rad^2] .= 2
    Set_PhaseRatios!(a)

    #--------------------------------------------#

    rvec = zeros(length(iter_params.α))
    err  = (x = zeros(iter_params.niter), y = zeros(iter_params.niter), p = zeros(iter_params.niter))
    to   = TimerOutput()

    #--------------------------------------------#

    Fig = nothing
    for it=1:nt

        iter, err = main_loop(a, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)

        #--------------------------------------------#

        # Post-processing: grid -> marker stress interpolation, bilinear vs. equilibrium projection
        Fig = marker_stress_comparison(a, nc, Δ, L, rad; nmark=4, w_mixed=1.0)
        display(Fig)

    end

    display(to)

    return Fig
end


let
    # # Boundary condition templates
    BCs = [
        :free_slip,
    ]

    # Boundary velocity gradient matrix
    er    = -1
    # ∂𝐕∂𝐱 - velocity gradient tensor
    D_BCs = [
        @SMatrix( [er 0;        #    ∂Vx∂x ∂Vx∂y
                   0 -er] ),    #    ∂Vy∂x ∂Vy∂y  div(V) = 0 = ∂Vx∂x + ∂Vy∂y --> ∂Vy∂y = - ∂Vx∂x
    ]

    # Run them all
    for iBC in eachindex(BCs)
        @info "Running $(string(BCs[iBC])) and D = $(D_BCs[iBC])"
        main(BCs[iBC], D_BCs[iBC])
    end
end
