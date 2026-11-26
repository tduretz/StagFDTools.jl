# Consistency check of the anisotropic (laminate) formulation: an isotropic matrix (mv = me = 1, n1 = n2 = n) computed
# with the anisotropic formulation (forced with materials.anisotropy.force) must give the same solution as the isotropic
# solver. Same set-up as Aniso_VE_inclusion.jl (stiff inclusion in a visco-elastic matrix under pure shear); both runs
# use mixing = :voigt and are advanced in lockstep, the fields are compared after each step.
#   n = 1: linear layers  (aniso_ViscoElastic)
#   n > 1: power-law layers (aniso_ViscoElastic_powerlaw)
using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics: mean
using TimerOutputs
import CairoMakie as cm

# Set-up of one run. formulation = :iso (isotropic solver) or :aniso (anisotropic formulation forced on the matrix)
@views function setup(formulation, n, θ, nc, Δt0, D_BC)
    config  = :free_slip
    nphases = 2
    materials = initialize_materials(nphases; compressible=false, mixing=:voigt,
                                     anisotropy = formulation === :aniso ? Anisotropy : Isotropic())

    # Matrix (phase 1) and stiff inclusion (phase 2)
    materials.g  .= [0., 0.]
    materials.ρ  .= [1.0, 1.0]
    materials.n  .= [n, 1.0]
    materials.η0 .= [1e1, 1e10]
    materials.G  .= [1e1, 1e1]

    # Isotropic laminate: equal layers (mv = me = 1, n1 = n2 = n), formulation forced on the matrix only
    if formulation === :aniso
        an = materials.anisotropy
        an.mv    .= [1., 1.]
        an.me    .= [1., 1.]
        an.α     .= [0.5, 0.5]
        an.n1    .= [n, 1.]
        an.n2    .= [n, 1.]
        an.dv    .= [-sind(θ), cosd(θ)]   # arbitrary: the result must not depend on it
    end
    preprocess!(materials)

    L = (x=1.0, y=1.0)
    Δ = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    x = (min=-L.x/2, max=L.x/2)
    y = (min=-L.y/2, max=L.y/2)
    a = Allocs(nc, config, x, y, Δ, nphases)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Initial velocity & pressure field
    a.V.x .= D_BC[1,1]*a.X.vx_e.x .+ D_BC[1,2]*a.X.vx_e.y'
    a.V.y .= D_BC[2,1]*a.X.vy_e.x .+ D_BC[2,2]*a.X.vy_e.y'
    a.Pt[inx_c, iny_c] .= 10.
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Boundary condition values
    BC = (Vx = zeros(size_x...), Vy = zeros(size_y...))
    BC.Vx[     2, iny_Vx] .= (a.type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (a.type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (a.type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[1]  )
    BC.Vx[inx_Vx,  end-1] .= (a.type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*a.X.v.x .+ D_BC[1,2]*a.X.v.y[end])
    BC.Vy[inx_Vy,     2 ] .= (a.type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (a.type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (a.type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[1]   .+ D_BC[2,2]*a.X.v.y)
    BC.Vy[ end-1, iny_Vy] .= (a.type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*a.X.v.x[end] .+ D_BC[2,2]*a.X.v.y)

    # Material geometry: circular inclusion
    rad = 0.1 + 1e-13
    a.phases.c[inx_c, iny_c][(a.X.c.x.^2 .+ (a.X.c.y').^2) .<= rad^2] .= 2
    a.phases.v[inx_v, iny_v][(a.X.v.x.^2 .+ (a.X.v.y').^2) .<= rad^2] .= 2
    Set_PhaseRatios!(a)

    iter_params = IterParams(niter=20)
    rvec = zeros(length(iter_params.α))
    err  = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    return (; a, materials, BC, nc, Δ, nphases, iter_params, rvec, err, to=TimerOutput(), inx_c, iny_c)
end

step!(r, it) = main_loop(r.a, it, r.materials, r.BC, r.nc, r.Δ, r.to, r.nphases, r.iter_params, r.rvec, r.err)

# max |aniso - iso| / max |iso| over the inner centroids (all cells, matrix cells)
function reldiff(fa, fi, I, m)
    d, s = abs.(fa[I...] .- fi[I...]), maximum(abs.(fi[I...]))
    return maximum(d) / s, maximum(d[m]) / s
end

function compare(; n=1.0, θ=30.0, nc=(x=50, y=50), Δt0=0.1, nt=50, er=-1.0)
    D_BC = @SMatrix([er 0; 0 -er])
    iso   = setup(:iso,   n, θ, nc, Δt0, D_BC)
    aniso = setup(:aniso, n, θ, nc, Δt0, D_BC)
    I = (iso.inx_c, iso.iny_c)
    m = iso.a.phases.c[I...] .== 1
    hist = (t=Float64[], τII=Float64[], τII_m=Float64[], ε̇II=Float64[], P=Float64[])
    for it in 1:nt
        step!(iso, it)
        step!(aniso, it)
        τ_all, τ_m = reldiff(aniso.a.τ.II, iso.a.τ.II, I, m)
        ε̇_all, _   = reldiff(aniso.a.ε̇.II, iso.a.ε̇.II, I, m)
        P_all, _   = reldiff(aniso.a.Pt, iso.a.Pt, I, m)
        push!(hist.t, it*Δt0); push!(hist.τII, τ_all); push!(hist.τII_m, τ_m); push!(hist.ε̇II, ε̇_all); push!(hist.P, P_all)
    end

    @printf("\n=== n = %g, director %g°: max |aniso - iso| / max |iso| ===\n", n, θ)
    @printf("%8s %12s %12s %12s %12s\n", "t", "τII", "τII matrix", "ε̇II", "P")
    for k in unique([1:max(1, nt ÷ 10):nt; nt])
        @printf("%8.2f %12.2e %12.2e %12.2e %12.2e\n", hist.t[k], hist.τII[k], hist.τII_m[k], hist.ε̇II[k], hist.P[k])
    end
    @printf("mean τII matrix at t = %g: iso %.6f, aniso %.6f\n", nt*Δt0, mean(iso.a.τ.II[I...][m]), mean(aniso.a.τ.II[I...][m]))

    # Fields at the last step and differences in time
    xc, yc = iso.a.X.c.x, iso.a.X.c.y
    τi, τa = iso.a.τ.II[I...], aniso.a.τ.II[I...]
    fig = cm.Figure(size=(800, 800))
    cm.Label(fig[0, 1:4], @sprintf("n = %g, director %g°, t = %g", n, θ, nt*Δt0), fontsize=18)
    lims = extrema(vcat(τi[m], τa[m]))
    for (k, (F, lab)) in enumerate(((τi, "τII isotropic solver"), (τa, "τII anisotropic formulation")))
        ax = cm.Axis(fig[1, 2k-1], aspect=cm.DataAspect(), title=lab)
        hm = cm.heatmap!(ax, xc, yc, F, colormap=:vik, colorrange=lims)
        cm.Colorbar(fig[1, 2k], hm)
    end
    ax = cm.Axis(fig[2, 1], aspect=cm.DataAspect(), title="log10 |ΔτII| / max τII")
    hm = cm.heatmap!(ax, xc, yc, log10.(abs.(τa .- τi) ./ maximum(τi) .+ 1e-16), colormap=:viridis)
    cm.Colorbar(fig[2, 2], hm)
    ax = cm.Axis(fig[2, 3], yscale=log10, xlabel="t", title="max relative difference")
    cm.lines!(ax, hist.t, max.(hist.τII, 1e-16), label="τII")
    cm.lines!(ax, hist.t, max.(hist.ε̇II, 1e-16), label="ε̇II")
    cm.lines!(ax, hist.t, max.(hist.P, 1e-16), label="P")
    cm.axislegend(ax, position=:rb)
    display(fig)
    return (; iso, aniso, hist, fig)
end

# Linear and power-law layers
let
    compare(n=1.0)
    compare(n=3.0)
    nothing
end
