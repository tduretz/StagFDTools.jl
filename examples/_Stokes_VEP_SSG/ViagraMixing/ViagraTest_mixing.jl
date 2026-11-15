# Viagra test (elastic cantilever bent by gravity, then released) with the three phase mixing schemes of interface cells:
#   :default (each phase sees the cell strain rate, shared τ0), :voigt (iso-strain rate, τ0 per phase), :reuss (iso-stress)
# Same setup as ../Example_ViagraTest.jl. Gravity is switched off at t_off: an elastic plate should recover its initial shape.
# Recovery: tip = mean y of the plate markers with x > 750 km, minus its initial value (500 km).
# Markers also carry their initial position (x0, y0) for the rms displacement, but injected markers copy x0, y0 from a donor (approximate).
# Results: ViagraMixing_<scheme>_<nc>.jld2 (time series + marker snapshots), plotted by plot_ViagraMixing.jl
using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
using JustPIC, JLD2
using CellArraysIndexing: @index
const backend = JustPIC.CPU
using DifferentiationInterface
using TimerOutputs, GridGeometryUtils

const year2sec = 3.154e7

function set_geometry!(phases, colors, x0, y0, particles, plate, cell_size)
    Threads.@threads for j in axes(phases, 2)
        for i in axes(phases, 1)
            for ip in cellaxes(phases)
                @index(particles.index[ip, i, j]) == 0 && continue
                x = @index particles.coords[1][ip, i, j]
                y = @index particles.coords[2][ip, i, j]
                𝐱 = @SVector([x, y])
                cb = iseven(Int(fld(x, cell_size)) + Int(fld(y, cell_size))) ? 0. : 1.
                @index x0[ip, i, j] = x
                @index y0[ip, i, j] = y
                if inside(𝐱, plate)
                    @index phases[ip, i, j] = 1.
                    @index colors[ip, i, j] = 2. + cb
                else
                    @index phases[ip, i, j] = 2.
                    @index colors[ip, i, j] = cb
                end
            end
        end
    end
end

# Plate markers: positions, initial positions, colours
function plate_markers(particles, phases, colors, x0, y0)
    idx = particles.index.data[:]
    ph  = phases.data[:][idx]
    sel = ph .≈ 1.0
    return (x=particles.coords[1].data[:][idx][sel], y=particles.coords[2].data[:][idx][sel],
            x0=x0.data[:][idx][sel], y0=y0.data[:][idx][sel], c=colors.data[:][idx][sel])
end

function all_markers(particles, colors)
    idx = particles.index.data[:]
    return (x=particles.coords[1].data[:][idx], y=particles.coords[2].data[:][idx], c=colors.data[:][idx])
end

@views function main(mixing::Symbol; nc=(x=51, y=51), nt=200, t_off=9900.0, advection=:standard, subgrid=1.0, rotation=:upper_convected, stress_markers=true, Δt_yr=1e3, tag="", outdir=@__DIR__)

    D_BC = @SMatrix([1e-10 0.0; 0.0 -1e-10])
    config = :free_slip

    width = 1e6
    ρ_plate, ρ_matrix = 4000., 1.
    G_plate, G_matrix = 1e10, 1e20
    η_plate, η_matrix = 1e27, 1e21

    # Material parameters
    sc = (σ=1e10, t=1e10, L=1e5)
    nphases = 2
    materials = initialize_materials(nphases; compressible=false, mixing=mixing)
    materials.g  .= [0.0, -9.81] ./ (sc.L/sc.t^2)
    materials.ρ  .= [ρ_plate, ρ_matrix] ./ (sc.σ*sc.t^2/sc.L^2)
    materials.η0 .= [η_plate, η_matrix] ./ (sc.σ*sc.t)
    materials.G  .= [G_plate, G_matrix] ./ sc.σ
    preprocess!(materials)

    L = (x=width/sc.L, y=width/sc.L)
    Δt0 = Δt_yr*year2sec/sc.t
    iter_params = IterParams(solver_type=:PH, niter=10, ϵ_nl=1e-8, α=LinRange(0.05, 1.0, 10))

    Δ = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    x = (min=0.0, max=L.x)
    y = (min=0.0, max=L.y)
    a = Allocs(nc, config, x, y, Δ, nphases)
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    a.V.x .= D_BC[1, 1]*a.X.vx_e.x .+ D_BC[1, 2]*a.X.vx_e.y'
    a.V.y .= D_BC[2, 1]*a.X.vy_e.x .+ D_BC[2, 2]*a.X.vy_e.y'
    a.Pt[inx_c, iny_c] .= 0.0
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

    # Free-slip box, no-slip left wall
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    a.type.Vy[2, iny_Vy] .= :Dirichlet_tangent
    begin
        BC.Vx[2, iny_Vx] .= (a.type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[end-1, iny_Vx] .= (a.type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1]
        BC.Vx[inx_Vx, 2] .= (a.type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1]*a.X.v.x .+ D_BC[1, 2]*a.X.v.y[1])
        BC.Vx[inx_Vx, end-1] .= (a.type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1]*a.X.v.x .+ D_BC[1, 2]*a.X.v.y[end])
        BC.Vy[inx_Vy, 2] .= (a.type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[inx_Vy, end-1] .= (a.type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2]
        BC.Vy[2, iny_Vy] .= (a.type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* 0.0
        BC.Vy[end-1, iny_Vy] .= (a.type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1]*a.X.v.x[end] .+ D_BC[2, 2]*a.X.v.y)
    end

    # Markers: phase, colour, initial position (+ stress tensor)
    nxcell, max_xcell, min_xcell = (4, 4), 25, 10
    particles, tm = TensorMarkers(backend, a, nxcell, max_xcell, min_xcell)
    adv = Markers(backend, a, particles, tm, nc, nphases; args=4)
    phases, colors, x0, y0 = adv.particle_args

    n_layers = 12
    plate = Rectangle((4e5/sc.L, 5e5/sc.L), 8e5/sc.L, 6e5/sc.L)
    cell_size = plate.h / n_layers
    set_geometry!(phases, colors, x0, y0, adv.particles, plate, cell_size)
    Set_PhaseRatios!(a, adv.phase_ratios, adv.particles, phases)

    rvec = zeros(length(iter_params.α))
    err = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    to = TimerOutput()
    time_tot = 0.

    # Time series
    ts = (t=Float64[], tip=Float64[], ymean=Float64[], rms=Float64[], maxdisp=Float64[], τIImax=Float64[], niter=Int[])
    snaps = Dict{String,Any}("initial" => all_markers(adv.particles, colors))
    tip_x0 = 7.5e5/sc.L # tip: plate markers beyond this x
    gravity_on = true

    for it=1:nt
        if gravity_on && time_tot/year2sec*sc.t >= t_off
            materials.g .= [0., 0.]
            gravity_on = false
            snaps["release"] = all_markers(adv.particles, colors)
        end

        Vmax = max(maximum(abs.(a.V.x)), maximum(abs.(a.V.y)))
        if Vmax > 0
            Δ = (x=Δ.x, y=Δ.y, t=min(Δt0, 0.5*min(Δ.x, Δ.y)/Vmax))
        end
        time_tot += Δ.t
        stress_markers ? main_loop(a, adv, tm, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err; advection, subgrid, rotation) :
                         main_loop(a, adv, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err; advection) # grid-only stress history

        # Recovery diagnostics (plate markers, displacement from initial position) [km]
        pm = plate_markers(adv.particles, phases, colors, x0, y0)
        dx, dy = (pm.x .- pm.x0) .* sc.L/1e3, (pm.y .- pm.y0) .* sc.L/1e3
        d = sqrt.(dx.^2 .+ dy.^2)
        tip = pm.x .> tip_x0 # geometric: current position (x0, y0 of injected markers are copied from donors)
        push!(ts.t, time_tot*sc.t/year2sec)
        push!(ts.tip, sum(pm.y[tip])/max(count(tip), 1)*sc.L/1e3 - 500.0)
        push!(ts.ymean, sum(dy)/length(dy))
        push!(ts.rms, sqrt(sum(d.^2)/length(d)))
        push!(ts.maxdisp, maximum(d))
        push!(ts.τIImax, maximum(a.τ.II[inx_c, iny_c])*sc.σ)
        @printf("[%s] step %04d  t = %.0f yr  tip ȳ-500 = %.3f km  rms = %.3f km\n", mixing, it, ts.t[end], ts.tip[end], ts.rms[end])
    end
    snaps["final"] = all_markers(adv.particles, colors)

    file = joinpath(outdir, "ViagraMixing_$(mixing)_$(nc.x)" * (advection === :standard ? "" : "_$(advection)") * tag * ".jld2")
    jldsave(file; ts, snaps, mixing, nc, t_off, advection, sc_L=sc.L)
    display(to)
    @info "Saved $file"
    return ts
end

# Run all schemes (sequentially): julia --project -t 4 ViagraTest_mixing.jl [nt] [ncx] [advection] [schemes...]
# advection: standard | LinP | MQS
if abspath(PROGRAM_FILE) == @__FILE__
    nt  = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 200
    ncx = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 51
    advection = length(ARGS) > 2 ? Symbol(ARGS[3]) : :standard
    schemes = length(ARGS) > 3 ? Symbol.(ARGS[4:end]) : [:default, :voigt, :reuss]
    for mixing in schemes
        main(mixing; nc=(x=ncx, y=ncx), nt, advection)
    end
end
