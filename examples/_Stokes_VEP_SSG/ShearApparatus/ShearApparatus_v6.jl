using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, GridGeometryUtils
import Statistics: mean
using DifferentiationInterface
using TimerOutputs, CairoMakie
using JustPIC
using CellArraysIndexing: @index
const backend = JustPIC.CPU

# comment rand() in initial pressure
# make large VP
# made :free_slip

function set_material_geometry!(phases, particles, gouge, salt, plate)
    Threads.@threads for j in axes(phases, 2)
        for i in axes(phases, 1)
            for ip in cellaxes(phases)
                @index(particles.index[ip, i, j]) == 0 && continue
                x = @index particles.coords[1][ip, i, j]
                y = @index particles.coords[2][ip, i, j]
                𝐱 = @SVector([x, y])
                if any(g -> inside(𝐱, g), salt)
                    @index phases[ip, i, j] = 3.
                elseif any(g -> inside(𝐱, g), plate)
                    @index phases[ip, i, j] = 4.
                elseif any(g -> inside(𝐱, g), gouge)
                    @index phases[ip, i, j] = 2.
                else
                    @index phases[ip, i, j] = 1.
                end
            end
        end
    end
end

function visualise_markers(particles, phases, sc, x, y, time)
    with_theme(theme_latexfonts()) do
    ppx, ppy = particles.coords
    idxv = particles.index.data[:]
    px   = Array(ppx.data[:][idxv]) .* sc.L
    py   = Array(ppy.data[:][idxv]) .* sc.L
    ph   = Int.(Array(phases.data[:][idxv]))
    phase_colors = [:blue, :red, :lightblue, :gray80] # rock, gouge, salt, plate
    Lx, Ly = x.max - x.min, y.max - y.min
    fig = Figure(size=(500, 500*Ly/Lx))
    ax  = Axis(fig[1, 1], aspect=DataAspect(), title=@sprintf("t = %.1f min", time*sc.t/60), xlabel="x [m]", ylabel="y [m]")
    scatter!(ax, px, py, color=phase_colors[ph], markersize=3)
    limits!(ax, x.min*sc.L, x.max*sc.L, y.min*sc.L, y.max*sc.L)
    Legend(fig[1, 2], [MarkerElement(color=c, marker=:circle, markersize=12) for c in phase_colors], ["rock", "gouge", "salt", "plate"], "Phase")
    resize_to_layout!(fig)
    display(fig)
    end
    return nothing
end

function softening(Ep, Cmax, Cmin, E0, w)
    return Cmin + (Cmax-Cmin)*(1 - tanh((Ep-E0)/w))
end

@views function main(nc, nt, θgouge)
    #--------------------------------------------#

    # Scaling
    sc = (σ=1e9, L=1e-3, t=1e6)

    # Parameters
    width = (12e-3 + 38e-3)/sc.L
    height = 100e-3/sc.L
    thickness = 2*2e-3/sc.L
    L = (x=width, y=height)
    θgouge = (θgouge) / 180*π
    Δt0 = 1e2/sc.t
    ε̇yy = 1.31e-6*sc.t
    Pbg = 1e7/sc.σ

    𝐱_probe_salt = (0.023/sc.L, height/2) # apparatus response (plate and salt)
    𝐱_probe_plate = (0.000/sc.L, 0.002/sc.L) # apparatus response (plate and salt)
    𝐱_probe_MC1   = (0.003/sc.L, height/2) # sandstone
    # 𝐱_probe_MC1 = (0.006/sc.L, height/1.5) # sandstone
    𝐱_probe_MC2 = (0.001/sc.L, height/2) # gouge

    # Boundary loading type
    config = :EW_stress
    # config = :free_slip
    D_BC = @SMatrix([ε̇yy 0.;
        0 -ε̇yy])
    σ_BC = @SMatrix([-Pbg 0.;
        0 -Pbg])

      # Material parameters
    nphases                   = 4
    materials                 = initialize_materials(nphases; compressible=true, plasticity=DruckerPrager)  #Dzz=-0.0*ε̇yy
    materials.n              .= [1.0, 1.0, 1.0, 1.0]                                                        # Power law exponent
    materials.η0             .= [1e48, 1e28, 1e10, 1e48] ./ sc.σ ./ sc.t                                    # Reference viscosity 
    materials.G              .= [1e10, 1e9, 1e60, 1e10] ./ sc.σ                                             # Shear modulus
    materials.β              .= [1e-11, 1e-10, 1e-12, 1e-12] .* sc.σ                                        # Compressibility
    materials.plasticity.C   .= [45e6, 0.67e6, 15e60, 15e60] ./ sc.σ                                        # Cohesion
    materials.plasticity.ϕ   .= [31., 30.5, 35., 35.]                                                       # Friction angle
    materials.plasticity.ψ   .= [0.0, 0.0, 0.0, 0.0]                                                        # Dilation angle
    materials.plasticity.ηvp .= [1e10, 1e9, 1e10, 1e10] .* 5/sc.σ ./ sc.t                                   # Viscoplastic regularisation
      # materials.plasticity.soft .= [0.0, 0.0, 0.0, 0.0]             # softening ?
    #                            rock     gouge     salt   plates
    preprocess!(materials)

    @show π*180/π/4 + materials.plasticity.ϕ[2]/2
    @show π*180/π/4 + (materials.plasticity.ϕ[2] + materials.plasticity.ψ[2])/4

    # Solver parameters
    # Pic2Newt = 1e10   # more than 1.0 - always Newton
    iter_params = IterParams(ϵ_nl=1e-9, solver_type=:GCR)

    # Discretisation
    Δ = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    x = (min=(-L.x/2), max=L.x/2)
    y = (min=0.0, max=L.y)

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    # Allocations
    a = Allocs(nc, config, x, y, Δ, nphases)
    # additional Allocations
    εp = (c=zeros(size_c...), v=zeros(size_v...))
    sp = (c=ones(size_c...), v=ones(size_v...))
    C = (c=zeros(size_c...), v=zeros(size_v...))

    # Initial velocity & pressure field
    @views a.V.x .= D_BC[1, 1]*a.X.vx_e.x .+ D_BC[1, 2]*a.X.vx_e.y'
    @views a.V.y .= D_BC[2, 1]*a.X.vy_e.x .+ D_BC[2, 2]*a.X.vy_e.y'
    UpdateSolution!(a.V, a.Pt, a.dx, a.number, a.type, nc)

      # Boundary condition values
    BC = (Vx=zeros(size_x...), Vy=zeros(size_y...))
    @views begin
        BC.Vx[2, iny_Vx]     .= (a.type.Vx[1, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1] .+ (a.type.Vx[1, iny_Vx] .== :normal_stress) .* σ_BC[1, 1]
        BC.Vx[end-1, iny_Vx] .= (a.type.Vx[end, iny_Vx] .== :Neumann_normal) .* D_BC[1, 1] .+ (a.type.Vx[end, iny_Vx] .== :normal_stress) .* σ_BC[1, 1]
        BC.Vx[inx_Vx, 2]     .= (a.type.Vx[inx_Vx, 2] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, 2] .== :Dirichlet_tangent) .* (D_BC[1, 1]*a.X.v.x .+ D_BC[1, 2]*a.X.v.y[1])
        BC.Vx[inx_Vx, end-1] .= (a.type.Vx[inx_Vx, end-1] .== :Neumann_tangent) .* D_BC[1, 2] .+ (a.type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1, 1]*a.X.v.x .+ D_BC[1, 2]*a.X.v.y[end])
        BC.Vy[inx_Vy, 2]     .= (a.type.Vy[inx_Vy, 1] .== :Neumann_normal) .* D_BC[2, 2] .+ (a.type.Vy[inx_Vy, 1] .== :normal_stress) .* σ_BC[2, 2]
        BC.Vy[inx_Vy, end-1] .= (a.type.Vy[inx_Vy, end] .== :Neumann_normal) .* D_BC[2, 2] .+ (a.type.Vy[inx_Vy, end] .== :normal_stress) .* σ_BC[2, 2]
        BC.Vy[2, iny_Vy]     .= (a.type.Vy[2, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1]*a.X.v.x[1] .+ D_BC[2, 2]*a.X.v.y)
        BC.Vy[end-1, iny_Vy] .= (a.type.Vy[end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2, 1] .+ (a.type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2, 1]*a.X.v.x[end] .+ D_BC[2, 2]*a.X.v.y)
    end

    # Markers
    nxcell = (4, 4)
    max_xcell = 25
    min_xcell = 10
    args = 1
    particles, tm = TensorMarkers(backend, a, nxcell, max_xcell, min_xcell)
    adv = Markers(backend, a, particles, tm, nc, nphases; args)
    phases, = adv.particle_args

    # Material geometry
    gouge = (
        Rectangle((0.0/sc.L, L.y/2), thickness, 20.0e-2/sc.L; θ=θgouge),
    )
    salt = (
        Rectangle((-L.x/2, L.y/2), 6e-3/sc.L, L.y; θ=0),
        Rectangle((L.x/2, L.y/2), 6e-3/sc.L, L.y; θ=0),
    )
    plate = (
        Rectangle((0.0/sc.L, 2e-3/sc.L/2), 1.1e-1/sc.L, 7e-3/sc.L; θ=0),
        Rectangle((0.0/sc.L, L.y-2e-3/sc.L/2), 1.1e-1/sc.L, 7.0e-3/sc.L; θ=0),
    )
    # seed = (
    #     Ellipse((0.0, 0.0), 0.1/sc.L, 0.1/sc.L; θ = 0),
    # )

    set_material_geometry!(phases, particles, gouge, salt, plate)

    # Set phase ratios on the grid
    Set_PhaseRatios!(a, adv.phase_ratios, adv.particles, phases)
    Set_Grid_Phases!(a)
    visualise_markers(particles,phases,sc,x,y,0)

    # check 
    for I in CartesianIndices(a.phase_ratios.c)
        s = sum(a.phase_ratios.c[I])
        if !(s ≈ 1.0)
            @warn "Invalid phase_ratios.c at $I: sum = $s, values = $(a.phase_ratios.c[I])"
        end
    end

    ΔCc = materials.plasticity.C[1]/5 .* rand(size(C.c)...)
    ΔCv = materials.plasticity.C[1]/5 .* rand(size(C.v)...)

    a.Pt .= Pbg #*rand(size(Pt)...)
    a.Pt0 .= a.Pt
    a.Pti .= a.Pt

    #--------------------------------------------#

    rvec        = zeros(length(iter_params.α))
    err         = (x=zeros(iter_params.niter), y=zeros(iter_params.niter), p=zeros(iter_params.niter))
    probes      = (τII=zeros(nt), τIIW=zeros(nt), τIIE=zeros(nt), τIIS=zeros(nt), τIIN=zeros(nt), true_fric=zeros(nt), app_fric=zeros(nt), app_fric_lab=zeros(nt), t=zeros(nt), εxx=zeros(nt), εyy=zeros(nt), σyyN=zeros(nt), σyyS=zeros(nt), σxxW=zeros(nt), σxxE=zeros(nt), PW=zeros(nt), PE=zeros(nt), minθ=zeros(nt), maxθ=zeros(nt), meanθ=zeros(nt))
    probe_salt  = (σxx=zeros(nt), σyy=zeros(nt), σzz=zeros(nt), σxy=zeros(nt), τII=zeros(nt), P=zeros(nt))
    probe_plate = (σxx=zeros(nt), σyy=zeros(nt), σzz=zeros(nt), σxy=zeros(nt), τII=zeros(nt), P=zeros(nt))
    probe_MC1   = (σxx=zeros(nt), σyy=zeros(nt), σzz=zeros(nt), σxy=zeros(nt), τII=zeros(nt), P=zeros(nt))
    probe_MC2   = (σxx=zeros(nt), σyy=zeros(nt), σzz=zeros(nt), σxy=zeros(nt), τII=zeros(nt), P=zeros(nt))
    to          = TimerOutput()

    # Swap old values 
    a.τ0.xx .= a.τ.xx
    a.τ0.yy .= a.τ.yy
    a.τ0.xy .= a.τ.xy
    a.Pt0 .= a.Pt
    #--------------------------------------------#

    time = 0
    for it=1:nt

        # Adaptive time step
        Vmax = max(maximum(abs.(a.V.x)), maximum(abs.(a.V.y)))
        if Vmax > 0
            Δ = (x=Δ.x, y=Δ.y, t=min(Δt0, 0.5*min(Δ.x, Δ.y)/Vmax))
        end
        time += Δ.t

        # Main loop: solver and advection -------------------------------------------------------------------
        @time main_loop(a, adv, tm, it, materials, BC, nc, Δ, to, nphases, iter_params, rvec, err)
        Set_Grid_Phases!(a)

        # Update platic strain
        εp.c .+= Δ.t .* a.λ̇.c
        εp.v .+= Δ.t .* a.λ̇.v

        # Softening
        sp.c .= softening.(εp.c, 1.0, 0.2, 0.0, 0.1)
        sp.v .= softening.(εp.v, 1.0, 0.2, 0.0, 0.1)
        @show extrema(sp.c)
        @show extrema(sp.v)

        #--------------------------------------------#

        # Post process stress and strain rate
        τxyc = av2D(a.τ.xy)

        isrock  = zeros(Bool, size_c)
        isgouge = zeros(Bool, size_c)
        [isrock[I] = a.phase_ratios.c[I][1]≈1.0 for I in eachindex(a.phase_ratios.c)]
        [isgouge[I] = a.phase_ratios.c[I][2]≈1.0 for I in eachindex(a.phase_ratios.c)]

        τII_rock = a.τ.II[isrock]
        P_rock   = a.Pt[isrock]
        λ̇_rock  = a.λ̇.c[isrock]

        τII_gouge = a.τ.II[isgouge]
        P_gouge   = a.Pt[isgouge]
        λ̇_gouge  = a.λ̇.c[isgouge]

        # Compute apparent and true friction locally
        app_fric = zeros(size(a.Pt))
        true_fric = zeros(size(a.Pt))
        Rot = @SMatrix[cos(π/2 - θgouge) sin(π/2 - θgouge) 0; -sin(π/2 - θgouge) cos(π/2 - θgouge) 0; 0 0 1.0]
        app_fric_sum = 0.0
        true_fric_sum = 0.0
        app_sum = 0
        true_sum = 0
        ind_mid_x = Int64(floor((nc.x+2)/2))
        ind_mid_y = Int64(floor((nc.y+2)/2))

        for i in inx_c, j in iny_c
            if a.phases.c[i, j] == 2 # λ̇.c[i,j] > 1e-10
                app_sum += 1
                σ = @SMatrix[-a.Pt[i, j]+a.τ.xx[i, j] τxyc[i, j] 0.; τxyc[i, j] -a.Pt[i, j]+a.τ.yy[i, j] 0.; 0. 0. -a.Pt[i, j]+(-a.τ.xx[i, j]-a.τ.yy[i, j])]
                # Compute apparent friction
                σ′ = Rot * σ * Rot'
                app_fric[i, j] = σ′[1, 2] / σ′[2, 2]
                app_fric_sum += app_fric[i, j]
                # Compute true friction
                if a.λ̇.c[i, j] > abs(ε̇yy) && i==ind_mid_x && j==ind_mid_y
                    true_sum += 1
                    ph = a.phases.c[i, j]
                    cxcosϕ = materials.plasticity.C[ph] * materials.plasticity.cosϕ[ph]
                    ηvp = materials.plasticity.ηvp[ph]
                    true_fric[i, j] = tand(asind(1/a.Pt[i, j] * (a.τ.II[i, j] - cxcosϕ - ηvp*a.λ̇.c[i, j])))
                    true_fric_sum += true_fric[i, j]
                end
            end
        end

        # Store probes data
        𝐈_probe_salt  = argmin(abs.(a.X.c_e.x .- 𝐱_probe_salt[1])),  argmin(abs.(a.X.c_e.y .- 𝐱_probe_salt[2])) # apparatus response (plate and salt)
        𝐈_probe_plate = argmin(abs.(a.X.c_e.x .- 𝐱_probe_plate[1])), argmin(abs.(a.X.c_e.y .- 𝐱_probe_plate[2]))# apparatus response (plate and salt)
        𝐈_probe_MC1   = (argmin(abs.(a.X.c_e.x .- 𝐱_probe_MC1[1]))), argmin(abs.(a.X.c_e.y .- 𝐱_probe_MC1[2]))
        𝐈_probe_MC2   = (argmin(abs.(a.X.c_e.x .- 𝐱_probe_MC2[1]))), argmin(abs.(a.X.c_e.y .- 𝐱_probe_MC2[2]))

        σyy             = a.τ.yy .- a.Pt
        σxx             = a.τ.xx .- a.Pt
        σzz             = a.τ.yy .- a.Pt
        ind_mid_x       = Int64(floor(nc.x/2))
        ind_mid_y       = Int64(floor(nc.y/2))
        probes.t[it]    = time
        probes.τII[it]  = mean(a.τ.II)
        probes.τIIW[it] = a.τ.II[2, ind_mid_y]
        probes.τIIE[it] = a.τ.II[end-1, ind_mid_y]
        probes.τIIS[it] = a.τ.II[ind_mid_x, 2]
        probes.τIIN[it] = a.τ.II[ind_mid_x, end-1]
        probes.PW[it]   = a.Pt[2, ind_mid_y]
        probes.PE[it]   = a.Pt[end-1, ind_mid_y]
        probes.σxxW[it] = a.τ.xx[2, ind_mid_y] - a.Pt[2, ind_mid_y]
        probes.σxxE[it] = a.τ.xx[end-1, ind_mid_y] - a.Pt[end-1, ind_mid_y]
        probes.σyyS[it] = mean(a.τ.yy[inx_c, 2] .- a.Pt[inx_c, 2])
        probes.σyyN[it] = mean(a.τ.yy[inx_c, end-1] .- a.Pt[inx_c, end-1])

        probe_salt.σxx[it] = σxx[𝐈_probe_salt...]
        probe_salt.σyy[it] = σyy[𝐈_probe_salt...]
        probe_salt.σzz[it] = σzz[𝐈_probe_salt...]
        probe_salt.σxy[it] = a.τ.xy[𝐈_probe_salt...]
        probe_salt.P[it]   = a.Pt[𝐈_probe_salt...]
        probe_salt.τII[it] = a.τ.II[𝐈_probe_salt...]

        probe_plate.σxx[it] = σxx[𝐈_probe_plate...]
        probe_plate.σyy[it] = σyy[𝐈_probe_plate...]
        probe_plate.σzz[it] = σzz[𝐈_probe_plate...]
        probe_plate.σxy[it] = a.τ.xy[𝐈_probe_plate...]
        probe_plate.P[it]   = a.Pt[𝐈_probe_plate...]
        probe_plate.τII[it] = a.τ.II[𝐈_probe_plate...]

        probe_MC1.σxx[it] = σxx[𝐈_probe_MC1...]
        probe_MC1.σyy[it] = σyy[𝐈_probe_MC1...]
        probe_MC1.σzz[it] = σzz[𝐈_probe_MC1...]
        probe_MC1.σxy[it] = a.τ.xy[𝐈_probe_MC1...]
        probe_MC1.P[it]   = a.Pt[𝐈_probe_MC1...]
        probe_MC1.τII[it] = a.τ.II[𝐈_probe_MC1...]

        probe_MC2.σxx[it] = σxx[𝐈_probe_MC2...]
        probe_MC2.σyy[it] = σyy[𝐈_probe_MC2...]
        probe_MC2.σzz[it] = σzz[𝐈_probe_MC2...]
        probe_MC2.σxy[it] = a.τ.xy[𝐈_probe_MC2...]
        probe_MC2.P[it]   = a.Pt[𝐈_probe_MC2...]
        probe_MC2.τII[it] = a.τ.II[𝐈_probe_MC2...]

        # σd = probes.σyyS[it] - probes.σxxW[it]
        # σm = -probes.PW[it]
        σd = 2 * 1/2*(mean(τII_gouge))
        σm = 1 * 1/2*(mean(P_gouge))
        τr = σd/2 * sin(2*(π/2 - θgouge))
        τn = σm + σd/2 * cos(2*(π/2 - θgouge))
        # probes.app_fric[it] =  τr / τn
        probes.app_fric[it] = app_fric_sum / app_sum
        probes.true_fric[it] = true_fric_sum / true_sum

        # σ  = @SMatrix[probes.σxxW[it] 0.0 0.; 0.0 probes.σyyS[it] 0.; 0. 0. 1/2*(probes.σxxW[it] + probes.σyyS[it])]
        σ = @SMatrix[probe_salt.σxx[it] 0.0 0.;
            0. probe_plate.σyy[it] 0.;
            0. 0. -probe_salt.P[it]]

        σ′ = Rot * σ * Rot'
        probes.app_fric_lab[it] = σ′[1, 2] / σ′[2, 2]

        # Stress angles
        i1 = argmin(abs.(a.X.c_e.x .- (-0.01/sc.L)))
        i2 = argmin(abs.(a.X.c_e.x .- (0.01/sc.L)))
        j1 = argmin(abs.(a.X.c_e.y .- (0.025/sc.L)))
        j2 = argmin(abs.(a.X.c_e.y .- (0.075/sc.L)))
        θawayBC = a.τ.θ[i1:i2, j1:j2]
        probes.minθ[it] = minimum(θawayBC[a.phases.c[i1:i2, j1:j2] .== 2])
        probes.meanθ[it] = mean(θawayBC[a.phases.c[i1:i2, j1:j2] .== 2])
        probes.maxθ[it] = maximum(θawayBC[a.phases.c[i1:i2, j1:j2] .== 2])

        # Visualise
        function figure()
            ftsz = 25
            fig = Figure(size=(1000, 1000))
            empty!(fig)

            # Split heatmap of the apparatus
            # ax  = Axis(fig[1:3,1], aspect=DataAspect(), title="Apparent / True friction", xlabel="x", ylabel="y", xlabelsize=ftsz,  ylabelsize=ftsz, titlesize=ftsz)
            ax = Axis(fig[1:3, 1], aspect=DataAspect(), title="Plastic strain rate", xlabel="x", ylabel="y", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            eps = 1e-1
            # field = (τ.xy)[inx_c,iny_c]  .* sc.σ / 1e6
            # field = app_fric[inx_c,iny_c]
            # field = (τ.yy .- Pt)[inx_c,iny_c]  .* sc.σ / 1e6
            field = log10.((a.λ̇.c[inx_c, iny_c] .+ eps)/sc.t)
            crange = (minimum(field)-eps, maximum(field)+eps)

            # field = C.c*sc.σ./1e6
            # crange = (5, 7)

            field1 = app_fric .+ eps
            field2 = true_fric .+ eps
            # field = phases.c[inx_c,iny_c]
            hm = heatmap!(ax, a.X.c.x .* sc.L, a.X.c.y .* sc.L, field, colormap=:bluesreds, colorrange=crange)
            # hm1 = heatmap!(ax, a.X.c.x[1:ind_mid_x].*sc.L, a.X.c.y.*sc.L, field1[1:ind_mid_x,:], colormap=:bluesreds, colorrange=(minimum(field1)-eps, maximum(field1)+eps))
            # hm2 = heatmap!(ax, a.X.c.x[ind_mid_x:end].*sc.L, a.X.c.y.*sc.L, field2[ind_mid_x:end,:], colormap=:bluesreds, colorrange=(minimum(field2)-eps, maximum(field2)+eps))
            contour!(ax, a.X.c.x .* sc.L, a.X.c.y .* sc.L, a.phases.c[inx_c, iny_c], color=:white)
            # Colorbar(fig[4, 1], hm1, label = L"$\phi^\text{app}$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
            # Colorbar(fig[4, 2], hm2, label = L"$\phi^\text{true}$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
            Vxc = (0.5*(a.V.x[1:(end-1), 2:(end-1)]+a.V.x[2:end, 2:(end-1)]))[2:(end-1), 2:(end-1)] .* sc.L/sc.t
            Vyc = (0.5*(a.V.y[2:(end-1), 1:(end-1)]+a.V.y[2:(end-1), 2:end]))[2:(end-1), 2:(end-1)] .* sc.L/sc.t
            step = 10
            # arrows2d!(ax, a.X.c.x[1:step:end].*sc.L, a.X.c.y[1:step:end].*sc.L, Vxc[1:step:end,1:step:end], Vyc[1:step:end,1:step:end], lengthscale=50000.4, color=:white)
            arrows2d!(ax, a.X.c.x[1:step:end] .* sc.L, a.X.c.y[1:step:end] .* sc.L, cos.(a.τ.θ)[inx_c, iny_c][1:step:end, 1:step:end], sin.(a.τ.θ)[inx_c, iny_c][1:step:end, 1:step:end], lengthscale=0.001, color=:white, tiplength=0)
            xlims!(ax, minimum(a.X.v.x) .* sc.L, maximum(a.X.v.x) .* sc.L)
            lines!(ax, a.X.c.x[ind_mid_x] .* sc.L * ones(size(a.X.c.y)), a.X.c.y .* sc.L, color=:white, linewidth=4)
            # Probes
            scatter!(ax, a.X.c.x[𝐈_probe_salt[1]] .* sc.L, a.X.c.y[𝐈_probe_salt[2]] .* sc.L, marker=:star5, markersize=20)
            scatter!(ax, a.X.c.x[𝐈_probe_plate[1]] .* sc.L, a.X.c.y[𝐈_probe_plate[2]] .* sc.L, marker=:star5, markersize=20)
            # Subdomain
            scatter!(ax, a.X.c.x[i1] .* sc.L, a.X.c.y[j1] .* sc.L, marker=:cross, color=:gray, markersize=20)
            scatter!(ax, a.X.c.x[i2] .* sc.L, a.X.c.y[j1] .* sc.L, marker=:cross, color=:gray, markersize=20)
            scatter!(ax, a.X.c.x[i1] .* sc.L, a.X.c.y[j2] .* sc.L, marker=:cross, color=:gray, markersize=20)
            scatter!(ax, a.X.c.x[i2] .* sc.L, a.X.c.y[j2] .* sc.L, marker=:cross, color=:gray, markersize=20)
            # Mohr-circles
            scatter!(ax, a.X.c.x[𝐈_probe_MC1[1]] .* sc.L, a.X.c.y[𝐈_probe_MC1[2]] .* sc.L, marker=:xcross, color=:white, markersize=20)
            scatter!(ax, a.X.c.x[𝐈_probe_MC2[1]] .* sc.L, a.X.c.y[𝐈_probe_MC2[2]] .* sc.L, marker=:xcross, color=:white, markersize=20)

            # # Zoom on the gouge
            # ax  = Axis(fig[2,2], aspect=DataAspect(), title="Plastic Strain rate", xlabel="x", ylabel="y", xlabelsize=ftsz,  ylabelsize=ftsz, titlesize=ftsz)
            # eps   = 1e-1
            # # field = (a.τ.xy)[inx_c,iny_c]  .* sc.σ / 1e6
            # # field = app_fric[inx_c,iny_c]
            # # field = (a.τ.yy .- a.Pt)[inx_c,iny_c]  .* sc.σ / 1e6
            # field = log10.((a.λ̇.c[inx_c,iny_c] .+ eps)/sc.t )
            # # field = a.phases.c[inx_c,iny_c]
            # hm = heatmap!(ax, a.X.c.x.*sc.L, a.X.c.y.*sc.L, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            # contour!(ax, a.X.c.x.*sc.L, a.X.c.y.*sc.L,  a.phases.c[inx_c,iny_c], color=:white)
            # # arrows2d!(ax, a.X.c.x[1:step:end].*sc.L, a.X.c.y[1:step:end].*sc.L, Vxc[1:step:end,1:step:end], Vyc[1:step:end,1:step:end], lengthscale=50000.4, color=:white)
            # arrows2d!(ax, a.X.c.x[1:step:end].*sc.L, a.X.c.y[1:step:end].*sc.L, cos.(τ.θ)[inx_c,iny_c][1:step:end,1:step:end], sin.(τ.θ)[inx_c,iny_c][1:step:end,1:step:end], lengthscale=0.001, color=:white, tiplength = 0)
            # xlims!(ax, -0.02, 0.02)
            # ylims!(ax, 0.04, 0.06)

            ax = Axis(fig[2, 2], aspect=DataAspect(), title="Mohr circles", xlabel="σn", ylabel="τ", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            θ = LinRange(0, π, 100)

            MC1 = (x=(probe_MC1.τII[it] .* cos.(θ) .+ probe_MC1.P[it]), y=(probe_MC1.τII[it] .* sin.(θ)))
            MC2 = (x=(probe_MC2.τII[it] .* cos.(θ) .+ probe_MC2.P[it]), y=(probe_MC2.τII[it] .* sin.(θ)))

            τII_MC3 = -(probe_plate.σyy[it] - probe_salt.σxx[it])/2
            P_MC3 = -(probe_plate.σyy[it] + probe_salt.σxx[it])/2
            MC3 = (x=(τII_MC3 .* cos.(θ) .+ P_MC3), y=(τII_MC3 .* sin.(θ)))
            MC3_μ = (x=(τII_MC3 .* (cos.(2*(π/2-θgouge))) .+ P_MC3), y=(τII_MC3 .* sin.(2*(π/2-θgouge))))

            lines!(ax, MC1.x*sc.σ ./ 1e6, MC1.y*sc.σ ./ 1e6, label="probe 1")
            lines!(ax, MC2.x*sc.σ ./ 1e6, MC2.y*sc.σ ./ 1e6, label="probe 2")
            lines!(ax, MC3.x*sc.σ ./ 1e6, MC3.y*sc.σ ./ 1e6, label="apparatus")
            σmin = minimum([minimum(MC1.x), minimum(MC2.x), minimum(MC3.x)])
            σmax = maximum([maximum(MC1.x), maximum(MC2.x), maximum(MC3.x)])
            scatter!(ax, MC3_μ.x*sc.σ ./ 1e6, MC3_μ.y*sc.σ ./ 1e6)
            probes.app_fric_lab[it] = MC3_μ.y / MC3_μ.x
            σn_ax = LinRange(σmin-0.1*(σmin), σmax+0.1*σmin, 100)
            # τ1    = materials.plasticity.cosϕ[1]/(1+materials.plasticity.sinϕ[1]) * (materials.plasticity.C[1] * materials.plasticity.cosϕ[1] .+ σn_ax.*materials.plasticity.sinϕ[1])
            # τ2    = materials.plasticity.cosϕ[2]/(1+materials.plasticity.sinϕ[2]) * (materials.plasticity.C[2] * materials.plasticity.cosϕ[2] .+ σn_ax.*materials.plasticity.sinϕ[2])
            τ1 = materials.plasticity.C[1] .+ σn_ax .* materials.plasticity.sinϕ[1] ./ materials.plasticity.cosϕ[1]
            τ2 = materials.plasticity.C[2] .+ σn_ax .* materials.plasticity.sinϕ[2] ./ materials.plasticity.cosϕ[2]
            lines!(ax, σn_ax*sc.σ/1e6, τ1*sc.σ/1e6, color=:black)
            lines!(ax, σn_ax*sc.σ/1e6, τ2*sc.σ/1e6, color=:black)

            # ax  = Axis(fig[1,2], xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            # scatter!(ax, 1:niter, log10.(err.x[1:niter]./err.x[1]) )
            # scatter!(ax, 1:niter, log10.(err.y[1:niter]./err.y[1]) )
            # scatter!(ax, 1:niter, log10.(err.p[1:niter]./err.p[1]) )
            # ylims!(ax, -15, 1)
            ax = Axis(fig[3, 2], title=L"$$Stress space", xlabel=L"$P$", ylabel=L"$\tau_{II}$", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            P_ax = LinRange(minimum(P_rock), maximum(P_rock), 100)
            τ_ax_rock = materials.plasticity.C[1] * materials.plasticity.cosϕ[1] .+ P_ax .* materials.plasticity.sinϕ[1]
            τ_ax_gouge = materials.plasticity.C[2] * materials.plasticity.cosϕ[2] .+ P_ax .* materials.plasticity.sinϕ[2]
            lines!(ax, P_ax*sc.σ/1e6, τ_ax_gouge*sc.σ/1e6, color=:black)
            lines!(ax, P_ax*sc.σ/1e6, τ_ax_rock*sc.σ/1e6, color=:black)
            scatter!(ax, P_rock*sc.σ/1e6, (τII_rock .- λ̇_rock .* materials.plasticity.ηvp[2])*sc.σ/1e6, color=:blue)
            scatter!(ax, P_gouge*sc.σ/1e6, (τII_gouge .- λ̇_gouge .* materials.plasticity.ηvp[2])*sc.σ/1e6, color=:red)

            # τ_ax_gouge = materials.C[2]*materials.cosϕ[2] .+ P_ax.*materials.sinϕ[2]
            # lines!(ax, P_ax*sc.σ/1e6, τ_ax_gouge*sc.σ/1e6, color=:red)
            # scatter!(ax, P_gouge*sc.σ/1e6, τII_gouge*sc.σ/1e6, color=:red )

            ax = Axis(fig[0, 1], xlabel="Axial shortening", ylabel="Axial stress [MPa]", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.PW[1:it]*sc.σ./1e6, marker=:diamond ) #, markersize=20
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.PE[1:it]*sc.σ./1e6, marker=:diamond ) #, markersize=20
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.σxxW[1:it]*sc.σ./1e6, marker=:star5 ) #, markersize=20
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.σxxE[1:it]*sc.σ./1e6, marker=:star5 ) #, markersize=20
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.σyyN[1:it]*sc.σ./1e6, marker=:circle )
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.σyyS[1:it]*sc.σ./1e6, marker=:circle )
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, -probe_salt.σxx[1:it]*sc.σ ./ 1e6, marker=:circle)
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, -probe_plate.σyy[1:it]*sc.σ ./ 1e6, marker=:circle)

            # ax  = Axis(fig[0,1], xlabel="Axial shortening", ylabel="Axial stress [MPa]", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            # scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.τII[1:it]*sc.σ./1e6, marker=:diamond ) #, markersize=20

            # ax  = Axis(fig[0,2], xlabel="time [hrs]", ylabel="τII [MPa]", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            # scatter!(ax, probes.t[1:it]*sc.t/3600, probes.τII[1:it]*sc.σ./1e6 )
            # scatter!(ax, probes.t[1:it]*sc.t/3600, probes.τIIW[1:it]*sc.σ./1e6, marker=:star5, markersize=20 )
            # scatter!(ax, probes.t[1:it]*sc.t/3600, probes.τIIE[1:it]*sc.σ./1e6, marker=:star5, markersize=20 )
            # scatter!(ax, probes.t[1:it]*sc.t/3600, probes.τIIS[1:it]*sc.σ./1e6, marker=:diamond, markersize=20 )
            # scatter!(ax, probes.t[1:it]*sc.t/3600, probes.τIIN[1:it]*sc.σ./1e6, marker=:diamond, markersize=20 )

            ax = Axis(fig[0, 2], xlabel="Axial shortening", ylabel="-τ/σ", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            lines!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, ones(it)*tand(materials.plasticity.ϕ[2]), linestyle=:dash, color=:gray)
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.app_fric[1:it])
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.app_fric_lab[1:it])
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probes.true_fric[1:it], marker=:star5) #, markersize=20  
            tanα = cosd(materials.plasticity.ψ[2])*sind(materials.plasticity.ϕ[2]) / (1 - sind(materials.plasticity.ψ[2])*sind(materials.plasticity.ϕ[2]))
            lines!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, ones(it)*tanα, linestyle=:dash, color=:gray)
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probe_salt.σxy[1:it] ./ probe_salt.σyy[1:it], marker=:circle)
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probe_plate.σxy[1:it] ./ probe_plate.σyy[1:it], marker=:star5)
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probe_MC1.σxy[1:it] ./ probe_MC1.σyy[1:it], marker=:diamond)
            scatter!(ax, probes.t[1:it]*ε̇yy*L.y*sc.L*1e3, probe_MC2.σxy[1:it] ./ probe_MC2.σyy[1:it], marker=:xcross)


            ax = Axis(fig[1, 2], xlabel="time [hrs]", ylabel="σ1 angle gouge", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            scatter!(ax, probes.t[1:it]*sc.t/3600, probes.minθ[1:it]*180/π)
            scatter!(ax, probes.t[1:it]*sc.t/3600, probes.meanθ[1:it]*180/π)
            scatter!(ax, probes.t[1:it]*sc.t/3600, probes.maxθ[1:it]*180/π)

            visualise_markers(particles, phases, sc, x, y, time)
            display(fig)
        end
        with_theme(figure, theme_latexfonts())
    end
    display(to)
end

let
    # main((x = 150, y = 250), 1000, 30) # matches friction
    main((x=100, y=200), 500, 30)
end