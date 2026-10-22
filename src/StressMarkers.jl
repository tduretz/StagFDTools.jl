# Helper for tuple of tensors CellArrays for particle movement
tensor_args(tm) = values(tm.p)

# Apply zero-gradient in ghost cells/verteces to avoid NaNs
function correct_boundaries!(a)
    a.τ0.xx[1, :] .= a.τ0.xx[2, :]
    a.τ0.xx[end, :] .= a.τ0.xx[end-1, :]
    a.τ0.xx[:, 1] .= a.τ0.xx[:, 2]
    a.τ0.xx[:, end] .= a.τ0.xx[:, end-1]
    a.τ0.yy[1, :] .= a.τ0.yy[2, :]
    a.τ0.yy[end, :] .= a.τ0.yy[end-1, :]
    a.τ0.yy[:, 1] .= a.τ0.yy[:, 2]
    a.τ0.yy[:, end] .= a.τ0.yy[:, end-1]
    a.τ0.xy[1, :] .= a.τ0.xy[2, :]
    a.τ0.xy[end, :] .= a.τ0.xy[end-1, :]
    a.τ0.xy[:, 1] .= a.τ0.xy[:, 2]
    a.τ0.xy[:, end] .= a.τ0.xy[:, end-1]
    a.Pt0[1, :] .= a.Pt0[2, :]
    a.Pt0[end, :] .= a.Pt0[end-1, :]
    a.Pt0[:, 1] .= a.Pt0[:, 2]
    a.Pt0[:, end] .= a.Pt0[:, end-1]
    return nothing
end

# From JR
function Jaumann_rate!(tm::TensorMarkers, particles, dt)
    (; index) = particles
    Threads.@threads for j in axes(tm.p.τxx, 2)
        for i in axes(tm.p.τxx, 1)
            I = (i, j)
            for ip in cellaxes(index)
                @index(index[ip, I...]) || continue
                ω_xy = @inbounds @index tm.t.ω[ip, I...]
                τ_xx = @inbounds @index tm.t.τxx[ip, I...]
                τ_yy = @inbounds @index tm.t.τyy[ip, I...]
                τ_xy = @inbounds @index tm.t.τxy[ip, I...]
                Δτxx = @inbounds @index tm.Δ.τxx[ip, I...]
                Δτyy = @inbounds @index tm.Δ.τyy[ip, I...]
                Δτxy = @inbounds @index tm.Δ.τxy[ip, I...]
                tmp = 2 * τ_xy * ω_xy
                @inbounds @index tm.Δ.τxx[ip, I...] = muladd(dt, tmp, Δτxx)
                @inbounds @index tm.Δ.τyy[ip, I...] = muladd(dt, -tmp, Δτyy)
                @inbounds @index tm.Δ.τxy[ip, I...] = muladd(dt, (τ_yy - τ_xx) * ω_xy, Δτxy)
            end
        end
    end
    return nothing
end

function upper_convected!(tm::TensorMarkers, particles, dt)
    (; index) = particles
    Threads.@threads for j in axes(tm.p.τxx, 2)
        for i in axes(tm.p.τxx, 1)
            I = (i, j)
            for ip in cellaxes(index)
                @index(index[ip, I...]) || continue
                ω_xy = @inbounds @index tm.t.ω[ip, I...]
                τ_xx = @inbounds @index tm.t.τxx[ip, I...]
                τ_yy = @inbounds @index tm.t.τyy[ip, I...]
                τ_xy = @inbounds @index tm.t.τxy[ip, I...]
                ε̇_xx = @inbounds @index tm.t.ε̇xx[ip, I...]
                ε̇_yy = @inbounds @index tm.t.ε̇yy[ip, I...]
                ε̇_xy = @inbounds @index tm.t.ε̇xy[ip, I...]
                Δτxx = @inbounds @index tm.Δ.τxx[ip, I...]
                Δτyy = @inbounds @index tm.Δ.τyy[ip, I...]
                Δτxy = @inbounds @index tm.Δ.τxy[ip, I...]
                ΔP = @inbounds @index tm.Δ.P[ip, I...]
                tmp = 2 * τ_xy * ω_xy
                # Full velocity gradient correction
                ε̇_zz = -(ε̇_xx + ε̇_yy)
                τ_zz = -(τ_xx + τ_yy)
                ε̇_kk = 3*(ε̇_xx + ε̇_yy)
                # trace correction
                tr = 2(ε̇_xx*τ_xx+ε̇_yy*τ_yy+ε̇_zz*τ_zz+ε̇_xy*τ_xy)/3
                @inbounds @index tm.Δ.τxx[ip, I...] = muladd(dt, (tmp + 2(ε̇_xx*τ_xx + ε̇_xy*τ_xy + ε̇_kk/3)), Δτxx)
                @inbounds @index tm.Δ.τyy[ip, I...] = muladd(dt, (-tmp + 2(ε̇_yy*τ_yy + ε̇_xy*τ_xy + ε̇_kk/3)), Δτyy)
                @inbounds @index tm.Δ.τxy[ip, I...] = muladd(dt, ((τ_yy - τ_xx) * ω_xy + ε̇_xy*(τ_xx + τ_yy) + τ_xy*(ε̇_xx + ε̇_yy)), Δτxy)
                @inbounds @index tm.Δ.P[ip, I...] = muladd(-dt, tr, ΔP)
            end
        end
    end
    return nothing
end

function rotation_increments!(tm::TensorMarkers, particles, dt)
    return upper_convected!(tm, particles, dt)
end

function constitutive_increments!(allocs::Allocs)
    allocs.D.τxx .= allocs.τ.xx .- allocs.τ0.xx
    allocs.D.τyy .= allocs.τ.yy .- allocs.τ0.yy
    allocs.D.τxy .= allocs.τ.xy .- allocs.τ0.xy
    allocs.D.Pt .= allocs.ΔPt.c
end

# update stress in the markers from increments
function update_stress_markers!(tm::TensorMarkers, particles)
    (; index) = particles
    Threads.@threads for j in axes(tm.p.τxx, 2)
        for i in axes(tm.p.τxx, 1)
            I = (i, j)
            for ip in cellaxes(index)
                @index(index[ip, I...]) || continue
                τ0_xx = @inbounds @index tm.p.τxx[ip, I...]
                τ0_yy = @inbounds @index tm.p.τyy[ip, I...]
                τ0_xy = @inbounds @index tm.p.τxy[ip, I...]
                P0 = @inbounds @index tm.p.P[ip, I...]
                Δτxx = @inbounds @index tm.Δ.τxx[ip, I...]
                Δτyy = @inbounds @index tm.Δ.τyy[ip, I...]
                Δτxy = @inbounds @index tm.Δ.τxy[ip, I...]
                ΔP = @inbounds @index tm.Δ.P[ip, I...]
                @inbounds @index tm.p.τxx[ip, I...] = τ0_xx + Δτxx
                @inbounds @index tm.p.τyy[ip, I...] = τ0_yy + Δτyy
                @inbounds @index tm.p.τxy[ip, I...] = τ0_xy + Δτxy
                @inbounds @index tm.p.P[ip, I...] = P0 + ΔP
            end
        end
    end
    return nothing
end

function stress_subgrid_diffusion!(tm::TensorMarkers, allocs::Allocs, particles, Δt, comp; d=1.)

    # Interpolate old grid stresses and Maxwell time parameters to markers
    centroid2particle!(tm.t.τ0xx, allocs.τ0.xx, particles)
    centroid2particle!(tm.t.τ0yy, allocs.τ0.yy, particles)
    centroid2particle!(tm.t.P0, allocs.Pt0, particles)
    grid2particle!(tm.t.τ0xy, allocs.τ0.xy, particles)
    centroid2particle!(tm.t.η, allocs.η.c, particles)
    centroid2particle!(tm.t.G, allocs.G.c, particles)

    # Compute subgrid increment
    (; index) = particles
    dΔt = -d*Δt
    Threads.@threads for j in axes(tm.p.τxx, 2)
        for i in axes(tm.p.τxx, 1)
            I = (i, j)
            for ip in cellaxes(index)
                @index(index[ip, I...]) || continue
                nτ0_xx = @inbounds @index tm.t.τ0xx[ip, I...]
                nτ0_yy = @inbounds @index tm.t.τ0yy[ip, I...]
                nτ0_xy = @inbounds @index tm.t.τ0xy[ip, I...]
                # nP0 = @inbounds @index tm.t.P0[ip, I...]
                τ0_xx = @inbounds @index tm.p.τxx[ip, I...]
                τ0_yy = @inbounds @index tm.p.τyy[ip, I...]
                τ0_xy = @inbounds @index tm.p.τxy[ip, I...]
                # P0 = @inbounds @index tm.p.P[ip, I...]

                η = @inbounds @index tm.t.η[ip, I...]
                G = @inbounds @index tm.t.G[ip, I...]
                Δt_mxw = η / G
                # ξ = @inbounds @index tm.t.ξ[ip, I...]
                # K = @inbounds @index tm.t.K[ip, I...]
                # Δt_P = comp * ξ/K

                diff = -expm1(dΔt / Δt_mxw)
                @inbounds @index tm.Δ_sub.τxx[ip, I...] = (nτ0_xx - τ0_xx)*diff
                @inbounds @index tm.Δ_sub.τyy[ip, I...] = (nτ0_yy - τ0_yy)*diff
                @inbounds @index tm.Δ_sub.τxy[ip, I...] = (nτ0_xy - τ0_xy)*diff
                # @inbounds @index tm.Δ_sub.P[ip, I...] = (nP0 - P0)*(-expm1(dΔt/Δt_P))
            end
        end
    end
    return nothing
end

function subgrid_toGrid!(tm::TensorMarkers, allocs::Allocs, particles)
    # Interpolate subgrid increments from markers to grid
    particle2centroid!(allocs.D_sub.τxx, tm.Δ_sub.τxx, particles)
    particle2centroid!(allocs.D_sub.τyy, tm.Δ_sub.τyy, particles)
    particle2grid!(allocs.D_sub.τxy, tm.Δ_sub.τxy, particles)
    particle2centroid!(allocs.D_sub.Pt, tm.Δ_sub.P, particles)

    # Correct increments in the grid
    allocs.D.τxx .-= allocs.D_sub.τxx
    allocs.D.τyy .-= allocs.D_sub.τyy
    allocs.D.τxy .-= allocs.D_sub.τxy
    allocs.D.Pt .-= allocs.D_sub.Pt
    return nothing
end

function add_subgrid_increments!(tm::TensorMarkers, particles)
    (; index) = particles
    Threads.@threads for j in axes(tm.Δ.τxx, 2)
        for i in axes(tm.Δ.τxx, 1)
            I = (i, j)
            for ip in cellaxes(index)
                @index(index[ip, I...]) || continue
                Δτxx = @inbounds @index tm.Δ.τxx[ip, I...]
                Δτyy = @inbounds @index tm.Δ.τyy[ip, I...]
                Δτxy = @inbounds @index tm.Δ.τxy[ip, I...]
                Δτxx_sub = @inbounds @index tm.Δ_sub.τxx[ip, I...]
                Δτyy_sub = @inbounds @index tm.Δ_sub.τyy[ip, I...]
                Δτxy_sub = @inbounds @index tm.Δ_sub.τxy[ip, I...]
                @inbounds @index tm.Δ.τxx[ip, I...] = Δτxx + Δτxx_sub
                @inbounds @index tm.Δ.τyy[ip, I...] = Δτyy + Δτyy_sub
                @inbounds @index tm.Δ.τxy[ip, I...] = Δτxy + Δτxy_sub
            end
        end
    end
    return nothing
end

function add_increments!(tm::TensorMarkers, allocs::Allocs, particles)
    # Interpolate (subgrid-corrected) grid increments to markers
    centroid2particle!(tm.Δ.τxx, allocs.D.τxx, particles)
    centroid2particle!(tm.Δ.τyy, allocs.D.τyy, particles)
    centroid2particle!(tm.Δ.P, allocs.D.Pt, particles)
    grid2particle!(tm.Δ.τxy, allocs.D.τxy, particles)

    # Add back subgrid increments to markers
    add_subgrid_increments!(tm, particles)
    return nothing
end

# Compute stress increments on markers 
function increment_stress!(allocs::Allocs, tm::TensorMarkers, particles, Δt, comp)
    constitutive_increments!(allocs)
    stress_subgrid_diffusion!(tm, allocs, particles, Δt, comp)
    subgrid_toGrid!(tm, allocs, particles)
    add_increments!(tm, allocs, particles)
    return nothing
end

function stress_ToParticles!(allocs::Allocs, tm::TensorMarkers, materials::Materials, particles, Δ)

    # Interpolate stress increments
    comp = materials.compressible
    compute_vorticity!(allocs, Δ)
    increment_stress!(allocs, tm, particles, Δ.t, comp)

    # temporary τ and ε̇ interpolaton
    centroid2particle!(tm.t.τxx, allocs.τ.xx, particles)
    centroid2particle!(tm.t.τyy, allocs.τ.yy, particles)
    centroid2particle!(tm.t.P, allocs.Pt, particles)
    grid2particle!((tm.t.τxy, tm.t.ω), (allocs.τ.xy, allocs.ω.xy_v), particles)
    centroid2particle!(tm.t.ε̇xx, allocs.ε̇.xx, particles)
    centroid2particle!(tm.t.ε̇yy, allocs.ε̇.yy, particles)
    centroid2particle!(tm.t.ε̇yy, allocs.ε̇.yy, particles)
    grid2particle!(tm.t.ε̇xy, allocs.ε̇.xy, particles)

    # update increments with rotation and add them to marker stresses
    rotation_increments!(tm, particles, Δ.t)
    update_stress_markers!(tm, particles)

    return nothing
end

# Interpolate stresses from markers to grid
function stress_ToGrid!(tm::TensorMarkers, allocs, particles)

    # Interpolate from markers to grid
    particle2centroid!(allocs.τ0.xx, tm.p.τxx, particles)
    particle2centroid!(allocs.τ0.yy, tm.p.τyy, particles)
    particle2grid!(allocs.τ0.xy, tm.p.τxy, particles)
    particle2centroid!(allocs.Pt0, tm.p.P, particles)

    correct_boundaries!(allocs)
    return nothing
end
