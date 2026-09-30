# Helper for tuple of tensors CellArrays for particle movement
tensor_args(tm) = (tm.p..., tm.Δ..., tm.t...)

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

function upper_advected!(tm::TensorMarkers, particles, dt)
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
                tmp = 2 * τ_xy * ω_xy
                @inbounds @index tm.Δ.τxx[ip, I...] = muladd(dt, (tmp + 2(ε̇_xx*τ_xx + ε̇_xy*τ_xy)), Δτxx)
                @inbounds @index tm.Δ.τyy[ip, I...] = muladd(dt, (-tmp + 2(ε̇_yy*τ_yy + ε̇_xy*τ_xy)), Δτyy)
                @inbounds @index tm.Δ.τxy[ip, I...] = muladd(dt, ((τ_yy - τ_xx) * ω_xy + ε̇_xy*(τ_xx + τ_yy) + τ_xy*(ε̇_xx + ε̇_yy)), Δτxy)
            end
        end
    end
    return nothing
end

function rotation_increments!(tm::TensorMarkers, particles, dt)
    return upper_advected!(tm, particles, dt)
end

function constitutive_increments!(allocs::Allocs)
    allocs.D.τxx .= allocs.τ.xx .- allocs.τ0.xx
    allocs.D.τyy .= allocs.τ.yy .- allocs.τ0.yy
    allocs.D.τxy .= allocs.τ.xy .- allocs.τ0.xy
    allocs.D.ω .= allocs.ω.xy_v .- allocs.ω0.xy_v
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
                ω0 = @inbounds @index tm.p.ω[ip, I...]
                Δτxx = @inbounds @index tm.Δ.τxx[ip, I...]
                Δτyy = @inbounds @index tm.Δ.τyy[ip, I...]
                Δτxy = @inbounds @index tm.Δ.τxy[ip, I...]
                ΔP = @inbounds @index tm.Δ.P[ip, I...]
                Δω = @inbounds @index tm.Δ.ω[ip, I...]
                @inbounds @index tm.p.τxx[ip, I...] = τ0_xx + Δτxx
                @inbounds @index tm.p.τyy[ip, I...] = τ0_yy + Δτyy
                @inbounds @index tm.p.τxy[ip, I...] = τ0_xy + Δτxy
                @inbounds @index tm.p.P[ip, I...] = P0 + ΔP
                @inbounds @index tm.p.ω[ip, I...] = ω0 + Δω
            end
        end
    end
    return nothing
end

# Compute increments to be advected including rotation
function increment_stress!(allocs::Allocs, tm::TensorMarkers, particles, Δ)

    compute_vorticity!(allocs, Δ)
    constitutive_increments!(allocs)
    stress_ToParticles!(allocs, tm, particles)

    # temporary stress interpolaton for rotation
    centroid2particle!(tm.t.τxx, allocs.τ.xx, particles)
    centroid2particle!(tm.t.τyy, allocs.τ.yy, particles)
    centroid2particle!(tm.t.P, allocs.Pt, particles)
    grid2particle!((tm.t.τxy, tm.t.ω), (allocs.τ.xy, allocs.ω.xy_v), particles)
    # temporary strain rate interpolation for rotation 
    centroid2particle!(tm.t.ε̇xx, allocs.ε̇.xx, particles)
    centroid2particle!(tm.t.ε̇yy, allocs.ε̇.yy, particles)
    grid2particle!(tm.t.ε̇xy, allocs.ε̇.xy, particles)

    # Update increments with rotation
    rotation_increments!(tm, particles, Δ.t)
    return nothing
end

# Interpolate stress increments to particles
function stress_ToParticles!(allocs::Allocs, tm::TensorMarkers, particles)
    centroid2particle!(tm.Δ.τxx, allocs.D.τxx, particles)
    centroid2particle!(tm.Δ.τyy, allocs.D.τyy, particles)
    centroid2particle!(tm.Δ.P, allocs.ΔPt.c, particles)
    grid2particle!((tm.Δ.τxy, tm.Δ.ω), (allocs.D.τxy, allocs.D.ω), particles)
    return nothing
end

# Interpolate stresses from markers to grid
function stress_ToGrid!(tm::TensorMarkers, allocs, particles)
    # Add increments to stresses
    update_stress_markers!(tm, particles)

    # Interpolate from markers to grid
    particle2centroid!(allocs.τ0.xx, tm.p.τxx, particles)
    particle2centroid!(allocs.τ0.yy, tm.p.τyy, particles)
    particle2grid!((allocs.τ0.xy, allocs.ω0.xy_v), (tm.p.τxy, tm.p.ω), particles)
    particle2centroid!(allocs.Pt0, tm.p.P, particles)

    correct_boundaries!(allocs)
    return nothing
end
