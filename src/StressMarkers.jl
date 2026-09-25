
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

function stress_ToParticles!(allocs, sm::StressMarkers, particles)
    centroid2particle!(sm.pτxx, allocs.τ.xx, particles)
    centroid2particle!(sm.pτyy, allocs.τ.yy, particles)
    centroid2particle!(sm.pP, allocs.Pt, particles)
    grid2particle!((sm.pτxy, sm.pω), (allocs.τ.xy, allocs.ω.xy_v), particles)
    return nothing
end

function stress_ToGrid!(sm::StressMarkers, allocs, particles)
    particle2centroid!(allocs.τ0.xx, sm.pτxx, particles)
    particle2centroid!(allocs.τ0.yy, sm.pτyy, particles)
    particle2grid!(allocs.τ0.xy, sm.pτxy, particles)
    particle2centroid!(allocs.Pt0, sm.pP, particles)
    correct_boundaries!(allocs)
    return nothing
end
