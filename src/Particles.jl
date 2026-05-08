function InitialiseParticleField(nc, nmpc, L, Δ, materials, noise)
    nphases = length(materials.n)
    num = (x=nmpc.x * (nc.x + 2), y=nmpc.y * (nc.y + 2))
    Δm = (x=L.x / num.x, y=L.y / num.y)
    xm = LinRange(-L.x / 2 - Δ.x + Δm.x / 2, L.x / 2 + Δ.x - Δm.x / 2, num.x)
    ym = LinRange(-L.y / 2 - Δ.y + Δm.y / 2, L.y / 2 + Δ.y - Δm.y / 2, num.y)
    Xm = [xm[i] for i in eachindex(xm), j in eachindex(ym)]
    Ym = [ym[j] for i in eachindex(xm), j in eachindex(ym)]

    # Add noise to marker coordinates
    if noise
        for ind = 1:(num.x*num.y)
            Xm[ind] += (rand() - 0.5) * Δm.x
            Ym[ind] += (rand() - 0.5) * Δm.y
        end
    end
    return (Xm=Xm, Ym=Ym, xm=xm, ym=ym, Δm=Δm, num=num, nphases=nphases)
end

function InitialisePhaseRatios(markers, f)
    phase_ratios = (
        c=[zeros(markers.nphases) for _ in axes(f.xx, 1), _ in axes(f.xx, 2)],
        v=[zeros(markers.nphases) for _ in axes(f.xy, 1), _ in axes(f.xy, 2)],
    )
    phase_weights = (
        c=[zeros(markers.nphases) for _ in axes(f.xx, 1), _ in axes(f.xx, 2)],
        v=[zeros(markers.nphases) for _ in axes(f.xy, 1), _ in axes(f.xy, 2)],
    )
    return phase_ratios, phase_weights
end

function InitialisePhaseRatios(phases::NamedTuple, nphases::Int)
    c = [
        let r = zeros(nphases)
            r[phases.c[i, j]] = 1.0
            r
        end
        for i in axes(phases.c, 1), j in axes(phases.c, 2)
    ]
    v = [
        let r = zeros(nphases)
            r[phases.v[i, j]] = 1.0
            r
        end
        for i in axes(phases.v, 1), j in axes(phases.v, 2)
    ]
    return (c=c, v=v)
end

function MarkerWeight(xm, x, Δx)
    # Compute marker-grid distance and weight
    dst = abs(xm - x)
    w = 1.0 - 2 * dst / Δx
    return w
end

function MarkerWeight_phase!(phase_ratio, phase_weight, x, y, xm, ym, Δ, phase, nphases)
    w_x = MarkerWeight(xm, x, Δ.x)
    w_y = MarkerWeight(ym, y, Δ.y)
    for k = 1:nphases
        phase_ratio[k] += (k === phase) * w_x * w_y
        phase_weight[k] += w_x * w_y
    end
end
function PhaseRatios!(phase_ratios, phase_weights, m, mphase, xce, yce, xve, yve, Δ)

    for I in CartesianIndices(mphase)
        # find indices of grid centroid
        ic = Int64(ceil((m.Xm[I] - xve[1]) / Δ.x))
        jc = Int64(ceil((m.Ym[I] - yve[1]) / Δ.y))
        # find indices of grid verteces
        iv = Int64(ceil((m.Xm[I] - xve[1]) / Δ.x + 0.5))
        jv = Int64(ceil((m.Ym[I] - yve[1]) / Δ.y + 0.5))

        MarkerWeight_phase!(phase_ratios.c[ic, jc], phase_weights.c[ic, jc], xce[ic], yce[jc], m.Xm[I], m.Ym[I], Δ, mphase[I], m.nphases)
        MarkerWeight_phase!(phase_ratios.v[iv, jv], phase_weights.v[iv, jv], xve[iv], yve[jv], m.Xm[I], m.Ym[I], Δ, mphase[I], m.nphases)
    end

    # centroids
    for i in axes(phase_ratios.c, 1), j in axes(phase_ratios.c, 2)
        #  normalize weights and assign to phase ratios
        for k = 1:m.nphases
            phase_ratios.c[i, j][k] = phase_ratios.c[i, j][k] / (phase_weights.c[i, j][k] == 0.0 ? 1 : phase_weights.c[i, j][k])
        end
    end
    # vertices
    for i in axes(phase_ratios.v, 1), j in axes(phase_ratios.v, 2)
        #  normalize weights and assign to phase ratios
        for k = 1:m.nphases
            phase_ratios.v[i, j][k] = phase_ratios.v[i, j][k] / (phase_weights.v[i, j][k] == 0.0 ? 1 : phase_weights.v[i, j][k])
        end
    end
end

function compute_grid_fields!(G, β, ρ, ξ, materials, phase_ratios, nc, size_c, size_v, nphases)
    sum = (c=ones(size_c...), v=ones(size_v...))

    for I in CartesianIndices(β.c)
        i, j = I[1], I[2]
        β.c[i, j] = 0.0
        G.c[i, j] = 0.0
        ρ.c[i, j] = 0.0
        ξ.c[i, j] = 0.0
        sum.c[i, j] = 0.0
        for p = 1:nphases # loop on phases
            if i > 1 && j > 1 && i < nc.x + 2 && j < nc.y + 2
                phase_ratio = phase_ratios.c[i-1, j-1][p]
                β.c[i, j] += phase_ratio * materials.β[p]
                G.c[i, j] += phase_ratio * materials.G[p]
                ρ.c[i, j] += phase_ratio * materials.ρ[p]
                ξ.c[i, j] += phase_ratio * materials.ρ[p]
                sum.c[i, j] += phase_ratio
            end
        end
    end
    G.c[[1 end], :] .= G.c[[2 end - 1], :]
    G.c[:, [1 end]] .= G.c[:, [2 end - 1]]
    β.c[[1 end], :] .= β.c[[2 end - 1], :]
    β.c[:, [1 end]] .= β.c[:, [2 end - 1]]
    ρ.c[[1 end], :] .= ρ.c[[2 end - 1], :]
    ρ.c[:, [1 end]] .= ρ.c[:, [2 end - 1]]
    ξ.c[[1 end], :] .= ξ.c[[2 end - 1], :]
    ξ.c[:, [1 end]] .= ξ.c[:, [2 end - 1]]

    for I in CartesianIndices(G.v)
        i, j = I[1], I[2]
        G.v[i, j] = 0.0
        sum.v[i, j] = 0.0
        for p = 1:nphases # loop on phases
            if i > 1 && j > 1 && i < nc.x + 3 && j < nc.y + 3
                phase_ratio = phase_ratios.v[i-1, j-1][p]
                G.v[i, j] += phase_ratio * materials.G[p]
                sum.v[i, j] += phase_ratio
            end
        end
    end
    G.v[[1 end], :] .= G.v[[2 end - 1], :]
    G.v[:, [1 end]] .= G.v[:, [2 end - 1]]
    @show extrema(sum.c[2:end-1, 2:end-1]), extrema(sum.v[2:end-1, 2:end-1])
end
