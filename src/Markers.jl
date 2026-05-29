function InitialiseMarkerField(nc, nmpc, L, Δ, x, y, noise)
    num = (x=nmpc.x * (nc.x + 2), y=nmpc.y * (nc.y + 2))
    Δm = (x=L.x / num.x, y=L.y / num.y)
    xm = LinRange(x.min - Δ.x + Δm.x / 2, x.max + Δ.x - Δm.x / 2, num.x)
    ym = LinRange(y.min - Δ.y + Δm.y / 2, y.max + Δ.y - Δm.y / 2, num.y)
    Xm = repeat(xm, outer=num.y)
    Ym = repeat(ym, inner=num.x)
    mphase = ones(Int64, num.x, num.y)
    mphase = vec(mphase)

    if noise
        Xm .+= (rand(length(Xm)) .- 0.5) .* Δm.x
        Ym .+= (rand(length(Ym)) .- 0.5) .* Δm.y
    end
    return (Xm=Xm, Ym=Ym, xm=xm, ym=ym, Δm=Δm, num=num, phase=mphase)
end

function InitialisePhaseRatios(nphases, f)
    phase_ratios = (
        c=[zeros(nphases) for _ in axes(f.xx, 1), _ in axes(f.xx, 2)],
        v=[zeros(nphases) for _ in axes(f.xy, 1), _ in axes(f.xy, 2)],
    )
    phase_weights = (
        c=[zeros(nphases) for _ in axes(f.xx, 1), _ in axes(f.xx, 2)],
        v=[zeros(nphases) for _ in axes(f.xy, 1), _ in axes(f.xy, 2)],
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

function SetPhaseRatios!(phase_ratios, phase_weights, m, xce, yce, xve, yve, Δ, nphases)

    for I in eachindex(m.Xm)
        x, y, phase = m.Xm[I], m.Ym[I], m.phase[I]
        xdx = (x - xve[1]) / Δ.x
        ydy = (y - yve[1]) / Δ.y
        ic, jc = ceil(Int, xdx), ceil(Int, ydy)
        iv, jv = ceil(Int, xdx + 0.5), ceil(Int, ydy + 0.5)

        MarkerWeight_phase!(phase_ratios.c[ic, jc], phase_weights.c[ic, jc], xce[ic], yce[jc], m.Xm[I], m.Ym[I], Δ, phase, nphases)
        MarkerWeight_phase!(phase_ratios.v[iv, jv], phase_weights.v[iv, jv], xve[iv], yve[jv], m.Xm[I], m.Ym[I], Δ, phase, nphases)
    end

    # centroids
    for i in axes(phase_ratios.c, 1), j in axes(phase_ratios.c, 2)
        #  normalize weights and assign to phase ratios
        for k = 1:nphases
            phase_ratios.c[i, j][k] = phase_ratios.c[i, j][k] / (phase_weights.c[i, j][k] == 0.0 ? 1 : phase_weights.c[i, j][k])
        end
    end
    # vertices
    for i in axes(phase_ratios.v, 1), j in axes(phase_ratios.v, 2)
        #  normalize weights and assign to phase ratios
        for k = 1:nphases
            phase_ratios.v[i, j][k] = phase_ratios.v[i, j][k] / (phase_weights.v[i, j][k] == 0.0 ? 1 : phase_weights.v[i, j][k])
        end
    end
end

function compute_grid_fields!(G, β, ρ, ξ, materials, phase_ratios, nc, nphases)
    nxc, nyc = size(G.c)
    @inbounds for j in 1:nyc, i in 1:nxc
        if 1 < i < nc.x + 2 && 1 < j < nc.y + 2
            βc = 0.0
            Gc = 0.0
            ρc = 0.0
            ξc = 0.0
            pr = phase_ratios.c[i, j]
            for p = 1:nphases
                r = pr[p]
                βc += r * materials.β[p]
                Gc += r * materials.G[p]
                ρc += r * materials.ρ[p]
                ξc += r * materials.ξ0[p]
            end
            β.c[i, j] = βc
            G.c[i, j] = Gc
            ρ.c[i, j] = ρc
            ξ.c[i, j] = ξc
        else
            β.c[i, j] = 0.0
            G.c[i, j] = 0.0
            ρ.c[i, j] = 0.0
            ξ.c[i, j] = 0.0
        end
    end

    @inbounds for j in 1:nyc
        G.c[1, j] = G.c[2, j]
        G.c[nxc, j] = G.c[nxc-1, j]
        β.c[1, j] = β.c[2, j]
        β.c[nxc, j] = β.c[nxc-1, j]
        ρ.c[1, j] = ρ.c[2, j]
        ρ.c[nxc, j] = ρ.c[nxc-1, j]
        ξ.c[1, j] = ξ.c[2, j]
        ξ.c[nxc, j] = ξ.c[nxc-1, j]
    end
    @inbounds for i in 1:nxc
        G.c[i, 1] = G.c[i, 2]
        G.c[i, nyc] = G.c[i, nyc-1]
        β.c[i, 1] = β.c[i, 2]
        β.c[i, nyc] = β.c[i, nyc-1]
        ρ.c[i, 1] = ρ.c[i, 2]
        ρ.c[i, nyc] = ρ.c[i, nyc-1]
        ξ.c[i, 1] = ξ.c[i, 2]
        ξ.c[i, nyc] = ξ.c[i, nyc-1]
    end

    nxv, nyv = size(G.v)
    @inbounds for j in 1:nyv, i in 1:nxv
        if 1 < i < nc.x + 3 && 1 < j < nc.y + 3
            Gv = 0.0
            pr = phase_ratios.v[i, j]
            for p = 1:nphases
                Gv += pr[p] * materials.G[p]
            end
            G.v[i, j] = Gv
        else
            G.v[i, j] = 0.0
        end
    end

    @inbounds for j in 1:nyv
        G.v[1, j] = G.v[2, j]
        G.v[nxv, j] = G.v[nxv-1, j]
    end
    @inbounds for i in 1:nxv
        G.v[i, 1] = G.v[i, 2]
        G.v[i, nyv] = G.v[i, nyv-1]
    end
    return nothing
end

function compute_grid_fields_two_phases!(G, Ks, KΦ, Kf, ξ, m, ρsi, ρfi, k_ηf0, n_CK, materials, phase_ratios, nc, nphases)
    nxc, nyc = size(G.c)

    # Centroid arrays
    @inbounds for j in 1:nyc, i in 1:nxc
        if 1 <= i <= nc.x + 2 && 1 <= j <= nc.y + 2
            Ksc    = 0.0
            KΦc    = 0.0
            Kfc    = 0.0
            Gc     = 0.0
            ξc     = 0.0
            mc     = 0.0 
            ρsic   = 0.0
            ρfic   = 0.0
            k_ηf0c = 0.0
            n_CKc  = 0.0
            pr     = phase_ratios.c[i, j]
            for p = 1:nphases
                r = pr[p]
                Ksc    += r * materials.Ks[p]
                KΦc    += r * materials.KΦ[p]
                Ksc    += r * materials.Ks[p]
                Gc     += r * materials.G[p]
                ξc     += r * materials.ξ0[p]
                mc     += r * materials.m[p]
                ρsic   += r * materials.ρs[p]
                ρfic   += r * materials.ρf[p]
                k_ηf0c += r * materials.k_ηf0[p]
                n_CKc  += r * materials.n_CK[p]
            end
            Ks.c[i, j]   = Ksc
            KΦ.c[i, j]   = KΦc
            Ks.c[i, j]   = Ksc
            G.c[i, j]    = Gc
            ξ.c[i, j]    = ξc
            m.c[i, j]    = mc
            ρsi.c[i, j]  = ρsic
            ρfi.c[i, j]  = ρfic
            k_ηf0.c[i,j] = k_ηf0c
            n_CK.c[i,j]  = n_CKc
        end
    end

    @inbounds for j in 1:nyc
        G.c[1, j]    = G.c[2, j]
        G.c[nxc, j]  = G.c[nxc-1, j]
        Ks.c[1, j]   = Ks.c[2, j]
        Ks.c[nxc, j] = Ks.c[nxc-1, j]
        KΦ.c[1, j]   = KΦ.c[2, j]
        KΦ.c[nxc, j] = KΦ.c[nxc-1, j]
        Kf.c[1, j]   = Kf.c[2, j]
        Kf.c[nxc, j] = Kf.c[nxc-1, j]
        ξ.c[1, j]    = ξ.c[2, j]
        ξ.c[nxc, j]  = ξ.c[nxc-1, j]
    end
    @inbounds for i in 1:nxc
        G.c[i, 1]    = G.c[i, 2]
        G.c[i, nyc]  = G.c[i, nyc-1]
        Ks.c[i, 1]   = Ks.c[i, 2]
        Ks.c[i, nyc] = Ks.c[i, nyc-1]
        KΦ.c[i, 1]   = KΦ.c[i, 2]
        KΦ.c[i, nyc] = KΦ.c[i, nyc-1]
        Kf.c[i, 1]   = Kf.c[i, 2]
        Kf.c[i, nyc] = Kf.c[i, nyc-1]
        ξ.c[i, 1]    = ξ.c[i, 2]
        ξ.c[i, nyc]  = ξ.c[i, nyc-1]
    end

    # Vertex arrays
    nxv, nyv = size(G.v)
    @inbounds for j in 1:nyv, i in 1:nxv
        if 1 < i < nc.x + 3 && 1 < j < nc.y + 3
            Gv = 0.0
            pr = phase_ratios.v[i, j]
            for p = 1:nphases
                Gv += pr[p] * materials.G[p]
            end
            G.v[i, j] = Gv
        else
            G.v[i, j] = 0.0
        end
    end

    @inbounds for j in 1:nyv
        G.v[1, j] = G.v[2, j]
        G.v[nxv, j] = G.v[nxv-1, j]
    end
    @inbounds for i in 1:nxv
        G.v[i, 1] = G.v[i, 2]
        G.v[i, nyv] = G.v[i, nyv-1]
    end
    return nothing
end