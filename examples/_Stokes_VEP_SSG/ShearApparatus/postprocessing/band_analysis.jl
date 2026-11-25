# Detection of strain localisation bands in the gouge, from one saved step (step_XXXX.jld2).
#
# Angles: all orientations are axial line angles α ∈ [0°, 180°), measured counter-clockwise from the model x axis (horizontal),
# and reported as angles from the vertical: fromvert(α) = α - 90 ∈ [-90°, 90°) (same sign convention as plot_stress_angle.jl:
# the gouge plane is at +θ from the vertical).
#
# 1. Localisation: inside the analysed gouge region (gouge cells, eroded by `nerode` cells, central `keep` fraction of the gouge length)
#      A50   = smallest fraction of the region that carries 50% of the total plastic strain rate (0.5: uniform, → 0: localised)
#      PR    = (Σλ̇)² / (N Σλ̇²), participation ratio (1: uniform, ≈ area fraction of the bands when localised)
#      fpl   = fraction of plastic cells (λ̇ > 0)
# 2. Band orientations: orientation spectrum of the plastic strain rate field f (λ̇/mean, or log10 λ̇), smoothed (Gaussian, σs cells,
#      normalised inside the gouge so that rock values do not leak in).
#      method = :ridge (default): Hessian of f (Scharr derivative of the Scharr gradient); every cell adds w = (-λmin - |λmax|)² to the bin
#               of the ridge direction (eigenvector of λmax). Crossings of two bands are blobs (λmin ≈ λmax < 0) and get w ≈ 0.
#      method = :gradient: every cell adds |∇f|² to the bin of angle(∇f) + 90° (band edges). Exact for bands that do not overlap,
#               but crossings of overlapping bands give spurious intermediate peaks.
#      The histogram (1° bins) is smoothed on the circle (σh degrees) and divided by its mean: 1 = isotropic.
#      Band families = peaks above 1 + prom, at least `sep` degrees apart. coh = |Σ w e^{2iα}| / Σ w (0: isotropic, 1: one orientation).
#      Synthetic tests (bands 0.6 mm wide, 3 mm apart, in the 30° gouge): one family is recovered within 0.5°; with two crossing families
#      the main one is within ~2° and the secondary one is pushed 3-6° away from it; families < ~15° apart are not separated.
# 3. Expected planes, from the gouge orientation (Y), the mean σ1 orientation in the region (axial mean of θc) and the gouge ϕ, ψ:
#      Y  = gouge boundaries (= C of the S-C-C' nomenclature)
#      R1 = synthetic Riedel shear, conjugate Coulomb plane at 45° - ϕ/2 from σ1 on the side of Y (≈ C')
#      R2 = antithetic Riedel shear, the other conjugate Coulomb plane
#      P  = mirror of R1 with respect to Y
#      R1ψ, R2ψ = the same with the Roscoe angle 45° - ψ/2 (non-associated plasticity: bands form between Coulomb and Roscoe)
#      R1A, R2A = the same with the Arthur angle 45° - (ϕ + ψ)/4 (in between)
#    The shear sense follows from the side of Y on which σ1 lies (σ1 clockwise from Y: dextral).
using Statistics

fromvert(α) = mod(α, 180) - 90                       # line angle → angle from the vertical, in [-90, 90)
adiff(a, b) = mod(a - b + 90, 180) - 90               # signed axial difference a - b, in [-90, 90)

function gauss_kernel(σ)
    r = max(1, ceil(Int, 3σ))
    k = exp.(-(-r:r) .^ 2 ./ (2σ^2))
    return k ./ sum(k)
end

# Separable Gaussian smoothing, zero padded
function smooth(A, σ)
    σ <= 0 && return Float64.(A)
    k = gauss_kernel(σ); r = (length(k) - 1) ÷ 2
    nx, ny = size(A)
    B, C = zeros(nx, ny), zeros(nx, ny)
    for j in 1:ny, i in 1:nx, m in -r:r
        1 <= i + m <= nx && (B[i, j] += k[m+r+1] * A[i+m, j])
    end
    for j in 1:ny, i in 1:nx, m in -r:r
        1 <= j + m <= ny && (C[i, j] += k[m+r+1] * B[i, j+m])
    end
    return C
end

# Erode a mask by n cells (3 x 3 neighbourhood)
function erode(M, n)
    nx, ny = size(M)
    for _ in 1:n
        E = falses(nx, ny)
        for j in 2:ny-1, i in 2:nx-1
            E[i, j] = all(@view M[i-1:i+1, j-1:j+1])
        end
        M = E
    end
    return M
end

# Smooth a histogram defined on the circle (bins of 1°, period 180°)
function smooth_circ(h, σ)
    n = length(h); k = gauss_kernel(σ); r = (length(k) - 1) ÷ 2
    return [sum(k[m+r+1] * h[mod1(b + m, n)] for m in -r:r) for b in 1:n]
end

function analyse_bands(d; gouge=2, keep=0.8, nerode=1, method=:ridge, σs=1.5, σbp=0.0, wpow=2, σh=3.0, field=:linear, prom=0.25, sep=10.0)
    sc   = d.sc
    x, y = d.xc .* (sc.L * 1e3), d.yc .* (sc.L * 1e3)                      # [mm]
    Δ    = x[2] - x[1]
    λ̇    = d.λ̇c ./ sc.t                                                    # [1/s]
    θg   = d.θgouge
    tg   = (-sin(θg), cos(θg))                                             # along the gouge
    ng   = (cos(θg), sin(θg))                                              # gouge normal
    x0, y0 = 0.0, d.Ly * sc.L * 1e3 / 2                                     # gouge centre (initial)
    S    = [(x[i] - x0) * tg[1] + (y[j] - y0) * tg[2] for i in eachindex(x), j in eachindex(y)]
    N    = [(x[i] - x0) * ng[1] + (y[j] - y0) * ng[2] for i in eachindex(x), j in eachindex(y)]
    M    = d.phc .== gouge
    Lh   = maximum(abs, S[M])                                              # gouge half-length [mm]
    K    = abs.(S) .<= keep * Lh
    R    = erode(M, nerode) .& K                                           # analysed region
    n0   = mean(N[M .& K])                                                 # current position of the gouge centre line

    # 3. Expected planes
    θr   = d.θc[R]
    α1   = mod(0.5 * rad2deg(atan(sum(sin.(2θr)), sum(cos.(2θr)))), 180)  # mean σ1 orientation
    αY   = mod(90 + rad2deg(θg), 180)
    s    = sign(adiff(α1, αY))                                             # -1: σ1 clockwise from Y (dextral)
    ϕ, ψ = d.plasticity.ϕ[gouge], d.plasticity.ψ[gouge]
    R1   = mod(α1 - s * (45 - ϕ / 2), 180)
    exp_ = (Y = αY, R1 = R1, R2 = mod(α1 + s * (45 - ϕ / 2), 180), P = mod(2αY - R1, 180), σ1 = α1,
            R1ψ = mod(α1 - s * (45 - ψ / 2), 180), R2ψ = mod(α1 + s * (45 - ψ / 2), 180),
            R1A = mod(α1 - s * (45 - (ϕ + ψ) / 4), 180), R2A = mod(α1 + s * (45 - (ϕ + ψ) / 4), 180))

    sense = s < 0 ? :dextral : :sinistral

    # 1. Localisation (no plastic cell: nothing to analyse)
    λr   = λ̇[R]
    if sum(λr) == 0
        return (; x, y, λ̇, M, R, S, N, n0, ng, tg, x0, y0, A50=NaN, PR=NaN, fpl=0.0, coh=NaN, αb=collect(0.5:1:179.5), dens=fill(NaN, 180),
                  peaks=NamedTuple{(:α, :dens), Tuple{Float64, Float64}}[], exp=exp_, sense, ϕ, ψ, prof=(n=Float64[], λ̇=Float64[]))
    end
    srt  = sort(λr, rev=true)
    A50  = findfirst(>=(0.5 * sum(srt)), cumsum(srt)) / length(srt)
    PR   = sum(λr)^2 / (length(λr) * sum(abs2, λr))
    fpl  = mean(λr .> 0)

    # 2. Gradient-orientation spectrum
    f    = field == :log ? log10.(max.(λ̇, 1e-3 * quantile(λr, 0.99))) : λ̇ ./ mean(λr)
    fs   = smooth(f .* M, σs) ./ max.(smooth(Float64.(M), σs), eps())
    if σbp > 0 # band-pass: remove the structures broader than ~σbp cells (difference of Gaussians, normalised inside the gouge)
        fs = fs .- smooth(fs .* M, σbp) ./ max.(smooth(Float64.(M), σbp), eps())
    end
    # Scharr derivatives (near isotropic); the Hessian is the Scharr derivative of the Scharr gradient
    scharr_x(A, i, j) = (3(A[i+1, j+1] - A[i-1, j+1]) + 10(A[i+1, j] - A[i-1, j]) + 3(A[i+1, j-1] - A[i-1, j-1])) / (32Δ)
    scharr_y(A, i, j) = (3(A[i+1, j+1] - A[i+1, j-1]) + 10(A[i, j+1] - A[i, j-1]) + 3(A[i-1, j+1] - A[i-1, j-1])) / (32Δ)
    nx, ny = size(fs)
    Gx, Gy = zeros(nx, ny), zeros(nx, ny)
    for j in 2:ny-1, i in 2:nx-1
        Gx[i, j], Gy[i, j] = scharr_x(fs, i, j), scharr_y(fs, i, j)
    end
    method == :ridge && (R = R .& erode(M, nerode + 1))   # the Hessian uses the gradient of the neighbours: one more cell from the rock
    h    = zeros(180)
    zc   = 0.0im
    for I in findall(R)
        i, j = Tuple(I)
        if method == :ridge # Hessian: ridge across the eigenvector of the most negative eigenvalue; crossings (blobs) get w ≈ 0
            fxx, fyy = scharr_x(Gx, i, j), scharr_y(Gy, i, j)
            fxy = (scharr_y(Gx, i, j) + scharr_x(Gy, i, j)) / 2
            m, r = (fxx + fyy) / 2, sqrt(((fxx - fyy) / 2)^2 + fxy^2)
            λmin, λmax = m - r, m + r
            w  = max(-λmin - abs(λmax), 0.0)^wpow * (-λmin > abs(λmax))
            α  = mod(rad2deg(0.5 * atan(2fxy, fxx - fyy)), 180) # λmax eigenvector = along the ridge
        else                # gradient: band edges
            w  = Gx[i, j]^2 + Gy[i, j]^2
            α  = mod(rad2deg(atan(Gy[i, j], Gx[i, j])) + 90, 180)
        end
        h[min(floor(Int, α) + 1, 180)] += w
        zc += w * cis(2deg2rad(α))
    end
    coh  = abs(zc) / sum(h)
    dens = smooth_circ(h, σh)
    dens = dens ./ mean(dens)
    αb   = collect(0.5:1:179.5)
    hw   = round(Int, sep / 2)
    pk   = [b for b in 1:180 if dens[b] > 1 + prom && all(dens[b] >= dens[mod1(b + m, 180)] for m in -hw:hw)]
    sort!(pk, by=b -> -dens[b])
    peaks = [(α = αb[b], dens = dens[b]) for b in pk]

    # Profile across the gouge: mean λ̇ in bins of one cell along the normal (central part of the gouge, all gouge cells)
    nb   = (N[M .& K] .- n0)
    edges = floor(minimum(nb) / Δ) * Δ : Δ : ceil(maximum(nb) / Δ) * Δ + Δ
    prof = map(1:length(edges)-1) do k
        v = λ̇[M .& K][(nb .>= edges[k]) .& (nb .< edges[k+1])]
        isempty(v) ? NaN : mean(v)
    end

    return (; x, y, λ̇, M, R, S, N, n0, ng, tg, x0, y0, A50, PR, fpl, coh, αb, dens, peaks, exp=exp_, sense,
              ϕ, ψ, prof = (n = edges[1:end-1] .+ Δ / 2, λ̇ = prof))
end

# Analysed region for Fatbox: the whole gouge strip (no cell removed at the contacts, where the Y shears localise), central keep of its length
fatbox_region(d; kw...) = analyse_bands(d; nerode=0, method=:fatbox, kw...)

# Name of the expected family closest to a band orientation α, and the misfit
function nearest_family(α, e)
    names = (:Y, :R1, :R1A, :R2, :R2A, :P)
    k = argmin([abs(adiff(α, getfield(e, nm))) for nm in names])
    return names[k], adiff(α, getfield(e, names[k]))
end

# Weak band family: directional Fourier power spectrum of λ̇ over the analysed gouge cells (direct DFT on a polar grid of wavevectors).
# For each wavelength λ in λs, the power vs orientation is divided by its median away from the main family (more than `excl` degrees);
# the contrast is averaged over λs and the weak family is the orientation of largest mean contrast away from the main family.
# Returns its line angle α, mean contrast, power relative to the main family (summed over λs) and the misfit to the nearest mesh
# direction (0°, ±45°, 90° from the vertical).
# Caveat: at λ ≲ 3-4 cells, a single family of strands 1-2 cells wide, oblique to the mesh, also gives such peaks (staircase aliasing),
# mostly along the mesh directions: a weak family found there cannot be distinguished from a discretisation artefact
# (check with another resolution).
function weak_family(r; λs=range(0.42, 0.8, length=20), excl=25.0)
    isempty(r.peaks) && return nothing
    cells = findall(r.R)
    X = [r.x[I[1]] for I in cells]; Y = [r.y[I[2]] for I in cells]
    g = r.λ̇[cells]; g = g ./ mean(g) .- 1
    φs = 0.5:1:179.5                                   # wavevector directions; band angle = φ + 90°
    αs = mod.(φs .+ 90, 180)
    αm = r.peaks[1].α
    far = [abs(adiff(a, αm)) > excl for a in αs]
    km  = argmin(abs.(adiff.(αs, αm)))
    C   = zeros(length(φs)); Pw = zeros(length(φs))
    for λ in λs
        p = [abs2(sum(g .* cis.(-(2π / λ) .* (cosd(φ) .* X .+ sind(φ) .* Y)))) for φ in φs]
        C  .+= p ./ median(p[far]) ./ length(λs)
        Pw .+= p
    end
    C[.!far] .= 0
    k = argmax(C)
    mesh = minimum(abs(adiff(αs[k], m)) for m in (0.0, 45.0, 90.0, 135.0))
    return (α = αs[k], contrast = C[k], rel = Pw[k] / Pw[km], mesh = mesh, λs = λs)
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Oriented line filter ("rotating seeded lines"): at every cell of the analysed region, a line segment of length ℓ is rotated over
# 180° (step dα); for each angle α the mean of f = λ̇/⟨λ̇⟩ along the segment (on) is compared with the mean along two parallel segments
# offset by ±d (side). Normalised contrast: Cn(α) = (on - side) / (on + side) ∈ [-1, 1] (independent of the band intensity, so weak
# bands are detected as well as strong ones). Samples outside the gouge are ignored (a segment needs ≥ 60% of its samples inside).
# A cell belongs to a band of orientation α when Cn(α) is a local maximum over α above `thr` and on > `fmin` (f units); up to two
# orientations are kept per cell (crossings), at least `sep2` degrees apart.
# Families: peaks of the histogram of band-cell orientations (smoothed by σh degrees) holding at least `minfrac` of the band cells;
# every band orientation within `win` degrees of a family peak is assigned to it.
# For each family: number of bands = connected groups (8-neighbours) of its cells, after closing gaps of up to `gap` cells along the
# family direction, that extend over at least `minlen` [mm] along that direction; dissipation =
# Σ λ̇ τII over its cells (a cell on two families is shared equally), as a fraction of Σ λ̇ τII over the region; families carrying
# less than `mindiss` of it are dropped. λ̇ τII = τ:ε̇p (deviatoric), since the plastic strain rate invariant is λ̇/2 in this code.
function line_bands(r, τII; ℓ=1.4, d=0.3, dα=2.0, thr=0.4, fmin=0.05, sep2=20.0, σh=4.0, minfrac=0.05, win=12.0, gap=2, minlen=1.0, mindiss=0.01)
    x, y, Δ = r.x, r.y, r.x[2] - r.x[1]
    nx, ny  = size(r.λ̇)
    f   = r.λ̇ ./ mean(r.λ̇[r.R])
    Mf  = Float64.(r.M)
    function bil(F, px, py)                            # bilinear interpolation (NaN outside the grid)
        fi = (px - x[1]) / Δ + 1; fj = (py - y[1]) / Δ + 1
        i = floor(Int, fi); j = floor(Int, fj)
        (1 <= i < nx && 1 <= j < ny) || return NaN
        a = fi - i; b = fj - j
        (1 - a) * (1 - b) * F[i, j] + a * (1 - b) * F[i+1, j] + (1 - a) * b * F[i, j+1] + a * b * F[i+1, j+1]
    end
    K   = round(Int, ℓ / 2 / Δ)
    αs  = collect(0:dα:180-dα)
    cells = findall(r.R)
    Cn  = zeros(length(αs), length(cells))
    Threads.@threads for c in eachindex(cells)
        i, j = Tuple(cells[c]); px, py = x[i], y[j]
        for (k, α) in enumerate(αs)
            u, nr = (cosd(α), sind(α)), (-sind(α), cosd(α))
            son = sside = 0.0; non = nside = 0
            for m in -K:K, (o, isside) in ((0.0, false), (d, true), (-d, true))
                qx, qy = px + m * Δ * u[1] + o * nr[1], py + m * Δ * u[2] + o * nr[2]
                bil(Mf, qx, qy) > 0.5 || continue
                v = bil(f, qx, qy)
                isside ? (sside += v; nside += 1) : (son += v; non += 1)
            end
            if non >= 0.6 * (2K + 1) && nside >= 0.6 * 2 * (2K + 1)
                on, side = son / non, sside / nside
                Cn[k, c] = on > fmin ? (on - side) / (on + side + eps()) : 0.0
            end
        end
    end
    # band orientations per cell: local maxima of Cn over α (circular) above thr, best two, ≥ sep2 apart
    na  = length(αs); hk = max(1, round(Int, sep2 / 2 / dα))
    bands = [Float64[] for _ in cells]
    for c in eachindex(cells)
        v  = view(Cn, :, c)
        mx = [k for k in 1:na if v[k] > thr && all(v[k] >= v[mod1(k + m, na)] for m in -hk:hk)]
        sort!(mx, by=k -> -v[k])
        for k in mx
            all(abs(adiff(αs[k], a)) >= sep2 for a in bands[c]) && push!(bands[c], αs[k])
            length(bands[c]) == 2 && break
        end
    end
    # orientation histogram of band cells (count), smoothed on the circle, and families
    h = zeros(180)
    for b in bands, a in b
        h[min(floor(Int, a) + 1, 180)] += 1
    end
    nband = sum(h)
    hs = smooth_circ(h, σh)
    hw = round(Int, win)
    pk = [b for b in 1:180 if hs[b] > 0 && all(hs[b] >= hs[mod1(b + m, 180)] for m in -hw:hw)]
    # family weight = band cells within ±win of the peak
    wt(b) = sum(h[mod1(b + m, 180)] for m in -hw:hw)
    filter!(b -> wt(b) >= minfrac * nband, pk)
    sort!(pk, by=b -> -wt(b))
    fam_α = [b - 0.5 for b in pk]
    # assignment, band counting and dissipation
    W    = r.λ̇ .* τII
    Wtot = sum(W[r.R])
    fams = map(fam_α) do αf
        Mfam = falses(nx, ny); w = 0.0; ncell = 0
        for (c, b) in enumerate(bands)
            hit = [a for a in b if abs(adiff(a, αf)) <= win]
            isempty(hit) && continue
            I = cells[c]; Mfam[I] = true; ncell += 1
            w += W[I] / length(b)                         # shared between the families of a crossing cell
        end
        # bands: close gaps of ≤ gap cells along the family direction, then connected groups (8-neighbours) extending ≥ minlen along it
        u  = (cosd(αf), sind(αf))
        Md = copy(Mfam)
        for I in findall(Mfam), m in -gap:gap
            K2 = CartesianIndex(I[1] + round(Int, m * u[1]), I[2] + round(Int, m * u[2]))
            checkbounds(Bool, Md, K2) && r.M[K2] && (Md[K2] = true)
        end
        lab = zeros(Int, nx, ny); nb = 0; lens = Float64[]
        for I in findall(Md)
            lab[I] == 0 || continue
            nb += 1; stack = [I]; lab[I] = nb; pmin, pmax = Inf, -Inf
            while !isempty(stack)
                J = pop!(stack)
                p = x[J[1]] * u[1] + y[J[2]] * u[2]; pmin = min(pmin, p); pmax = max(pmax, p)
                for di in -1:1, dj in -1:1
                    K2 = CartesianIndex(J[1] + di, J[2] + dj)
                    checkbounds(Bool, Md, K2) && Md[K2] && lab[K2] == 0 && (lab[K2] = nb; push!(stack, K2))
                end
            end
            push!(lens, pmax - pmin + Δ)
        end
        keep = lens .>= minlen
        (α = αf, ncell = ncell, frac_cells = ncell / length(cells), nbands = count(keep),
         len = isempty(lens[keep]) ? NaN : mean(lens[keep]), diss = w / Wtot, mask = Mfam, lab = lab, kept = findall(keep))
    end
    filter!(fm -> fm.diss >= mindiss, fams)               # families carrying < mindiss of the dissipation are dropped
    return (; αs, Cn, cells, bands, h, hs = hs ./ max(nband, 1), fams, Wtot, nband)
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Band volumes: the band cores of each family (line_bands: cells where a line along the band is brighter than the parallel lines
# at ±d, i.e. the crest of each band, 1-2 cells wide) are grown ACROSS the band to its full width at half maximum: starting from each
# core cell, a neighbour (4-neighbours, inside the analysed region) joins the band when it stays within `along` cells of the core
# cell along the local band direction (the core's own orientation), its λ̇ is at least `frac` × the λ̇ of the core cell, and it is
# at most `wmax` [mm] from it. The growth is breadth first for all families at once, so a cell joins the closest core; growing only
# across the band keeps a family from flooding the bands of another family where they cross or converge.
# Core cells shared by two families (crossings) are split equally. Returns, per family, the volume mask, the volume fraction of the
# analysed region and the share of the plastic dissipation rate (Σ λ̇ τII), the diffuse remainder (cells in no band), and the local
# band orientation of every band cell (α, line angle in degrees, NaN outside the bands; the orientation of its core cell).
function band_volumes(r, lb, τII; frac=0.5, wmax=0.8, along=1.0)
    nx, ny = size(r.λ̇); Δ = r.x[2] - r.x[1]
    nf   = length(lb.fams)
    nmax = max(1, round(Int, wmax / Δ))
    share = zeros(nf, nx, ny)                              # fraction of each cell owned by each family
    αloc  = fill(NaN, nx, ny)                              # local band orientation of the band cells
    claimed = falses(nx, ny)
    queue = Tuple{CartesianIndex{2}, Int, Float64, CartesianIndex{2}, Float64}[]  # cell, family, λ̇ of the core, core cell, core α
    for (c, I) in enumerate(lb.cells)
        hits = [(k, a) for k in 1:nf for a in lb.bands[c] if abs(adiff(a, lb.fams[k].α)) <= 12.0 && lb.fams[k].mask[I]]
        isempty(hits) && continue
        claimed[I] = true
        for (k, a) in hits
            share[k, I] += 1 / length(hits); push!(queue, (I, k, r.λ̇[I], I, a))
        end
        αloc[I] = hits[1][2]
    end
    q = 1
    while q <= length(queue)
        I, k, seed, I0, a = queue[q]; q += 1
        u = (cosd(a), sind(a))
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            K = CartesianIndex(I[1] + di, I[2] + dj)
            (checkbounds(Bool, claimed, K) && r.R[K] && !claimed[K] && r.λ̇[K] >= frac * seed) || continue
            o = (K[1] - I0[1], K[2] - I0[2])                # offset from the core cell [cells]
            abs(o[1] * u[1] + o[2] * u[2]) <= along || continue               # stay across the band
            abs(-o[1] * u[2] + o[2] * u[1]) <= nmax || continue
            claimed[K] = true; share[k, K] = 1.0; αloc[K] = a
            push!(queue, (K, k, seed, I0, a))
        end
    end
    W    = r.λ̇ .* τII
    Wtot = sum(W[r.R]); ncell = count(r.R)
    vols = [(mask = share[k, :, :] .> 0, vol = sum(share[k, :, :][r.R]) / ncell, diss = sum((share[k, :, :] .* W)[r.R]) / Wtot) for k in 1:nf]
    diffuse = (vol = 1 - sum(v.vol for v in vols; init=0.0), diss = 1 - sum(v.diss for v in vols; init=0.0))
    return (; vols, diffuse, α = αloc)
end

# Plastic dissipation rate per orientation: every cell of the analysed region gets a band orientation, binned with its dissipation
# rate λ̇ τII (1° bins, smoothed by σh degrees on the circle): fraction of the total dissipation rate of the region per degree (sums to 1).
#   band cells (αband, from band_volumes): the orientation of their band core (line filter, so weak and crossing families keep their own
#   orientation); other cells: band direction from the structure tensor of λ̇/⟨λ̇⟩ (smoothed by σs cells inside the gouge, tensor
#   averaged over ρ cells), i.e. normal to the dominant gradient.
function work_orientation(r, τII; αband=nothing, σs=1.0, ρ=2.0, σh=2.0)
    nx, ny = size(r.λ̇); Δ = r.x[2] - r.x[1]
    Mf  = Float64.(r.M)
    nrm(A, σ) = smooth(A .* Mf, σ) ./ max.(smooth(Mf, σ), eps())
    fs  = nrm(r.λ̇ ./ mean(r.λ̇[r.R]), σs)
    Gx, Gy = zeros(nx, ny), zeros(nx, ny)
    for j in 2:ny-1, i in 2:nx-1
        Gx[i, j] = (3(fs[i+1, j+1] - fs[i-1, j+1]) + 10(fs[i+1, j] - fs[i-1, j]) + 3(fs[i+1, j-1] - fs[i-1, j-1])) / (32Δ)
        Gy[i, j] = (3(fs[i+1, j+1] - fs[i+1, j-1]) + 10(fs[i, j+1] - fs[i, j-1]) + 3(fs[i-1, j+1] - fs[i-1, j-1])) / (32Δ)
    end
    Jxx, Jyy, Jxy = nrm(Gx .^ 2, ρ), nrm(Gy .^ 2, ρ), nrm(Gx .* Gy, ρ)
    α   = mod.(rad2deg.(0.5 .* atan.(2Jxy, Jxx .- Jyy)) .+ 90, 180)   # band direction (normal to the dominant gradient)
    αband === nothing || (α = [isnan(ab) ? at : mod(ab, 180) for (ab, at) in zip(αband, α)])
    W   = r.λ̇ .* τII
    h   = zeros(180)
    for I in findall(r.R)
        h[min(floor(Int, α[I]) + 1, 180)] += W[I]
    end
    h ./= sum(h)
    return (; α, hw = smooth_circ(h, σh))                             # hw: fraction of the plastic work per degree
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Oriented ridge filter (replaces line_bands + band_volumes for the band families, handles crossings):
# f = λ̇/⟨λ̇⟩ in the gouge (0 outside) is filtered at every cell of the analysed region with an elongated ridge kernel at the
# orientations α = 0, dα, … < 180°: negative second derivative of a Gaussian across the band (σn [mm]) times a Gaussian along it
# (σt [mm]), zero-mean. Contrast C(α) = response / local mean of f (Gaussian of the same envelope): dimensionless, so weak bands
# count as well as strong ones. The elongated kernel resolves orientations ~ atan(σn/σt) apart and follows each band through
# a crossing, so a crossing cell responds at both orientations.
# Families: peaks (≥ sep apart, smoothed by σh degrees) of the dissipation-rate-weighted histogram of the best orientation of the
# ridge cells (max C > c0), each carrying at least `minshare` of the dissipation rate of the region.
# Membership: a cell belongs to family k when C has a local maximum over α within ±win of the family orientation (not at the edge
# of that window: leakage from another family's ridge only gives a slope there) and this maximum is above c0. A cell can belong to
# several families (crossings). Its dissipation rate W = λ̇ τII is split between them in proportion to their contrasts (counted once);
# cells in no family are the diffuse part.
# Returns per family: α, mask, nbands (connected groups of its cells extending ≥ minlen along α), vol (fraction of the region's
# cells; crossings counted in each family), diss (share of the dissipation rate of the region); the crossing mask; the diffuse part;
# hs (orientation histogram of the family cells, per degree, sums to 1; a crossing cell counts once per family) and hw (dissipation
# rate per degree, sums to 1: family cells at their own orientation with their split W, other cells at the structure-tensor
# orientation, see work_orientation).
function ridge_bands(r, τII; σn=0.25, σt=1.5, dα=2.0, c0=0.3, sep=10.0, win=8.0, σh=2.0, minshare=0.02, minlen=1.0)
    nx, ny = size(r.λ̇); Δ = r.x[2] - r.x[1]
    f   = r.λ̇ ./ mean(r.λ̇[r.R]) .* r.M
    αs  = collect(0:dα:180-dα); na = length(αs)
    rad = ceil(Int, 3σt / Δ)
    # kernels: taps (di, dj, k_ridge, k_mean) with |across| ≤ 3σn
    kern = map(αs) do α
        u, v = (cosd(α), sind(α)), (-sind(α), cosd(α))
        taps = Tuple{Int,Int,Float64,Float64}[]
        for di in -rad:rad, dj in -rad:rad
            a = (di * u[1] + dj * u[2]) * Δ; n = (di * v[1] + dj * v[2]) * Δ
            abs(n) <= 3σn && abs(a) <= 3σt || continue
            g = exp(-n^2 / (2σn^2) - a^2 / (2σt^2))
            push!(taps, (di, dj, (1 - n^2 / σn^2) * g, g))
        end
        kr = [t[3] for t in taps]; kr .-= mean(kr); kr ./= sum(max.(kr, 0))   # zero mean, positive lobe normalised to 1
        km = [t[4] for t in taps]; km ./= sum(km)
        [(t[1], t[2], kr[m], km[m]) for (m, t) in enumerate(taps)]
    end
    cells = findall(r.R)
    C = zeros(na, length(cells))
    Threads.@threads for c in eachindex(cells)
        i, j = Tuple(cells[c])
        for k in 1:na
            sr = sm = 0.0
            for (di, dj, kr, km) in kern[k]
                (1 <= i + di <= nx && 1 <= j + dj <= ny) || continue
                v = f[i+di, j+dj]; sr += kr * v; sm += km * v
            end
            C[k, c] = sm > 0 ? sr / sm : 0.0
        end
    end
    W    = r.λ̇ .* τII
    Wc   = W[cells]; Wtot = sum(Wc)
    # families from the best orientation of the ridge cells, weighted by W
    h = zeros(180)
    for c in eachindex(cells)
        k = argmax(view(C, :, c)); C[k, c] > c0 || continue
        h[min(floor(Int, αs[k]) + 1, 180)] += Wc[c]
    end
    hsm = smooth_circ(h, σh); hw2 = round(Int, sep / 2)
    pk  = [b for b in 1:180 if hsm[b] > 0 && all(hsm[b] >= hsm[mod1(b + m, 180)] for m in -hw2:hw2)]
    wt(b) = sum(h[mod1(b + m, 180)] for m in -round(Int, win):round(Int, win))
    filter!(b -> wt(b) >= minshare * Wtot, pk); sort!(pk, by=b -> -wt(b))
    famα = [b - 0.5 for b in pk]; nf = length(famα)
    # membership: local maximum of C over α inside the family window
    kw  = round(Int, win / dα)
    mem = zeros(nf, length(cells)); αc = fill(NaN, nf, length(cells))
    for c in eachindex(cells), (q, αf) in enumerate(famα)
        k0 = argmin(abs.(adiff.(αs, αf)))
        ks = [mod1(k0 + m, na) for m in -kw:kw]
        m  = argmax(C[ks, c]); km = ks[m]
        (1 < m < length(ks)) && C[km, c] > c0 || continue                 # interior maximum above the threshold
        mem[q, c] = C[km, c]; αc[q, c] = αs[km]
    end
    split = mem ./ max.(sum(mem, dims=1), eps())                        # W fraction of each family in each cell
    masks = [falses(nx, ny) for _ in 1:nf]; cross = falses(nx, ny)
    for c in eachindex(cells)
        q = findall(>(0), view(mem, :, c))
        for k in q; masks[k][cells[c]] = true; end
        length(q) > 1 && (cross[cells[c]] = true)
    end
    # dissipation-rate histogram: family cells at their orientation (split W), other cells at the structure-tensor orientation
    wo  = work_orientation(r, τII)
    hW  = zeros(180); hC = zeros(180)
    for c in eachindex(cells)
        if sum(view(mem, :, c)) > 0
            for q in 1:nf
                mem[q, c] > 0 || continue
                b = min(floor(Int, mod(αc[q, c], 180)) + 1, 180)
                hW[b] += split[q, c] * Wc[c]; hC[b] += 1
            end
        else
            hW[min(floor(Int, wo.α[cells[c]]) + 1, 180)] += Wc[c]
        end
    end
    fams = map(1:nf) do q
        M = masks[q]; u = (cosd(famα[q]), sind(famα[q]))
        lab = zeros(Int, nx, ny); nb = 0; nlong = 0
        for I in findall(M)
            lab[I] == 0 || continue
            nb += 1; stack = [I]; lab[I] = nb; pmin, pmax = Inf, -Inf
            while !isempty(stack)
                J = pop!(stack); p = r.x[J[1]] * u[1] + r.y[J[2]] * u[2]; pmin = min(pmin, p); pmax = max(pmax, p)
                for di in -1:1, dj in -1:1
                    K = CartesianIndex(J[1] + di, J[2] + dj)
                    checkbounds(Bool, M, K) && M[K] && lab[K] == 0 && (lab[K] = nb; push!(stack, K))
                end
            end
            pmax - pmin + Δ >= minlen && (nlong += 1)
        end
        (α = famα[q], mask = M, nbands = nlong, vol = count(M) / length(cells), diss = sum(split[q, :] .* Wc) / Wtot)
    end
    diffuse = (vol = count(c -> sum(view(mem, :, c)) == 0, eachindex(cells)) / length(cells),
               diss = 1 - sum(fm.diss for fm in fams; init=0.0))
    return (; fams, cross, diffuse, hs = smooth_circ(hC, σh) ./ max(sum(hC), 1), hw = smooth_circ(hW, σh) ./ Wtot, C, αs, cells)
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Band families from Fatbox segments (fatbox/fatbox_extract.py: thresholded λ̇ → Guo-Hall skeleton → pixel graph → triple
# junctions split by orientation, bends > 60° cut, short segments removed). Reads fatbox/Tet<θ>_nodes.csv and _edges.csv.
# Segments: orientation from a principal-axis fit of their skeleton pixels, length = Σ edge lengths.
# Families: peaks (≥ sep apart) of the length-weighted histogram of segment orientations (smoothed by σh degrees), carrying ≥ minshare
# of the total segment length; each segment joins the nearest family within ±win (else it is left unassigned: "other").
# Band volume: cells of the analysed region within wmax [mm] of a segment's skeleton whose λ̇ is at least frac × λ̇ on the nearest
# skeleton pixel; a cell (nearly) equidistant (± 0.5 cell) from skeleton pixels of several segments belongs to all of them (crossings),
# its dissipation rate λ̇ τII is split equally between them (counted once).
# Returns the same fields as ridge_bands (fams, cross, diffuse, hs = segment length per degree, hw = dissipation rate per degree)
# plus the segments (pixel lists, orientation, length, family).
# planes = true: instead of histogram peaks, every segment is classified by the nearest expected plane R1 (Arthur angle), R2 (Arthur
# angle), P or Y, if within maxmis degrees (R2: up to maxrot on the side of rotation with the shear sense; else no family); scattered short segments of one type then form one family.
# Contact rule (planes = true): the skeleton pixels within ycont [mm] of a gouge contact of a segment within ytol degrees of the gouge
# orientation form a Y shear (the rest of the segment is classified by its angle), even when another plane is nearer in angle (at
# θ = 30° the Arthur R1 is only ~3° from Y, at 50° ~5°): strands that run along a contact and then cut into the gouge are split.
# Use with the full gouge strip as analysed region (fatbox_region): Y shears localise on the contacts.
# Refit (planes = true, refit): a segment still of no family (typically a short crooked skeleton piece, whose principal-axis angle is
# meaningless) gets the orientation of the straight line through its own skeleton pixels holding the most λ̇ (its longest straight
# part; the λ̇ field around is not used, so that a strong neighbouring band cannot capture it), then is classified again.
# Blobs (elongation < elmin) take the family of the elongated segments they touch (shown at that family's expected angle).
function fatbox_bands(r, τII, θ; dir=joinpath(@__DIR__, "fatbox"), sep=10.0, σh=3.0, minshare=0.03, win=12.0, wmax=0.8, frac=0.5,
                      planes=false, maxmis=20.0, prefix="Tet$(θ)", ycont=0.5, ytol=8.0, refit=true, maxrot=30.0, elmin=3.5)
    nx, ny = size(r.λ̇); Δ = r.x[2] - r.x[1]
    rows(f) = [parse.(Int, split(l, ",")) for l in readlines(joinpath(dir, f))[2:end]]
    nodes = rows("$(prefix)_nodes.csv"); edges = rows("$(prefix)_edges.csv")
    pos   = Dict(n[1] => (n[3], n[4]) for n in nodes); comp = Dict(n[1] => n[2] for n in nodes)
    labs  = sort(unique(n[2] for n in nodes))
    segs  = map(labs) do c
        px = [pos[n[1]] for n in nodes if n[2] == c]
        X  = [r.x[p[1]] for p in px]; Y = [r.y[p[2]] for p in px]
        cx, cy = mean(X), mean(Y)
        sxx, syy, sxy = mean((X .- cx) .^ 2), mean((Y .- cy) .^ 2), mean((X .- cx) .* (Y .- cy))
        α  = mod(rad2deg(0.5 * atan(2sxy, sxx - syy)), 180)
        L  = sum(hypot((pos[e[1]] .- pos[e[2]])...) for e in edges if comp[e[1]] == c; init=0.0) * Δ + Δ
        (; px, α, L)
    end
    # families
    h = zeros(180)
    for s in segs; h[min(floor(Int, s.α) + 1, 180)] += s.L; end
    Ltot = sum(h); hsm = smooth_circ(h, σh); hw2 = round(Int, sep / 2)
    pk = [b for b in 1:180 if hsm[b] > 0 && all(hsm[b] >= hsm[mod1(b + m, 180)] for m in -hw2:hw2)]
    wt(b) = sum(h[mod1(b + m, 180)] for m in -round(Int, win):round(Int, win))
    filter!(b -> wt(b) >= minshare * Ltot, pk); sort!(pk, by=b -> -wt(b))
    famα = [b - 0.5 for b in pk]; famname = fill(:none, length(famα)); famcol = collect(eachindex(famα))
    segfam = [isempty(famα) ? 0 : (k = argmin(abs.(adiff.(s.α, famα))); abs(adiff(s.α, famα[k])) <= win ? k : 0) for s in segs]
    if planes   # classify by the nearest expected plane (R1, R2 at the Arthur angle; Y; P) within maxmis, families in that order
        pl  = [(:R1, r.exp.R1A, 1), (:R2, r.exp.R2A, 2), (:P, r.exp.P, 3), (:Y, r.exp.Y, 4)]   # name, line angle, colour index
        # contact rule: a segment near the Y orientation is split into its pixels within ycont of a contact (→ Y) and the others
        rc  = ceil(Int, ycont / Δ) + 1
        dcont(p) = minimum((hypot(di, dj) * Δ for di in -rc:rc, dj in -rc:rc
                            if 1 <= p[1] + di <= nx && 1 <= p[2] + dj <= ny && !r.M[p[1] + di, p[2] + dj]); init=Inf)
        function pxangle(px, α0)   # principal-axis orientation of a part of a segment (≥ 5 pixels, else that of the segment)
            length(px) < 5 && return α0
            X = [r.x[p[1]] for p in px]; Y = [r.y[p[2]] for p in px]; cx, cy = mean(X), mean(Y)
            mod(rad2deg(0.5 * atan(2mean((X .- cx) .* (Y .- cy)), mean((X .- cx) .^ 2) - mean((Y .- cy) .^ 2))), 180)
        end
        segs2, iscont = eltype(segs)[], Bool[]
        for s in segs
            if abs(adiff(s.α, r.exp.Y)) <= ytol
                near = dcont.(s.px) .<= ycont; fn = mean(near)
                fn > 0 && (push!(segs2, (; px = s.px[near], α = pxangle(s.px[near], s.α), L = fn * s.L)); push!(iscont, true))
                fn < 1 && (push!(segs2, (; px = s.px[.!near], α = pxangle(s.px[.!near], s.α), L = (1 - fn) * s.L)); push!(iscont, false))
            else
                push!(segs2, s); push!(iscont, false)
            end
        end
        segs = segs2
        # misfit to plane k; R2 (antithetic) rotates with the shear sense (bookshelf, clockwise when dextral): up to maxrot that side
        rot = r.sense === :dextral ? -1 : 1
        fits(α, k) = (m = adiff(α, pl[k][2]); pl[k][1] === :R2 && sign(m) == rot ? abs(m) <= maxrot : abs(m) <= maxmis)
        classify(α) = (o = sortperm([abs(adiff(α, p[2])) for p in pl]); k = findfirst(k -> fits(α, k), o); k === nothing ? 0 : o[k])
        elong(px) = length(px) < 3 ? 1.0 : (X = [r.x[p[1]] for p in px]; Y = [r.y[p[2]] for p in px]; cx = mean(X); cy = mean(Y);
                    a = mean((X .- cx) .^ 2); b = mean((Y .- cy) .^ 2); c = mean((X .- cx) .* (Y .- cy));
                    m = (a + b) / 2; q = sqrt(((a - b) / 2)^2 + c^2); sqrt((m + q) / max(m - q, 1e-12 * m + 1e-30)))
        sp  = [iscont[q] ? 4 : classify(s.α) for (q, s) in enumerate(segs)]
        if refit   # segments of no family (short, crooked skeleton pieces, whose principal-axis angle is meaningless): orientation of
                   # the straight line through the segment's own skeleton pixels (any offset) that holds the largest λ̇ within 0.75 cell
            for (q, s) in enumerate(segs)
                sp[q] == 0 || continue
                X = [r.x[p[1]] for p in s.px]; Y = [r.y[p[2]] for p in s.px]; w = [r.λ̇[p...] for p in s.px]
                cx, cy = mean(X), mean(Y); best, αf = -Inf, s.α
                for a in 0.0:1.0:179.0
                    n = (-sind(a), cosd(a)); dn = (X .- cx) .* n[1] .+ (Y .- cy) .* n[2]
                    for o in unique(round.(dn ./ (0.25Δ)) .* 0.25Δ)
                        v = sum(w[abs.(dn .- o) .<= 0.75Δ]); v > best && (best = v; αf = a)
                    end
                end
                segs[q] = (; px = s.px, α = αf, L = s.L)
                sp[q] = classify(αf)
            end
        end
        # blobs (elongation < elmin: pixel clusters and zig-zags cut out of a band by the skeleton splitting; their angle is meaningless)
        # take the family of the well-elongated segments they touch (8-neighbours, majority by pixel count), if any
        el    = [elong(s.px) for s in segs]
        owner = Dict(p => q for (q, s) in enumerate(segs) for p in s.px)
        for (q, s) in enumerate(segs)
            (el[q] < elmin && !iscont[q]) || continue
            cnt = Dict{Int, Int}()
            for p in s.px, di in -1:1, dj in -1:1
                o = get(owner, (p[1] + di, p[2] + dj), q)
                (o != q && el[o] >= elmin && sp[o] > 0) && (cnt[sp[o]] = get(cnt, sp[o], 0) + 1)
            end
            isempty(cnt) || (sp[q] = argmax(cnt); segs[q] = (; px = s.px, α = pl[sp[q]][2], L = s.L))
        end
        h .= 0.0   # length histogram at the final (split / refitted) segment orientations
        for s in segs; h[min(floor(Int, s.α) + 1, 180)] += s.L; end
        used = [k for k in eachindex(pl) if any(==(k), sp)]                    # planes with at least one segment
        famα = [pl[k][2] for k in used]; famname = [pl[k][1] for k in used]; famcol = [pl[k][3] for k in used]
        segfam = [k == 0 ? 0 : findfirst(==(k), used) for k in sp]
    end
    nf = length(famα)
    # band volume: nearest skeleton pixel of every segment within wmax
    rad = ceil(Int, wmax / Δ)
    owner = [Int[] for _ in 1:nx, _ in 1:ny]
    dmin  = fill(Inf, nx, ny); dseg = Dict{Tuple{Int,Int,Int}, Float64}()
    for (q, s) in enumerate(segs), p in s.px, di in -rad:rad, dj in -rad:rad
        i, j = p[1] + di, p[2] + dj
        (1 <= i <= nx && 1 <= j <= ny && r.R[i, j]) || continue
        dd = hypot(di, dj); dd * Δ <= wmax || continue
        r.λ̇[i, j] >= frac * r.λ̇[p...] || continue
        key = (i, j, q); dseg[key] = min(get(dseg, key, Inf), dd); dmin[i, j] = min(dmin[i, j], dd)
    end
    for ((i, j, q), dd) in dseg
        dd <= dmin[i, j] + 0.5 && push!(owner[i, j], q)
    end
    W = r.λ̇ .* τII; Wtot = sum(W[r.R]); ncell = count(r.R)
    masks = [falses(nx, ny) for _ in 1:nf]; cross = falses(nx, ny)
    wf = zeros(nf); hW = zeros(180); segW = zeros(length(segs)); hWf = zeros(nf, 180)
    wo = work_orientation(r, τII)
    for I in findall(r.R)
        os = owner[I]
        if isempty(os)
            hW[min(floor(Int, wo.α[I]) + 1, 180)] += W[I]; continue
        end
        fs = unique(segfam[o] for o in os)
        length(filter(>(0), fs)) > 1 && (cross[I] = true)
        for o in os
            k = segfam[o]; w = W[I] / length(os); segW[o] += w
            b = min(floor(Int, segs[o].α) + 1, 180); hW[b] += w
            k > 0 && (masks[k][I] = true; wf[k] += w; hWf[k, b] += w)
        end
    end
    fams = [(α = famα[k], name = famname[k], cidx = famcol[k], mask = masks[k], nbands = count(==(k), segfam), vol = count(masks[k]) / ncell, diss = wf[k] / Wtot,
             Lmean = mean([s.L for (q, s) in enumerate(segs) if segfam[q] == k])) for k in 1:nf]
    banded = sum(W[I] for I in findall(r.R) if !isempty(owner[I]); init=0.0) / Wtot
    other  = banded - sum(wf) / Wtot
    diffuse = (vol = count(I -> isempty(owner[I]), findall(r.R)) / ncell, diss = 1 - banded)
    return (; fams, cross, diffuse, other, hs = smooth_circ(h, 2.0) ./ max(Ltot, eps()), hw = smooth_circ(hW, 2.0) ./ Wtot, segs, segfam,
              hwf = [smooth_circ(hWf[k, :], 2.0) ./ Wtot for k in 1:nf],   # work rate per degree of each family (at its segments' angles)
              segshare = segW ./ Wtot)
end
