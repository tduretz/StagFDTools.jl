printxy(x) = display(rotr90(x[end:-1:1, end:-1:1]))
av2D(x) = @views @. 0.25 * (x[1:end-1, 1:end-1] + x[2:end-0, 1:end-1,] + x[1:end-1, 2:end-0] + x[2:end-0, 2:end-0])

@inline function av2D(x::SMatrix{M,N,T}) where {M,N,T}

    SMatrix{M - 1,N - 1,T}(ntuple(k -> begin

            i = (k - 1) % (M - 1) + 1
            j = (k - 1) ÷ (M - 1) + 1

            @inbounds 0.25 * (
                x[i, j] +
                x[i+1, j] +
                x[i, j+1] +
                x[i+1, j+1]
            )

        end, Val((M - 1) * (N - 1))))
end


@views function GenerateGrid(x, y, Δ, nc)

    X = (
        v=(
            x=LinRange(x.min, x.max, nc.x + 1),
            y=LinRange(y.min, y.max, nc.y + 1),
        ),
        # With ghost vertices
        v_e=(
            x=LinRange(x.min - Δ.x, x.max + Δ.x, nc.x + 3),
            y=LinRange(y.min - Δ.y, y.max + Δ.y, nc.y + 3),
        ),
        c=(
            x=LinRange(x.min + Δ.x / 2, x.max - Δ.x / 2, nc.x),
            y=LinRange(y.min + Δ.y / 2, y.max - Δ.y / 2, nc.y),
        ),
        # With ghost centroids
        c_e=(
            x=LinRange(x.min - Δ.x / 2, x.max + Δ.x / 2, nc.x + 2),
            y=LinRange(y.min - Δ.y / 2, y.max + Δ.y / 2, nc.y + 2),
        ),
        vx=(
            x=LinRange(x.min, x.max, nc.x + 1),
            y=LinRange(y.min + Δ.y / 2, y.max - Δ.y / 2, nc.y),
        ),
        vx_e=(
            x=LinRange(x.min - Δ.x, x.max + Δ.x, nc.x + 3),
            y=LinRange(y.min - 3 / 2 * Δ.y, y.max + 3 / 2 * Δ.y, nc.y + 4),
        ),
        vy=(
            x=LinRange(x.min + Δ.x / 2, x.max - Δ.x / 2, nc.x),
            y=LinRange(y.min, y.max, nc.y + 1),
        ),
        vy_e=(
            x=LinRange(x.min - 3 / 2 * Δ.x, x.max + 3 / 2 * Δ.x, nc.x + 4),
            y=LinRange(y.min - Δ.y, y.max + Δ.y, nc.y + 3),
        )
    )

    return X
end

function Plot_Tangent_Operator(𝐷, Grid)

    Fig_D = Figure(size=(1600, 1600))

    titles = [
        "∂τxx∂εxx" "" "∂τxx∂εyy" "" "∂τxx∂εxy" "" "∂τxx∂εkk" "";
        "∂τyy∂εxx" "" "∂τyy∂εyy" "" "∂τyy∂εxy" "" "∂τyy∂εkk" "";
        "∂τxy∂εxx" "" "∂τxy∂εyy" "" "∂τxy∂εxy" "" "∂τxy∂εkk" "";
        "∂P∂εxx" "" "∂P∂εyy" "" "∂P∂εxy" "" "∂P∂εkk" ""
    ]

    nx, ny = size(𝐷)
    comps = [zeros(nx, ny) for _ in 1:4, _ in 1:4]

    for I in CartesianIndices(𝐷)
        Dloc = 𝐷[I]
        for i in 1:4, j in 1:4
            comps[i, j][I] = Dloc[i, j]
        end
    end

    for i in 1:4, j in 1:2:7
        jc = (j + 1) ÷ 2
        ax = Axis(Fig_D[i, j], title=titles[i, j],
            xlabel="x", ylabel="y", aspect=DataAspect())
        hm = heatmap!(ax, Grid.x, Grid.y, comps[i, jc], colormap=:turbo)
        Colorbar(Fig_D[i, j+1], hm)
    end
    return Fig_D
end
