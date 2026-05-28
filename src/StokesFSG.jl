function Ranges(nc)
    return
end

struct FSG_Array{T1, T2}
    node1::T1
    node2::T2
end

function Base.getindex(x::FSG_Array, i::Int64)
    @assert 0 < i < 3
    i == 1 && return x.node1
    return i == 2 && return x.node2
end

struct Fields{Tx, Ty, Tp}
    Vx::Tx
    Vy::Ty
    Pt::Tp
end

function Base.getindex(x::Fields, i::Int64)
    @assert 0 < i < 4
    i == 1 && return x.Vx
    i == 2 && return x.Vy
    return i == 3 && return x.Pt
end

function Patterns()

    # Stencil extent for each block matrix
    VV = FSG_Array(
        FSG_Array(@SMatrix([0 1 0; 1 1 1; 0 1 0]), @SMatrix([1 1; 1 1])),
        FSG_Array(@SMatrix([1 1; 1 1]), @SMatrix([0 1 0; 1 1 1; 0 1 0]))
    )
    VP = FSG_Array(
        FSG_Array(@SMatrix([1; 1]), @SMatrix([1  1])),
        FSG_Array(@SMatrix([1  1]), @SMatrix([1; 1]))
    )
    PV = FSG_Array(
        FSG_Array(@SMatrix([1; 1]), @SMatrix([1  1])),
        FSG_Array(@SMatrix([1  1]), @SMatrix([1; 1]))
    )
    PP = FSG_Array(@SMatrix([1]), @SMatrix([1]))

    pattern = Fields(
        Fields(VV, VV, VP),
        Fields(VV, VV, VP),
        Fields(PV, PV, PP),
    )
    return pattern
end

function AllocateSparseMatrix(number)

    nVx = [maximum(number.Vx[1]) maximum(number.Vx[2])]
    nVy = [maximum(number.Vy[1]) maximum(number.Vy[2])]
    nPt = [maximum(number.Pt[1]) maximum(number.Pt[2])]

    VxVx = FSG_Array(
        FSG_Array(ExtendableSparseMatrix(nVx[1], nVx[1]), ExtendableSparseMatrix(nVx[1], nVx[2])),
        FSG_Array(ExtendableSparseMatrix(nVx[2], nVx[1]), ExtendableSparseMatrix(nVx[2], nVx[2])),
    )
    VxVy = FSG_Array(
        FSG_Array(ExtendableSparseMatrix(nVx[1], nVy[1]), ExtendableSparseMatrix(nVx[1], nVy[2])),
        FSG_Array(ExtendableSparseMatrix(nVx[2], nVy[1]), ExtendableSparseMatrix(nVx[2], nVy[2])),
    )
    VyVx = FSG_Array(
        FSG_Array(ExtendableSparseMatrix(nVy[1], nVx[1]), ExtendableSparseMatrix(nVy[1], nVx[2])),
        FSG_Array(ExtendableSparseMatrix(nVy[2], nVx[1]), ExtendableSparseMatrix(nVy[2], nVx[2])),
    )
    VyVy = FSG_Array(
        FSG_Array(ExtendableSparseMatrix(nVy[1], nVy[1]), ExtendableSparseMatrix(nVy[1], nVy[2])),
        FSG_Array(ExtendableSparseMatrix(nVy[2], nVy[1]), ExtendableSparseMatrix(nVy[2], nVy[2])),
    )
    VxP = FSG_Array(
        FSG_Array(ExtendableSparseMatrix(nVx[1], nPt[1]), ExtendableSparseMatrix(nVx[1], nPt[2])),
        FSG_Array(ExtendableSparseMatrix(nVx[2], nPt[1]), ExtendableSparseMatrix(nVx[2], nPt[2])),
    )
    VyP = FSG_Array(
        FSG_Array(ExtendableSparseMatrix(nVy[1], nPt[1]), ExtendableSparseMatrix(nVy[1], nPt[2])),
        FSG_Array(ExtendableSparseMatrix(nVy[2], nPt[1]), ExtendableSparseMatrix(nVy[2], nPt[2])),
    )
    PVx = FSG_Array(
        FSG_Array(ExtendableSparseMatrix(nPt[1], nVx[1]), ExtendableSparseMatrix(nPt[1], nVx[2])),
        FSG_Array(ExtendableSparseMatrix(nPt[2], nVx[1]), ExtendableSparseMatrix(nPt[2], nVx[2])),
    )
    PVy = FSG_Array(
        FSG_Array(ExtendableSparseMatrix(nPt[1], nVy[1]), ExtendableSparseMatrix(nPt[1], nVy[2])),
        FSG_Array(ExtendableSparseMatrix(nPt[2], nVy[1]), ExtendableSparseMatrix(nPt[2], nVy[2])),
    )
    PP = FSG_Array(
        FSG_Array(ExtendableSparseMatrix(nPt[1], nPt[1]), ExtendableSparseMatrix(nPt[1], nPt[2])),
        FSG_Array(ExtendableSparseMatrix(nPt[2], nPt[1]), ExtendableSparseMatrix(nPt[2], nPt[2])),
    )

    M = Fields(
        Fields(VxVx, VxVy, VxP),
        Fields(VyVx, VyVy, VyP),
        Fields(PVx, PVy, PP),
    )
    return M
end

function SetRHSSG1!(r, R, number, type, nc)

    nVx = [maximum(number.Vx[1]) maximum(number.Vx[2])]
    nVy = [maximum(number.Vy[1]) maximum(number.Vy[2])]
    nPt = [maximum(number.Pt[1]) maximum(number.Pt[2])]

    for j in 2:(size(R.x[1], 2) - 1), i in 2:(size(R.x[1], 1) - 1)
        if type.Vx[1][i, j] == :in
            ind = number.Vx[1][i, j]
            r[ind] = R.x[1][i, j]
        end
    end
    for j in 2:(size(R.x[2], 2) - 1), i in 2:(size(R.x[2], 1) - 1)
        # if type.Vx[2][i,j] == :in
        #     ind = number.Vx[2][i,j] + nVx[1]
        #     r[ind] = R.x[2][i,j]
        # end
        if type.Vy[2][i, j] == :in
            ind = number.Vy[2][i, j] + nVx[1] #+ nVx[2] + nVy[1]
            r[ind] = R.y[2][i, j]
        end
    end
    for j in 2:(size(R.p[1], 2) - 1), i in 2:(size(R.x[1], 1) - 1)
        if type.Pt[1][i, j] == :in
            ind = number.Pt[1][i, j] + nVx[1] + 0 * nVy[1] + nVy[2] + 0 * nVx[2]
            r[ind] = R.p[1][i, j]
        end
    end
    # for j=1:size(R.p[2],2)-0, i=1:size(R.x[2],1)-0
    #     if type.Pt[1][i,j] == :in
    #         ind = number.Pt[2][i,j] + nVx[1] + nVx[2] + nVy[1] + nVy[2] + nPt[1]
    #         r[ind] = R.p[2][i,j]
    #     end
    # end
    return
end

function UpdateSolutionSG1!(V, Pt, dx, number, type, nc)

    nVx = [maximum(number.Vx[1]) maximum(number.Vx[2])]
    nVy = [maximum(number.Vy[1]) maximum(number.Vy[2])]
    nPt = [maximum(number.Pt[1]) maximum(number.Pt[2])]

    for j in 2:(size(V.x[1], 2) - 1), i in (2:(size(V.x[1], 1) - 1))
        if type.Vx[1][i, j] == :in
            ind = number.Vx[1][i, j]
            V.x[1][i, j] += dx[ind]
        end
        # if type.Vy[1][i,j] == :in
        #     ind = number.Vy[1][i,j] + nVx[1] + nVx[2]
        #     V.y[1][i,j] += dx[ind]
        # end
    end
    for j in 2:(size(V.x[2], 2) - 1), i in 2:(size(V.x[2], 1) - 1)
        # if type.Vx[2][i,j] == :in
        #     ind = number.Vx[2][i,j] + nVx[1]
        #     V.x[2][i,j] += dx[ind]
        # end
        if type.Vy[2][i, j] == :in
            ind = number.Vy[2][i, j] + nVx[1] + 0 * nVx[2] + 0 * nVy[1]
            V.y[2][i, j] += dx[ind]
        end
    end
    for j in 2:(size(Pt[1], 2) - 1), i in 2:(size(Pt[1], 1) - 1)
        if type.Pt[1][i, j] == :in
            ind = number.Pt[1][i, j] + nVx[1] + 0 * nVx[2] + 0 * nVy[1] + nVy[2]
            Pt[1][i, j] += dx[ind]
        end
    end
    # for j=1:size(Pt[2],2)-0, i=1:size(Pt[2],1)-0
    #     if type.Pt[1][i,j] == :in
    #         ind = number.Pt[2][i,j] + nVx[1] + nVx[2] + nVy[1] + nVy[2] + nPt[1]
    #         Pt[2][i,j] += dx[ind]
    #     end
    # end
    return
end

function SetRHSSG2!(r, R, number, type, nc)

    nVx = [maximum(number.Vx[1]) maximum(number.Vx[2])]
    nVy = [maximum(number.Vy[1]) maximum(number.Vy[2])]
    nPt = [maximum(number.Pt[1]) maximum(number.Pt[2])]

    for j in 2:(size(R.x[1], 2) - 1), i in 2:(size(R.x[1], 1) - 1)
        # if type.Vx[1][i,j] == :in
        #     ind = number.Vx[1][i,j]
        #     r[ind] = R.x[1][i,j]
        # end
        if type.Vy[1][i, j] == :in
            ind = number.Vy[1][i, j] + 0 * nVx[1] + nVx[2]
            r[ind] = R.y[1][i, j]
        end
    end
    for j in 2:(size(R.x[2], 2) - 1), i in 2:(size(R.x[2], 1) - 1)
        if type.Vx[2][i, j] == :in
            ind = number.Vx[2][i, j] + 0 * nVx[1]
            r[ind] = R.x[2][i, j]
        end
        # if type.Vy[2][i,j] == :in
        #     ind = number.Vy[2][i,j] + nVx[1] + nVx[2] + nVy[1]
        #     r[ind] = R.y[2][i,j]
        # end
    end
    # for j=2:size(R.p[1],2)-1, i=2:size(R.x[1],1)-1
    #     if type.Pt[1][i,j] == :in
    #         ind = number.Pt[1][i,j] + nVx[1] + nVx[2] + nVy[1] + nVy[2]
    #         r[ind] = R.p[1][i,j]
    #     end
    # end
    for j in 1:(size(R.p[2], 2) - 0), i in 1:(size(R.p[2], 1) - 0)
        if type.Pt[2][i, j] == :in
            ind = number.Pt[2][i, j] + 0 * nVx[1] + nVx[2] + nVy[1] + 0 * nVy[2] + 0 * nPt[1]
            r[ind] = R.p[2][i, j]
        end
    end
    return
end

function UpdateSolutionSG2!(V, Pt, dx, number, type, nc)

    nVx = [maximum(number.Vx[1]) maximum(number.Vx[2])]
    nVy = [maximum(number.Vy[1]) maximum(number.Vy[2])]
    nPt = [maximum(number.Pt[1]) maximum(number.Pt[2])]

    for j in 2:(size(V.x[1], 2) - 1), i in 2:(size(V.x[1], 1) - 1)
        # if type.Vx[1][i,j] == :in
        #     ind = number.Vx[1][i,j]
        #     V.x[1][i,j] += dx[ind]
        # end
        if type.Vy[1][i, j] == :in
            ind = number.Vy[1][i, j] + 0 * nVx[1] + nVx[2]
            V.y[1][i, j] += dx[ind]
        end
    end
    for j in 2:(size(V.x[2], 2) - 1), i in 2:(size(V.x[2], 1) - 1)
        if type.Vx[2][i, j] == :in
            ind = number.Vx[2][i, j] + 0 * nVx[1]
            V.x[2][i, j] += dx[ind]
        end
        # if type.Vy[2][i,j] == :in
        #     ind = number.Vy[2][i,j] + nVx[1] + nVx[2] + nVy[1]
        #     V.y[2][i,j] += dx[ind]
        # end
    end
    # for j=2:size(Pt[1],2)-1, i=2:size(Pt[1],1)-1
    #     if type.Pt[1][i,j] == :in
    #         ind = number.Pt[1][i,j] + nVx[1] + nVx[2] + nVy[1] + nVy[2]
    #         Pt[1][i,j] += dx[ind]
    #     end
    # end
    for j in 1:(size(Pt[2], 2) - 0), i in 1:(size(Pt[2], 1) - 0)
        if type.Pt[2][i, j] == :in
            ind = number.Pt[2][i, j] + 0 * nVx[1] + nVx[2] + nVy[1] + 0 * nVy[2] + 0 * nPt[1]
            Pt[2][i, j] += dx[ind]
        end
    end
    return
end

function SetRHS!(r, R, number, type, nc)

    nVx = [maximum(number.Vx[1]) maximum(number.Vx[2])]
    nVy = [maximum(number.Vy[1]) maximum(number.Vy[2])]
    nPt = [maximum(number.Pt[1]) maximum(number.Pt[2])]

    for j in 2:(size(R.x[1], 2) - 1), i in (2:(size(R.x[1], 1) - 1))
        if type.Vx[1][i, j] == :in
            ind = number.Vx[1][i, j]
            r[ind] = R.x[1][i, j]
        end
        if type.Vy[1][i, j] == :in
            ind = number.Vy[1][i, j] + nVx[1] + nVx[2]
            r[ind] = R.y[1][i, j]
        end
    end
    for j in 2:(size(R.x[2], 2) - 1), i in 2:(size(R.x[2], 1) - 1)
        if type.Vx[2][i, j] == :in
            ind = number.Vx[2][i, j] + nVx[1]
            r[ind] = R.x[2][i, j]
        end
        if type.Vy[2][i, j] == :in
            ind = number.Vy[2][i, j] + nVx[1] + nVx[2] + nVy[1]
            r[ind] = R.y[2][i, j]
        end
    end
    for j in 2:(size(R.p[1], 2) - 1), i in 2:(size(R.x[1], 1) - 1)
        if type.Pt[1][i, j] == :in
            ind = number.Pt[1][i, j] + nVx[1] + nVx[2] + nVy[1] + nVy[2]
            r[ind] = R.p[1][i, j]
        end
    end
    for j in 1:(size(R.p[2], 2) - 0), i in 1:(size(R.x[2], 1) - 0)
        if type.Pt[1][i, j] == :in
            ind = number.Pt[2][i, j] + nVx[1] + nVx[2] + nVy[1] + nVy[2] + nPt[1]
            r[ind] = R.p[2][i, j]
        end
    end
    return
end

function UpdateSolution!(V, Pt, dx, number, type, nc)

    nVx = [maximum(number.Vx[1]) maximum(number.Vx[2])]
    nVy = [maximum(number.Vy[1]) maximum(number.Vy[2])]
    nPt = [maximum(number.Pt[1]) maximum(number.Pt[2])]

    for j in 2:(size(V.x[1], 2) - 1), i in 2:(size(V.x[1], 1) - 1)
        if type.Vx[1][i, j] == :in
            ind = number.Vx[1][i, j]
            V.x[1][i, j] += dx[ind]
        end
        if type.Vy[1][i, j] == :in
            ind = number.Vy[1][i, j] + nVx[1] + nVx[2]
            V.y[1][i, j] += dx[ind]
        end
    end
    for j in 2:(size(V.x[2], 2) - 1), i in 2:(size(V.x[2], 1) - 1)
        if type.Vx[2][i, j] == :in
            ind = number.Vx[2][i, j] + nVx[1]
            V.x[2][i, j] += dx[ind]
        end
        if type.Vy[2][i, j] == :in
            ind = number.Vy[2][i, j] + nVx[1] + nVx[2] + nVy[1]
            V.y[2][i, j] += dx[ind]
        end
    end
    for j in 2:(size(Pt[1], 2) - 1), i in 2:(size(Pt[1], 1) - 1)
        if type.Pt[1][i, j] == :in
            ind = number.Pt[1][i, j] + nVx[1] + nVx[2] + nVy[1] + nVy[2]
            Pt[1][i, j] += dx[ind]
        end
    end
    for j in 1:(size(Pt[2], 2) - 0), i in 1:(size(Pt[2], 1) - 0)
        if type.Pt[2][i, j] == :in
            ind = number.Pt[2][i, j] + nVx[1] + nVx[2] + nVy[1] + nVy[2] + nPt[1]
            Pt[2][i, j] += dx[ind]
        end
    end
    return
end


function Numbering!(N, type, nc)

    ndof_x = 0
    ndof_y = 0
    noisy = false

    ############ Numbering Vx ############
    periodic_west = (sum(any(i -> i == :periodic, type.Vx[1][2, 2:(end - 1)], dims = 2)) > 0 || sum(any(i -> i == :periodic, type.Vy[1][2, 2:(end - 1)], dims = 2)) > 0)
    periodic_south = (sum(any(i -> i == :periodic, type.Vx[1][2:(end - 1), 1], dims = 1)) > 0 || sum(any(i -> i == :periodic, type.Vy[1][2:(end - 1), 1], dims = 1)) > 0)

    # Loop through inner nodes of the mesh
    for j in 2:(size(type.Vx[1], 2) - 1), i in 2:(size(type.Vx[1], 1) - 1)
        if type.Vx[1][i, j] === :Dirichlet_normal || (type.Vx[1][i, j] === :periodic && i == size(type.Vx[1], 1) - 1)
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof_x += 1
            N.Vx[1][i, j] = ndof_x
        end
        if type.Vy[1][i, j] == :Dirichlet_normal || (type.Vy[1][i, j] == :periodic && i == size(type.Vx[1], 1) - 1)
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof_y += 1
            N.Vy[1][i, j] = ndof_y
        end
    end

    # Copy equation indices for periodic cases
    if periodic_west
        N.Vx[1][1, :] .= N.Vx[1][end - 3, :]
        N.Vy[1][1, :] .= N.Vy[1][end - 3, :]
        N.Vx[1][[end - 1 end], :] .= N.Vx[1][[2 3], :]
        N.Vy[1][[end - 1 end], :] .= N.Vy[1][[2 3], :]
    end

    # Copy equation indices for periodic cases
    if periodic_south
        N.Vx[1][:, [1 end]] .= N.Vx[1][:, [end - 1 2]]
        N.Vy[1][:, [1 end]] .= N.Vy[1][:, [end - 1 2]]
    end
    noisy ? printxy(N.Vx[1]) : nothing

    ############ Numbering Vy ############
    ndof_x = 0
    ndof_y = 0
    periodic_west = (sum(any(i -> i == :periodic, type.Vx[2][1, 2:(end - 1)], dims = 2)) > 0 || sum(any(i -> i == :periodic, type.Vy[2][1, 2:(end - 1)], dims = 2)) > 0)
    periodic_south = (sum(any(i -> i == :periodic, type.Vx[2][2:(end - 1), 2], dims = 1)) > 0 || sum(any(i -> i == :periodic, type.Vy[2][2:(end - 1), 2], dims = 1)) > 0)

    # Loop through inner nodes of the mesh
    for j in 2:(size(type.Vx[2], 2) - 1), i in 2:(size(type.Vx[2], 1) - 1)
        if type.Vx[2][i, j] == :Dirichlet_normal || (type.Vx[2][i, j] == :periodic && j > 2)
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof_x += 1
            N.Vx[2][i, j] = ndof_x
        end
        if type.Vy[2][i, j] == :Dirichlet_normal || (type.Vy[2][i, j] != :periodic && j == size(type.Vy[2], 2) - 1)
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof_y += 1
            N.Vy[2][i, j] = ndof_y
        end
    end

    # Copy equation indices for periodic cases
    if periodic_south
        N.Vx[2][:, 1] .= N.Vx[2][:, end - 3]
        N.Vy[2][:, 1] .= N.Vy[2][:, end - 3]
        N.Vx[2][:, [end - 1 end]] .= N.Vx[2][:, [2 3]]
        N.Vy[2][:, [end - 1 end]] .= N.Vy[2][:, [2 3]]
    end

    # Copy equation indices for periodic cases
    if periodic_west
        N.Vx[2][[1 end], :] .= N.Vx[2][[end - 1 2], :]
        N.Vy[2][[1 end], :] .= N.Vy[2][[end - 1 2], :]
    end
    noisy ? printxy(N.Vy[2]) : nothing

    ############ Numbering Pt - CENTROID ############
    neq_Pt = nc.x * nc.y
    N.Pt[1][2:(end - 1), 2:(end - 1)] .= reshape((1:neq_Pt), nc.x, nc.y)

    if periodic_west
        N.Pt[1][1, :] .= N.Pt[1][end - 1, :]
        N.Pt[1][end, :] .= N.Pt[1][2, :]
    end

    if periodic_south
        N.Pt[1][:, 1] .= N.Pt[1][:, end - 1]
        N.Pt[1][:, end] .= N.Pt[1][:, 2]
    end
    noisy ? printxy(N.Pt) : nothing

    ############ Numbering Pt --- VERTEX ############
    ndof = 0

    # Loop through inner nodes of the mesh
    for j in 1:size(type.Pt[2], 2), i in 1:size(type.Pt[2], 1)
        if type.Pt[2][i, j] == :Dirichlet_normal || (type.Pt[2][i, j] == :periodic && (i == size(type.Pt[2], 1) || j == size(type.Pt[2], 2)))
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof += 1
            N.Pt[2][i, j] = ndof
        end
    end

    if periodic_west
        N.Pt[2][end, :] .= N.Pt[2][1, :]
    end
    return if periodic_south
        N.Pt[2][:, end] .= N.Pt[2][:, 1]
    end

end


function Continuity(Vx, V̄x, Vy, V̄y, Pt, P̄t, phase, materials, tx, t̄x, ty, t̄y, bc_val, Δ)
    invΔx = 1 / Δ.x
    invΔy = 1 / Δ.y
    if tx[1, 1] == :Neu_norm_half # West
        Vx[1, 1] = Vx[2, 1] - Δ.x * bc_val.D[1, 1]
    elseif tx[1, 1] == :Dir_norm_half
        Vx[1, 1] = 2 * bc_val.x.W[1] - Vx[2, 1]
    end
    if tx[2, 1] == :Neu_norm_half # East
        Vx[2, 1] = Vx[1, 1] + Δ.x * bc_val.D[1, 1]
    elseif tx[2, 1] == :Dir_norm_half
        Vx[2, 1] = 2 * bc_val.x.E[1] - Vx[1, 1]
    end
    if ty[1, 1] == :Neu_norm_half # South
        Vy[1, 1] = Vy[1, 2] - Δ.y * bc_val.D[2, 2]
    elseif ty[1, 1] == :Dir_norm_half
        Vy[1, 1] = 2 * bc_val.y.S[1] - Vy[1, 2]
    end
    if ty[1, 2] == :Neu_norm_half # North
        Vy[1, 2] = Vy[1, 1] + Δ.y * bc_val.D[2, 2]
    elseif ty[1, 2] == :Dir_norm_half
        Vy[1, 2] = 2 * bc_val.y.N[1] - Vy[1, 1]
    end
    η = materials.η0[phase[1]]
    fp = ((Vx[2, 1] - Vx[1, 1]) * invΔx + (Vy[1, 2] - Vy[1, 1]) * invΔy + 0 * Pt[1] / (η))
    fp *= -1 #η/(Δ.x+Δ.y)
    return fp
end


function ResidualContinuity2D_1!(R, V, Pt, phases, materials, number, types, BC, nc, Δ)

    for j in 2:(size(Pt[1], 2) - 1), i in 2:(size(Pt[1], 1) - 1)
        Vx = FSG_Array(
            MMatrix{2, 1}(V.x[1][ii, jj] for ii in i:(i + 1), jj in j:j),
            MMatrix{1, 2}(V.x[2][ii, jj] for ii in i:i,   jj in j:(j + 1))
        )
        Vy = FSG_Array(
            MMatrix{2, 1}(V.y[1][ii, jj] for ii in i:(i + 1), jj in j:j),
            MMatrix{1, 2}(V.y[2][ii, jj] for ii in i:i,   jj in j:(j + 1))
        )
        typex = FSG_Array(
            SMatrix{2, 1}(types.Vx[1][ii, jj] for ii in i:(i + 1), jj in j:j),
            SMatrix{1, 2}(types.Vx[2][ii, jj] for ii in i:i,   jj in j:(j + 1))
        )
        typey = FSG_Array(
            SMatrix{2, 1}(types.Vy[1][ii, jj] for ii in i:(i + 1), jj in j:j),
            SMatrix{1, 2}(types.Vy[2][ii, jj] for ii in i:i,   jj in j:(j + 1))
        )
        P = FSG_Array(
            MMatrix{1, 1}(Pt[1][ii, jj] for ii in i:i,   jj in j:j),
            MMatrix{2, 2}(Pt[2][ii, jj] for ii in (i - 1):i, jj in (j - 1):j)
        )
        phase = FSG_Array(
            SMatrix{1, 1}(phases[1][ii, jj] for ii in i:i,   jj in j:j),
            SMatrix{2, 2}(phases[2][ii, jj] for ii in (i - 1):i, jj in (j - 1):j)
        )
        bcx = (
            W = SMatrix{1, 2}(BC.W.Vx[jj] for jj in (j - 1):j),
            E = SMatrix{1, 2}(BC.E.Vx[jj] for jj in (j - 1):j),
            S = SMatrix{2, 1}(BC.S.Vx[ii] for ii in (i - 1):i),
            N = SMatrix{2, 1}(BC.N.Vx[ii] for ii in (i - 1):i),
        )
        bcy = (
            W = SMatrix{1, 2}(BC.W.Vy[jj] for jj in (j - 1):j),
            E = SMatrix{1, 2}(BC.E.Vy[jj] for jj in (j - 1):j),
            S = SMatrix{2, 1}(BC.S.Vy[ii] for ii in (i - 1):i),
            N = SMatrix{2, 1}(BC.N.Vy[ii] for ii in (i - 1):i),
        )
        bc_val = (x = bcx, y = bcy, D = BC.W.D)
        if types.Pt[1][i, j] == :in
            R.p[1][i, j] = Continuity(Vx[1], Vx[2], Vy[2], Vy[1], P[1], P[2], phase[1], materials, typex[1], typex[2], typey[2], typey[1], bc_val, Δ)
        end
    end
    return nothing
end

function ResidualContinuity2D_2!(R, V, Pt, phases, materials, number, types, BC, nc, Δ)

    for j in 1:size(Pt[2], 2), i in 1:size(Pt[2], 1)
        Vx = FSG_Array(
            MMatrix{1, 2}(V.x[1][ii, jj] for ii in (i + 1):(i + 1), jj in j:(j + 1)),
            MMatrix{2, 1}(V.x[2][ii, jj] for ii in i:(i + 1),   jj in (j + 1):(j + 1))
        )
        Vy = FSG_Array(
            MMatrix{1, 2}(V.y[1][ii, jj] for ii in (i + 1):(i + 1), jj in j:(j + 1)),
            MMatrix{2, 1}(V.y[2][ii, jj] for ii in i:(i + 1),   jj in (j + 1):(j + 1))
        )
        typex = FSG_Array(
            SMatrix{1, 2}(types.Vx[1][ii, jj] for ii in (i + 1):(i + 1), jj in j:(j + 1)),
            SMatrix{2, 1}(types.Vx[2][ii, jj] for ii in i:(i + 1),   jj in (j + 1):(j + 1))
        )
        typey = FSG_Array(
            SMatrix{1, 2}(types.Vy[1][ii, jj] for ii in (i + 1):(i + 1), jj in j:(j + 1)),
            SMatrix{2, 1}(types.Vy[2][ii, jj] for ii in i:(i + 1),   jj in (j + 1):(j + 1))
        )
        P = FSG_Array(
            MMatrix{2, 2}(Pt[1][ii, jj] for ii in i:(i + 1),   jj in j:(j + 1)),
            MMatrix{1, 1}(Pt[2][ii, jj] for ii in i:i,     jj in j:j)
        )
        phase = FSG_Array(
            SMatrix{2, 2}(phases[1][ii, jj] for ii in i:(i + 1),   jj in j:(j + 1)),
            SMatrix{1, 1}(phases[2][ii, jj] for ii in i:i,     jj in j:j)
        )
        bcx = (
            W = SMatrix{1, 1}(BC.W.Vx[jj] for jj in j:j),
            E = SMatrix{1, 1}(BC.E.Vx[jj] for jj in j:j),
            S = SMatrix{1, 1}(BC.S.Vx[ii] for ii in i:i),
            N = SMatrix{1, 1}(BC.N.Vx[ii] for ii in i:i),
        )
        bcy = (
            W = SMatrix{1, 1}(BC.W.Vy[jj] for jj in j:j),
            E = SMatrix{1, 1}(BC.E.Vy[jj] for jj in j:j),
            S = SMatrix{1, 1}(BC.S.Vy[ii] for ii in i:i),
            N = SMatrix{1, 1}(BC.N.Vy[ii] for ii in i:i),
        )
        bc_val = (x = bcx, y = bcy, D = BC.W.D)
        if types.Pt[2][i, j] == :in
            R.p[2][i, j] = Continuity(Vx[2], Vx[1], Vy[1], Vy[2], P[2], P[1], phase[2], materials, typex[2], typex[1], typey[1], typey[2], bc_val, Δ)
        end
    end
    return nothing
end

function AssembleContinuity2D_1!(K, V, Pt, phases, materials, num, pattern, types, BC, nc, Δ)

    ∂Rp∂Vx1 = @MMatrix ones(2, 1)
    ∂Rp∂Vx2 = @MMatrix ones(1, 2)
    ∂Rp∂Vy1 = @MMatrix ones(2, 1)
    ∂Rp∂Vy2 = @MMatrix ones(1, 2)
    ∂Rp∂Pt1 = @MMatrix ones(1, 1)
    ∂Rp∂Pt2 = @MMatrix ones(2, 2)

    for j in 2:(size(Pt[1], 2) - 1), i in 2:(size(Pt[1], 1) - 1)

        if types.Pt[1][i, j] == :in

            Vx = FSG_Array(
                MMatrix{2, 1}(V.x[1][ii, jj] for ii in i:(i + 1), jj in j:j),
                MMatrix{1, 2}(V.x[2][ii, jj] for ii in i:i,   jj in j:(j + 1))
            )
            Vy = FSG_Array(
                MMatrix{2, 1}(V.y[1][ii, jj] for ii in i:(i + 1), jj in j:j),
                MMatrix{1, 2}(V.y[2][ii, jj] for ii in i:i,   jj in j:(j + 1))
            )
            typex = FSG_Array(
                SMatrix{2, 1}(types.Vx[1][ii, jj] for ii in i:(i + 1), jj in j:j),
                SMatrix{1, 2}(types.Vy[2][ii, jj] for ii in i:i,   jj in j:(j + 1))
            )
            typey = FSG_Array(
                SMatrix{2, 1}(types.Vy[1][ii, jj] for ii in i:(i + 1), jj in j:j),
                SMatrix{1, 2}(types.Vy[2][ii, jj] for ii in i:i,   jj in j:(j + 1))
            )
            P = FSG_Array(
                MMatrix{1, 1}(Pt[1][ii, jj] for ii in i:i,   jj in j:j),
                MMatrix{2, 2}(Pt[2][ii, jj] for ii in (i - 1):i, jj in (j - 1):j)
            )
            phase = FSG_Array(
                SMatrix{1, 1}(phases[1][ii, jj] for ii in i:i,   jj in j:j),
                SMatrix{2, 2}(phases[2][ii, jj] for ii in (i - 1):i, jj in (j - 1):j)
            )
            bcx = (
                W = SMatrix{1, 2}(BC.W.Vx[jj] for jj in (j - 1):j),
                E = SMatrix{1, 2}(BC.E.Vx[jj] for jj in (j - 1):j),
                S = SMatrix{2, 1}(BC.S.Vx[ii] for ii in (i - 1):i),
                N = SMatrix{2, 1}(BC.N.Vx[ii] for ii in (i - 1):i),
            )
            bcy = (
                W = SMatrix{1, 2}(BC.W.Vy[jj] for jj in (j - 1):j),
                E = SMatrix{1, 2}(BC.E.Vy[jj] for jj in (j - 1):j),
                S = SMatrix{2, 1}(BC.S.Vy[ii] for ii in (i - 1):i),
                N = SMatrix{2, 1}(BC.N.Vy[ii] for ii in (i - 1):i),
            )
            bc_val = (x = bcx, y = bcy, D = BC.W.D)

            ∂Rp∂Vx1 .= 0.0
            ∂Rp∂Vx2 .= 0.0
            ∂Rp∂Vy1 .= 0.0
            ∂Rp∂Vy2 .= 0.0
            ∂Rp∂Pt1 .= 0.0
            ∂Rp∂Pt2 .= 0.0
            ∂Vx1, ∂Vx2, ∂Vy2, ∂Vy1, ∂Pt1, ∂Pt2 = ad_partial_gradients(Continuity, (Vx[1], Vx[2], Vy[2], Vy[1], P[1], P[2]), phase[1], materials, typex[1], typex[2], typey[2], typey[1], bc_val, Δ)
            ∂Rp∂Vx1 .= ∂Vx1
            ∂Rp∂Vx2 .= ∂Vx2
            ∂Rp∂Vy2 .= ∂Vy2
            ∂Rp∂Vy1 .= ∂Vy1
            ∂Rp∂Pt1 .= ∂Pt1
            ∂Rp∂Pt2 .= ∂Pt2

            ieq = num.Pt[1][i, j]

            ##################################################################
            # P1 --> Vy1, Vy1
            local_x = num.Vx[1][i:(i + 1), j:j] .* pattern.Pt.Vx[1][1]
            local_y = num.Vy[1][i:(i + 1), j:j] .* pattern.Pt.Vy[1][1]
            for jj in axes(local_x, 2), ii in axes(local_x, 1)
                if (local_x[ii, jj] > 0)
                    K.Pt.Vx[1][1][ieq, local_x[ii, jj]] = ∂Rp∂Vx1[ii, jj]
                end
            end
            for jj in axes(local_y, 2), ii in axes(local_y, 1)
                if (local_y[ii, jj] > 0)
                    K.Pt.Vy[1][1][ieq, local_y[ii, jj]] = ∂Rp∂Vy1[ii, jj]
                end
            end
            # ##################################################################
            # P1 --> Vy2, Vy2
            local_x = num.Vx[2][i:i, j:(j + 1)] .* pattern.Pt.Vx[1][2]
            local_y = num.Vy[2][i:i, j:(j + 1)] .* pattern.Pt.Vy[1][2]
            for jj in axes(local_x, 2), ii in axes(local_x, 1)
                if (local_x[ii, jj] > 0)
                    K.Pt.Vx[1][2][ieq, local_x[ii, jj]] = ∂Rp∂Vx2[ii, jj]
                end
            end
            for jj in axes(local_y, 2), ii in axes(local_y, 1)
                if (local_y[ii, jj] > 0)
                    K.Pt.Vy[1][2][ieq, local_y[ii, jj]] = ∂Rp∂Vy2[ii, jj]
                end
            end
            ##################################################################
            # P1 --> P1

            ##################################################################
            # P1 --> P2
        end
    end

    return
end


function AssembleContinuity2D_2!(K, V, Pt, phases, materials, num, pattern, types, BC, nc, Δ)

    ∂Rp∂Vx2 = @MMatrix ones(2, 1)
    ∂Rp∂Vx1 = @MMatrix ones(1, 2)
    ∂Rp∂Vy2 = @MMatrix ones(2, 1)
    ∂Rp∂Vy1 = @MMatrix ones(1, 2)
    ∂Rp∂Pt1 = @MMatrix ones(2, 2)
    ∂Rp∂Pt2 = @MMatrix ones(1, 1)

    for j in 1:size(Pt[2], 2), i in 1:size(Pt[2], 1)

        if types.Pt[2][i, j] == :in

            Vx = FSG_Array(
                MMatrix{1, 2}(V.x[1][ii, jj] for ii in (i + 1):(i + 1), jj in j:(j + 1)),
                MMatrix{2, 1}(V.x[2][ii, jj] for ii in i:(i + 1),   jj in (j + 1):(j + 1))
            )
            Vy = FSG_Array(
                MMatrix{1, 2}(V.y[1][ii, jj] for ii in (i + 1):(i + 1), jj in j:(j + 1)),
                MMatrix{2, 1}(V.y[2][ii, jj] for ii in i:(i + 1),   jj in (j + 1):(j + 1))
            )
            typex = FSG_Array(
                SMatrix{1, 2}(types.Vx[1][ii, jj] for ii in (i + 1):(i + 1), jj in j:(j + 1)),
                SMatrix{2, 1}(types.Vx[2][ii, jj] for ii in i:(i + 1),   jj in (j + 1):(j + 1))
            )
            typey = FSG_Array(
                SMatrix{1, 2}(types.Vy[1][ii, jj] for ii in (i + 1):(i + 1), jj in j:(j + 1)),
                SMatrix{2, 1}(types.Vy[2][ii, jj] for ii in i:(i + 1),   jj in (j + 1):(j + 1))
            )
            P = FSG_Array(
                MMatrix{2, 2}(Pt[1][ii, jj] for ii in i:(i + 1),   jj in j:(j + 1)),
                MMatrix{1, 1}(Pt[2][ii, jj] for ii in i:i,     jj in j:j)
            )
            phase = FSG_Array(
                SMatrix{2, 2}(phases[1][ii, jj] for ii in i:(i + 1),   jj in j:(j + 1)),
                SMatrix{1, 1}(phases[2][ii, jj] for ii in i:i,     jj in j:j)
            )
            bcx = (
                W = SMatrix{1, 1}(BC.W.Vx[jj] for jj in j:j),
                E = SMatrix{1, 1}(BC.E.Vx[ii] for ii in j:j),
                S = SMatrix{1, 1}(BC.S.Vx[ii] for ii in i:i),
                N = SMatrix{1, 1}(BC.N.Vx[ii] for ii in i:i),
            )
            bcy = (
                W = SMatrix{1, 1}(BC.W.Vy[ii] for ii in j:j),
                E = SMatrix{1, 1}(BC.E.Vy[ii] for ii in j:j),
                S = SMatrix{1, 1}(BC.S.Vy[ii] for ii in i:i),
                N = SMatrix{1, 1}(BC.N.Vy[ii] for ii in i:i),
            )
            bc_val = (x = bcx, y = bcy, D = BC.W.D)

            ∂Rp∂Vx1 .= 0.0
            ∂Rp∂Vx2 .= 0.0
            ∂Rp∂Vy1 .= 0.0
            ∂Rp∂Vy2 .= 0.0
            ∂Rp∂Pt1 .= 0.0
            ∂Rp∂Pt2 .= 0.0
            ∂Vx2, ∂Vx1, ∂Vy1, ∂Vy2, ∂Pt2, ∂Pt1 = ad_partial_gradients(Continuity, (Vx[2], Vx[1], Vy[1], Vy[2], P[2], P[1]), phase[2], materials, typex[2], typex[1], typey[1], typey[2], bc_val, Δ)
            ∂Rp∂Vx2 .= ∂Vx2
            ∂Rp∂Vx1 .= ∂Vx1
            ∂Rp∂Vy1 .= ∂Vy1
            ∂Rp∂Vy2 .= ∂Vy2
            ∂Rp∂Pt2 .= ∂Pt2
            ∂Rp∂Pt1 .= ∂Pt1
            ieq = num.Pt[2][i, j]

            ##################################################################
            # P2 --> Vy1, Vy1
            local_x = num.Vx[1][(i + 1):(i + 1), j:(j + 1)] .* pattern.Pt.Vx[2][1]
            local_y = num.Vy[1][(i + 1):(i + 1), j:(j + 1)] .* pattern.Pt.Vy[2][1]
            for jj in axes(local_x, 2), ii in axes(local_x, 1)
                if (local_x[ii, jj] > 0)
                    K.Pt.Vx[2][1][ieq, local_x[ii, jj]] = ∂Rp∂Vx1[ii, jj]
                end
            end
            for jj in axes(local_y, 2), ii in axes(local_y, 1)
                if (local_y[ii, jj] > 0)
                    K.Pt.Vy[2][1][ieq, local_y[ii, jj]] = ∂Rp∂Vy1[ii, jj]
                end
            end

            ##################################################################
            # P2 --> Vy2, Vy2
            local_x = num.Vx[2][i:(i + 1), (j + 1):(j + 1)] .* pattern.Pt.Vx[2][2]
            local_y = num.Vy[2][i:(i + 1), (j + 1):(j + 1)] .* pattern.Pt.Vy[2][2]
            for jj in axes(local_x, 2), ii in axes(local_x, 1)
                if (local_x[ii, jj] > 0)
                    K.Pt.Vx[2][2][ieq, local_x[ii, jj]] = ∂Rp∂Vx2[ii, jj]
                end
            end
            for jj in axes(local_y, 2), ii in axes(local_y, 1)
                if (local_y[ii, jj] > 0)
                    K.Pt.Vy[2][2][ieq, local_y[ii, jj]] = ∂Rp∂Vy2[ii, jj]
                end
            end
            ##################################################################
            # P2 --> P1

            ##################################################################
            # P2 --> P2

        end
    end

    return
end
