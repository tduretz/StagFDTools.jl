struct Fields{Tu}
    u::Tu
end

function Ranges(nc)
    return (inx = 2:(nc.x + 1), iny = 2:(nc.y + 1))
end

function Numbering!(N, type, nc)
    neq = nc.x * nc.y
    N.u[2:(end - 1), 2:(end - 1)] .= reshape(1:neq, nc.x, nc.y)

    # Make periodic in x
    for j in axes(type.u, 2)
        if type.u[1, j] === :periodic
            N.u[1, j] = N.u[end - 1, j]
        end
        if type.u[end, j] === :periodic
            N.u[end, j] = N.u[2, j]
        end
    end

    # Make periodic in y
    for i in axes(type.u, 1)
        if type.u[i, 1] === :periodic
            N.u[i, 1] = N.u[i, end - 1]
        end
        if type.u[i, end] === :periodic
            N.u[i, end] = N.u[i, 2]
        end
    end
    return
end

@views function SparsityPattern!(K, num, pattern, nc)
    shift = (x = 1, y = 1)
    for j in (1 + shift.y):(nc.y + shift.y), i in (1 + shift.x):(nc.x + shift.x)
        Local = SMatrix(num.u[(i - 1):(i + 1), (j - 1):(j + 1)] .* pattern.u.u)
        for jj in axes(Local, 2), ii in axes(Local, 1)
            if Local[ii, jj] > 0
                K.u.u[num.u[i, j], Local[ii, jj]] = 1
            end
        end
    end
end

function SparsityPatternPoisson_SA(num, pattern::SMatrix{N, N, T}, nc) where {N, T}
    ndof = maximum(num)
    K = ExtendableSparseMatrix(ndof, ndof)
    shift = (x = 1, y = 1)
    star_shift = (N >>> 1) + 1
    Local = @MMatrix zeros(T, N, N)
    for j in (1 + shift.y):(nc.y + shift.y), i in (1 + shift.x):(nc.x + shift.x)
        gen_local_numbering!(Local, num, pattern, star_shift, i, j)

        for jj in axes(Local, 2), ii in axes(Local, 1)
            idx = Local[ii, jj]
            if idx > 0
                K[num[i, j], idx] = 1
            end
        end
    end
    return K
end

@inline @generated function gen_local_numbering!(Local::MMatrix{N, N}, num, pattern::SMatrix{N, N}, star_shift, i, j) where {N}
    return quote
        Base.@nexprs $N jj -> begin
            Base.@nexprs $N ii -> begin
                @inline
                Local[ii, jj] = num[ii - star_shift + i, jj - star_shift + j]
            end
        end
        Local .*= pattern
        return nothing
    end
end
