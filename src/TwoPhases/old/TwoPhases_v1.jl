struct Fields{Tx,Ty,Tp,Tpf}
    Vx::Tx
    Vy::Ty
    Pt::Tp
    Pf::Tpf
end

function Base.getindex(x::Fields, i::Int64)
    @assert 0 < i < 5 
    i == 1 && return x.Vx
    i == 2 && return x.Vy
    i == 3 && return x.Pt
    i == 4 && return x.Pf
end

function Ranges(nc)     
    return (inx_Vx = 2:nc.x+2, iny_Vx = 3:nc.y+2, inx_Vy = 3:nc.x+2, iny_Vy = 2:nc.y+2, inx_c = 2:nc.x+1, iny_c = 2:nc.y+1, inx_v = 2:nc.x+2, iny_v = 2:nc.y+2, size_x = (nc.x+3, nc.y+4), size_y = (nc.x+4, nc.y+3), size_c = (nc.x+2, nc.y+2), size_v = (nc.x+3, nc.y+3))
end

function Numbering!(N, type, nc)
    
    ndof  = 0
    neq   = 0
    noisy = false

    ############ Numbering Vx ############
    periodic_west  = sum(any(i->i==:periodic, type.Vx[2,:], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, type.Vx[:,2], dims=1)) > 0

    shift  = (periodic_west) ? 1 : 0 
    # Loop through inner nodes of the mesh
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vx[i,j] == :Dirichlet_normal || (type.Vx[i,j] != :periodic && i==nc.x+3-1) || type.Vx[i,j] == :constant 
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof+=1
            N.Vx[i,j] = ndof  
        end
    end

    # Copy equation indices for periodic cases
    if periodic_west
        N.Vx[1,:] .= N.Vx[end-2,:]
    end

    # Copy equation indices for periodic cases
    if periodic_south
        # South
        N.Vx[:,1] .= N.Vx[:,end-3]
        N.Vx[:,2] .= N.Vx[:,end-2]
        # North
        N.Vx[:,end]   .= N.Vx[:,4]
        N.Vx[:,end-1] .= N.Vx[:,3]
    end
    noisy ? printxy(N.Vx) : nothing

    neq = maximum(N.Vx)

    ############ Numbering Vy ############
    ndof  = 0
    periodic_west  = sum(any(i->i==:periodic, type.Vy[2,:], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, type.Vy[:,2], dims=1)) > 0
    shift = periodic_south ? 1 : 0
    # Loop through inner nodes of the mesh
    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vy[i,j] == :Dirichlet_normal || (type.Vy[i,j] != :periodic && j==nc.y+3-1) || type.Vy[i,j] == :constant 
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof+=1
            N.Vy[i,j] = ndof  
        end
    end

    # Copy equation indices for periodic cases
    if periodic_south
        N.Vy[:,1] .= N.Vy[:,end-2]
    end

    # Copy equation indices for periodic cases
    if periodic_west
        # West
        N.Vy[1,:] .= N.Vy[end-3,:]
        N.Vy[2,:] .= N.Vy[end-2,:]
        # East
        N.Vy[end,:]   .= N.Vy[4,:]
        N.Vy[end-1,:] .= N.Vy[3,:]
    end
    noisy ? printxy(N.Vy) : nothing

    neq = maximum(N.Vy)

    ############ Numbering Pt ############
    # neq_Pt                     = nc.x * nc.y
    # N.Pt[2:end-1,2:end-1] .= reshape((1:neq_Pt) .+ 0*neq, nc.x, nc.y)
    ii = 0
    for j=1:nc.y, i=1:nc.x
        if type.Pt[i+1,j+1] != :constant
            ii += 1
            N.Pt[i+1,j+1] = ii
        end
    end

    if periodic_west
        N.Pt[1,:]   .= N.Pt[end-1,:]
        N.Pt[end,:] .= N.Pt[2,:]
    end

    if periodic_south
        N.Pt[:,1]   .= N.Pt[:,end-1]
        N.Pt[:,end] .= N.Pt[:,2]
    end
    noisy ? printxy(N.Pt) : nothing

    neq = maximum(N.Pt)

    ############ Numbering Pf ############

    # neq_Pf                    = nc.x * nc.y
    # N.Pf[2:end-1,2:end-1] .= reshape(1:neq_Pf, nc.x, nc.y)
    ii = 0
    for j=1:nc.y, i=1:nc.x
        if type.Pf[i+1,j+1] != :constant
            ii += 1
            N.Pf[i+1,j+1] = ii
        end
    end

    # Make periodic in x
    for j in axes(type.Pf,2)
        if type.Pf[1,j] === :periodic
            N.Pf[1,j] = N.Pf[end-1,j]
        end
        if type.Pf[end,j] === :periodic
            N.Pf[end,j] = N.Pf[2,j]
        end
    end

    # Make periodic in y
    for i in axes(type.Pf,1)
        if type.Pf[i,1] === :periodic
            N.Pf[i,1] = N.Pf[i,end-1]
        end
        if type.Pf[i,end] === :periodic
            N.Pf[i,end] = N.Pf[i,2]
        end
    end

end

function SetRHS!(r, R, number, type, nc)

    nVx, nVy, nPt   = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)

    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vx[i,j] == :in
            ind = number.Vx[i,j]
            r[ind] = R.x[i,j]
        end
    end
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vy[i,j] == :in
            ind = number.Vy[i,j] + nVx
            r[ind] = R.y[i,j]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pt[i,j] == :in
            ind = number.Pt[i,j] + nVx + nVy
            r[ind] = R.pt[i,j]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pf[i,j] == :in
            ind = number.Pf[i,j] + nVx + nVy + nPt
            r[ind] = R.pf[i,j]
        end
    end
end

function UpdateSolution!(V, P, dx, number, type, nc)

    nVx, nVy, nPt   = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)

    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vx[i,j] == :in
            ind = number.Vx[i,j]
            V.x[i,j] += dx[ind] 
        end
    end
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vy[i,j] == :in
            ind = number.Vy[i,j] + nVx
            V.y[i,j] += dx[ind]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pt[i,j] == :in
            ind = number.Pt[i,j] + nVx + nVy
            P.t[i,j] += dx[ind]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pf[i,j] == :in
            ind = number.Pf[i,j] + nVx + nVy + nPt
            P.f[i,j] += dx[ind]
        end
    end
end

@views function SparsityPattern!(K, num, pattern, nc) 
    ############ Fields Vx ############
    shift  = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Vx --- Vx
        Local = num.Vx[i-1:i+1,j-1:j+1] .* pattern[1][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][1][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Vy
        Local = num.Vy[i-1:i+2,j-2:j+1] .* pattern[1][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][2][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Pt
        Local = num.Pt[i-1:i,j-2:j] .* pattern[1][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][3][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Pf
        Local = num.Pf[i-1:i,j-2:j] .* pattern[1][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][4][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Fields Vy ############
    shift  = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Vy --- Vx
        Local = num.Vx[i-2:i+1,j-1:j+2] .* pattern[2][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][1][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Vy
        Local = num.Vy[i-1:i+1,j-1:j+1] .* pattern[2][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][2][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Pt
        Local = num.Pt[i-2:i,j-1:j] .* pattern[2][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][3][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Pf
        Local = num.Pf[i-2:i,j-1:j] .* pattern[2][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][4][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Fields Pt ############
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Pt --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[3][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][1][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[3][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][2][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Pt
        Local = num.Pt[i,j] .* pattern[3][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][3][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Pf
        Local = num.Pf[i,j] .* pattern[3][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][4][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Fields Pf ############
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Pf --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[4][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][1][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pf --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[4][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][2][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pf --- Pt
        Local = num.Pt[i,j] .* pattern[4][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][3][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pf --- Pf
        Local = num.Pf[i-1:i+1,j-1:j+1] .* pattern[4][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][4][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ End ############
end


function Momentum_x(Vx, Vy, Pt, Pf, η, type, bcv, Δ)
    
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y

    for j=1:4
        if type.y[1,j] == :Dirichlet_tangent
            Vy[1,j] = fma(2, bcv.y[1,j], -Vy[2,j])
        elseif type.y[1,j] == :Neumann_tangent
            Vy[1,j] = fma(Δ.x, bcv.y[1,j], Vy[2,j])
        end
        if type.y[4,j] == :Dirichlet_tangent 
            Vy[4,j] = fma(2, bcv.y[4,j], -Vy[3,j])
        elseif type.y[4,j] == :Neumann_tangent
            Vy[4,j] = fma(Δ.x, bcv.y[4,j], Vy[3,j])
        end
    end

    for i=1:3
        if type.x[i,1] == :Dirichlet_tangent
            Vx[i,1] = fma(2, bcv.x[i,1], -Vx[i,2])
        elseif type.x[i,1] == :Neumann_tangent
            Vx[i,1] = fma(Δ.y, bcv.x[i,1], Vx[i,2])
        end
        if type.x[i,end] == :Dirichlet_tangent 
            Vx[i,end] = fma(2, bcv.x[i,end], -Vx[i,end-1])
        elseif type.x[i,end] == :Neumann_tangent
            Vx[i,end] = fma(Δ.y, bcv.x[i,end], Vx[i,end-1])
        end
    end
     
    Dxx = (Vx[2:end,:] - Vx[1:end-1,:]) * invΔx             # Static Arrays ???
    Dyy = (Vy[2:end-1,2:end] - Vy[2:end-1,1:end-1]) * invΔy             
    Dkk = Dxx + Dyy

    Dxy = (Vx[:,2:end] - Vx[:,1:end-1]) * invΔy 
    Dyx = (Vy[2:end,2:end-1] - Vy[1:end-1,2:end-1]) * invΔx 

    ε̇xx = Dxx - 1/3*Dkk
    ε̇yy = Dyy - 1/3*Dkk
    ε̇xy = 1/2 * ( Dxy + Dyx ) 

    ηc = 0.25*(η.x[1:end-1,:] .+ η.x[2:end,:] .+ η.y[2:end-1,1:end-1] .+ η.y[2:end-1,2:end])
    ηv = 0.25*(η.x[:,1:end-1] .+ η.x[:,2:end] .+ η.y[1:end-1,2:end-1] .+ η.y[2:end,2:end-1])

    τxx = 2 * ηc .* ε̇xx
    τxy = 2 * ηv .* ε̇xy

    fx  = (τxx[2,2] - τxx[1,2]) * invΔx 
    fx += (τxy[2,2] - τxy[2,1]) * invΔy 
    fx -= ( Pt[2,2] -  Pt[1,2]) * invΔx
    fx *= -1#Δ.x*Δ.y

    return fx
end

function Momentum_y(Vx, Vy, Pt, Pf, η, type, bcv, Δ)
    
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y
    
    for i=1:4
        if type.x[i,1] == :Dirichlet_tangent 
            Vx[i,1] = fma(2, bcv.x[i,1], -Vx[i,2])
        elseif type.x[i,1] == :Neumann_tangent
            Vx[i,1] = fma(Δ.y, bcv.x[i,1], Vx[i,2])
        end
        if type.x[i,4] == :Dirichlet_tangent 
            Vx[i,4] = fma(2, bcv.x[i,4], -Vx[i,3])
        elseif type.x[i,4] == :Neumann_tangent
            Vx[i,4] = fma(Δ.y, bcv.x[i,4], Vx[i,3])
        end
    end

    for j=1:3
        if type.y[1,j] == :Dirichlet_tangent 
            Vy[1,j] = fma(2, bcv.y[1,j], -Vy[2,j])
        elseif type.y[1,j] == :Neumann_tangent
            Vy[1,j] = fma(Δ.x, bcv.y[1,j], Vy[2,j])
        end
        if type.y[end,j] == :Dirichlet_tangent 
            Vy[end,j] = fma(2, bcv.y[end,j], -Vy[end-1,j])
        elseif type.y[end,j] == :Neumann_tangent
            Vy[end,j] = fma(Δ.x, bcv.y[end,j], Vy[end-1,j])
        end
    end
     
    Dxx = (Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1]) * invΔx             # Static Arrays ???
    Dyy = (Vy[:,2:end] - Vy[:,1:end-1]) * invΔy             
    Dkk = Dxx + Dyy

    Dxy = (Vx[2:end-1,2:end] - Vx[2:end-1,1:end-1]) * invΔy 
    Dyx = (Vy[2:end,:] - Vy[1:end-1,:]) * invΔx 

    ε̇xx = Dxx - 1/3*Dkk
    ε̇yy = Dyy - 1/3*Dkk
    ε̇xy = 1/2 * ( Dxy + Dyx ) 

    ηc = 0.25*(η.x[1:end-1,2:end-1] .+ η.x[2:end,2:end-1] .+ η.y[:,1:end-1] .+ η.y[:,2:end])
    ηv = 0.25*(η.x[2:end-1,1:end-1] .+ η.x[2:end-1,2:end] .+ η.y[1:end-1,:] .+ η.y[2:end,:])

    τyy = 2 * ηc .* ε̇yy
    τxy = 2 * ηv .* ε̇xy

    fy  = (τyy[2,2] - τyy[2,1]) * invΔy 
    fy += (τxy[2,2] - τxy[1,2]) * invΔx 
    fy -= (Pt[2,2] - Pt[2,1]) * invΔy
    fy *= -1#Δ.x*Δ.y

    return fy
end

function Continuity(Vx, Vy, Pt, Pf, ηΦ, ϕ, type_loc, bcv_loc, Δ)
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y
    fp = ((Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy + (Pt[1] - Pf[2,2])/((1-ϕ)*ηΦ))
    # fp *= η/(Δ.x+Δ.y)
    return fp
end

function FluidContinuity(Vx, Vy, Pt, Pf, ηΦ, ϕ, kμ, type_loc, bcv_loc, Δ)
    
    PfC       = Pf[2,2]

    if type_loc[1,2] === :Dirichlet
        PfW = 2*bcv_loc[1,2] - PfC
    elseif type_loc[1,2] === :Neumann
        PfW = Δ.x*bcv_loc[1,2] + PfC
    elseif type_loc[1,2] === :periodic || type_loc[1,2] === :in || type_loc[1,2] === :constant
        PfW = Pf[1,2] 
    else
        PfW =  1.
    end

    if type_loc[3,2] === :Dirichlet
        PfE = 2*bcv_loc[3,2] - PfC
    elseif type_loc[3,2] === :Neumann
        PfE = -Δ.x*bcv_loc[3,2] + PfC
    elseif type_loc[3,2] === :periodic || type_loc[3,2] === :in || type_loc[3,2] === :constant
        PfE = Pf[3,2] 
    else
        PfE =  1.
    end

    if type_loc[2,1] === :Dirichlet
        PfS = 2*bcv_loc[2,1] - PfC
    elseif type_loc[2,1] === :Neumann
        PfS = Δ.y*bcv_loc[2,1] + PfC
    elseif type_loc[2,1] === :periodic || type_loc[2,1] === :in || type_loc[2,1] === :constant
        PfS = Pf[2,1] 
    else
        PfS =  1.
    end

    if type_loc[2,3] === :Dirichlet
        PfN = 2*bcv_loc[2,3] - PfC
    elseif type_loc[2,3] === :Neumann
        PfN = -Δ.y*bcv_loc[2,3] + PfC
    elseif type_loc[2,3] === :periodic || type_loc[2,3] === :in || type_loc[2,3] === :constant
        PfN = Pf[2,3] 
    else
        PfN =  1.
    end

    # if type_loc[2,2] != :constant
        qxW = -kμ.xx[1]*(PfC - PfW)/Δ.x
        qxE = -kμ.xx[2]*(PfE - PfC)/Δ.x
        qyS = -kμ.yy[1]*(PfC - PfS)/Δ.y
        qyN = -kμ.yy[2]*(PfN - PfC)/Δ.y
        F   = (qxE - qxW)/Δ.x + (qyN - qyS)/Δ.y - (Pt[1]-Pf[2,2])/((1-ϕ)*ηΦ)
    # else
    #     F = 0
    # end
    return F
end

function ResidualMomentum2D_x!(R, V, P, rheo, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        Vx_loc     = MMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vy_loc     = MMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        ηx_loc     = SMatrix{3,3}( rheo.η.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ηy_loc     = SMatrix{4,4}( rheo.η.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        Pt_loc     = SMatrix{2,3}(      P.t[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        Pf_loc     = SMatrix{2,3}(      P.f[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        η_loc      = (x=ηx_loc, y=ηy_loc)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        if type.Vx[i,j] == :in
            R.x[i,j]   = Momentum_x(Vx_loc, Vy_loc, Pt_loc, Pf_loc, η_loc, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_x!(K, V, P, rheo, num, pattern, type, BC, nc, Δ) 

    ∂R∂Vx = @MMatrix zeros(3,3)
    ∂R∂Vy = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(2,3)
    ∂R∂Pf = @MMatrix zeros(2,3)
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        Vx_loc     = MMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vy_loc     = MMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        ηx_loc     = SMatrix{3,3}( rheo.η.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ηy_loc     = SMatrix{4,4}( rheo.η.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        Pt_loc     = MMatrix{2,3}(      P.t[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        Pf_loc     = MMatrix{2,3}(      P.f[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        η_loc      = (x=ηx_loc, y=ηy_loc)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        if type.Vx[i,j] == :in
            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            ∂Vx, ∂Vy, ∂Pt, ∂Pf = ad_partial_gradients(Momentum_x, (Vx_loc, Vy_loc, Pt_loc, Pf_loc), η_loc, type_loc, bcv_loc, Δ)
            ∂R∂Vx .= ∂Vx
            ∂R∂Vy .= ∂Vy
            ∂R∂Pt .= ∂Pt
            ∂R∂Pf .= ∂Pf
            # Vx --- Vx
            Local = num.Vx[i-1:i+1,j-1:j+1] .* pattern[1][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][1][num.Vx[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
                end
            end
            # Vx --- Vy
            Local = num.Vy[i-1:i+2,j-2:j+1] .* pattern[1][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][2][num.Vx[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj]  
                end
            end
            # Vx --- Pt
            Local = num.Pt[i-1:i,j-2:j] .* pattern[1][3]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][3][num.Vx[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
                end
            end 
            # Vx --- Pf
            Local = num.Pf[i-1:i,j-2:j] .* pattern[1][4]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][4][num.Vx[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
                end
            end 
        end
    end
    return nothing
end

function ResidualMomentum2D_y!(R, V, P, rheo, number, type, BC, nc, Δ)                 
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = MMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = MMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ηx_loc     = SMatrix{4,4}( rheo.η.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        ηy_loc     = SMatrix{3,3}( rheo.η.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt_loc     = MMatrix{3,2}(      P.t[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Pf_loc     = MMatrix{3,2}(      P.f[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        η_loc      = (x=ηx_loc, y=ηy_loc)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        if type.Vy[i,j] == :in
            R.y[i,j]   = Momentum_y(Vx_loc, Vy_loc, Pt_loc, Pf_loc, η_loc, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_y!(K, V, P, rheo, num, pattern, type, BC, nc, Δ) 
    
    ∂R∂Vy = @MMatrix zeros(3,3)
    ∂R∂Vx = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(3,2)
    ∂R∂Pf = @MMatrix zeros(3,2)

    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = MMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = MMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ηx_loc     = SMatrix{4,4}( rheo.η.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        ηy_loc     = SMatrix{3,3}( rheo.η.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt_loc     = MMatrix{3,2}(      P.t[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Pf_loc     = MMatrix{3,2}(      P.f[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        η_loc      = (x=ηx_loc, y=ηy_loc)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        if type.Vy[i,j] == :in
            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            ∂Vx, ∂Vy, ∂Pt, ∂Pf = ad_partial_gradients(Momentum_y, (Vx_loc, Vy_loc, Pt_loc, Pf_loc), η_loc, type_loc, bcv_loc, Δ)
            ∂R∂Vx .= ∂Vx
            ∂R∂Vy .= ∂Vy
            ∂R∂Pt .= ∂Pt
            ∂R∂Pf .= ∂Pf
            # Vy --- Vx
            Local = num.Vx[i-2:i+1,j-1:j+2] .* pattern[2][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][1][num.Vy[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
                end
            end
            # Vy --- Vy
            Local = num.Vy[i-1:i+1,j-1:j+1] .* pattern[2][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][2][num.Vy[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj]  
                end
            end
            # Vy --- Pt
            Local = num.Pt[i-2:i,j-1:j] .* pattern[2][3]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][3][num.Vy[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
                end
            end 
            # Vy --- Pf
            Local = num.Pf[i-2:i,j-1:j] .* pattern[2][4]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][4][num.Vy[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
                end
            end       
        end
    end
    return nothing
end

function ResidualContinuity2D!(R, V, P, rheo, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    # (; bc_val, type, pattern, num) = numbering
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Pf_loc     = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcx_loc    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        bcy_loc    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        typex_loc  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        typey_loc  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        R.pt[i,j]  = Continuity(Vx_loc, Vy_loc, P.t[i,j], Pf_loc, rheo.ηΦ[i,j], rheo.ϕ[i,j], type_loc, bcv_loc, Δ)
    end
    return nothing
end

function AssembleContinuity2D!(K, V, P, rheo, num, pattern, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    ∂R∂Vx = @MMatrix zeros(3,2)
    ∂R∂Vy = @MMatrix zeros(2,3)
    ∂R∂Pt = @MMatrix zeros(1,1)
    ∂R∂Pf = @MMatrix zeros(3,3)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Pt_loc     = MMatrix{1,1}(      P.t[ii,jj] for ii in i:i, jj in j:j)
        Pf_loc     = MMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcx_loc    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        bcy_loc    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        typex_loc  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        typey_loc  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        
        ∂R∂Vx .= 0.
        ∂R∂Vy .= 0.
        ∂R∂Pt .= 0.
        ∂R∂Pf .= 0.
        ∂Vx, ∂Vy, ∂Pt, ∂Pf = ad_partial_gradients(Continuity, (Vx_loc, Vy_loc, Pt_loc, Pf_loc), rheo.ηΦ[i,j], rheo.ϕ[i,j], type_loc, bcv_loc, Δ)
        ∂R∂Vx .= ∂Vx
        ∂R∂Vy .= ∂Vy
        ∂R∂Pt .= ∂Pt
        ∂R∂Pf .= ∂Pf

        # Pt --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[3][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K[3][1][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
            end
        end
        # Pt --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[3][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K[3][2][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj] 
            end
        end
        # Pt --- Pt
        Local = num.Pt[i,j] .* pattern[3][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][3][num.Pt[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
            end
        end
        # Pt --- Pf
        Local = num.Pf[i-1:i+1,j-1:j+1] .* pattern[3][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][4][num.Pt[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
            end
        end
    end
    return nothing
end

function ResidualFluidContinuity2D!(R, V, P, rheo, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        if type.Pf[i,j] !== :constant 
            Pf_loc     = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            type_loc   = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcv_loc    = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
            Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
            k_loc_xx   = @SVector [rheo.kμf.x[i,j+1], rheo.kμf.x[i+1,j+1]]
            k_loc_yy   = @SVector [rheo.kμf.y[i+1,j], rheo.kμf.y[i+1,j+1]]
            k_loc      = (xx = k_loc_xx,    xy = 0.,
                          yx = 0.,          yy = k_loc_yy)
            R.pf[i,j]  = FluidContinuity(Vx_loc, Vy_loc, P.t[i,j], Pf_loc, rheo.ηΦ[i,j], rheo.ϕ[i,j], k_loc, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleFluidContinuity2D!(K, V, P, rheo, num, pattern, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    ∂R∂Vx = @MMatrix zeros(3,2)
    ∂R∂Vy = @MMatrix zeros(2,3)
    ∂R∂Pt = @MMatrix zeros(1,1)
    ∂R∂Pf = @MMatrix zeros(3,3)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Pt_loc     = MMatrix{1,1}(      P.t[ii,jj] for ii in i:i, jj in j:j)
        Pf_loc     = MMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc   = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcv_loc    = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        k_loc_xx   = @SVector [rheo.kμf.x[i,j+1], rheo.kμf.x[i+1,j+1]]
        k_loc_yy   = @SVector [rheo.kμf.y[i+1,j], rheo.kμf.y[i+1,j+1]]
        k_loc      = (xx = k_loc_xx,    xy = 0.,
                      yx = 0.,          yy = k_loc_yy)

        ∂R∂Vx .= 0.
        ∂R∂Vy .= 0.
        ∂R∂Pt .= 0.
        ∂R∂Pf .= 0.
        ∂Vx, ∂Vy, ∂Pt, ∂Pf = ad_partial_gradients(FluidContinuity, (Vx_loc, Vy_loc, Pt_loc, Pf_loc), rheo.ηΦ[i,j], rheo.ϕ[i,j], k_loc, type_loc, bcv_loc, Δ)
        ∂R∂Vx .= ∂Vx
        ∂R∂Vy .= ∂Vy
        ∂R∂Pt .= ∂Pt
        ∂R∂Pf .= ∂Pf
             
        # Pf --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[4][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pf[i,j]>0
                K[4][1][num.Pf[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
            end
        end
        # Pf --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[4][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pf[i,j]>0
                K[4][2][num.Pf[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj] 
            end
        end
        # Pf --- Pt
        Local = num.Pt[i,j] .* pattern[4][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][3][num.Pf[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
            end
        end
        # Pf --- Pf
        Local = num.Pf[i-1:i+1,j-1:j+1] .* pattern[4][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][4][num.Pf[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
            end
        end
           
    end
    return nothing
end

function SetBCVx1(Vx, typex, bcx, Δ)

    MVx = MMatrix(Vx)
    # N/S
    for ii in axes(typex, 1)
        if typex[ii,1] == :Dirichlet_tangent
            MVx[ii,1] = fma(2, bcx[ii,1], -Vx[ii,2])
        elseif typex[ii,1] == :Neumann_tangent
            MVx[ii,1] = fma(Δ.y, bcx[ii,1], Vx[ii,2])
        end

        if typex[ii,end] == :Dirichlet_tangent
            MVx[ii,end] = fma(2, bcx[ii,end], -Vx[ii,end-1])
        elseif typex[ii,end] == :Neumann_tangent
            MVx[ii,end] = fma(Δ.y, bcx[ii,end], Vx[ii,end-1])
        end
    end
    # E/W
    for jj in axes(typex, 2)
        if typex[1,jj] == :Neumann_normal
            MVx[1,jj] = fma(2, Δ.x*bcx[1,jj], Vx[2,jj])
        end
        if typex[end,jj] == :Neumann_normal
            MVx[end,jj] = fma(2,-Δ.x*bcx[end,jj], Vx[end-1,jj])
        end
    end
    return SMatrix(MVx)
end

function SetBCVy1(Vy, typey, bcy, Δ)
    MVy = MMatrix(Vy)
    # E/W
    for jj in axes(typey, 2)
        if typey[1,jj] == :Dirichlet_tangent
            MVy[1,jj] = fma(2, bcy[1,jj], -Vy[2,jj])
        elseif typey[1,jj] == :Neumann_tangent
            MVy[1,jj] = fma(Δ.y, bcy[1,jj], Vy[2,jj])
        end

        if typey[end,jj] == :Dirichlet_tangent
            MVy[end,jj] = fma(2, bcy[end,jj], -Vy[end-1,jj])
        elseif typey[end,jj] == :Neumann_tangent
            MVy[end,jj] = fma(Δ.y, bcy[end,jj], Vy[end-1,jj])
        end
    end
    # N/S
    for ii in axes(typey, 1)
        if typey[ii,1] == :Neumann_normal
            MVy[ii,1] = fma(2, Δ.y*bcy[ii,1], Vy[ii,2])
        end
        if typey[ii,end] == :Neumann_normal
            MVy[ii,end] = fma(2,-Δ.y*bcy[ii,end], Vy[ii,end-1])
        end
    end
    return SMatrix(MVy)
end
