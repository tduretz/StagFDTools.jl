function SetBCVx1(Vx, typex, bcx, Δ)

    MVx = MMatrix(Vx)
    # N/S
    for ii in axes(typex, 1)
        if typex[ii,1] === :Dirichlet_tangent
            MVx[ii,1] = muladd(2, bcx[ii,1], -Vx[ii,2])
        elseif typex[ii,1] === :Neumann_tangent
            MVx[ii,1] = muladd(-Δ.y, bcx[ii,1], Vx[ii,2])
        end
        if typex[ii,end] === :Dirichlet_tangent
            MVx[ii,end] = muladd(2, bcx[ii,end], -Vx[ii,end-1])
        elseif typex[ii,end] === :Neumann_tangent
            MVx[ii,end] = muladd(Δ.y, bcx[ii,end], Vx[ii,end-1])
        end
        # Wide stencils (e.g. y-momentum at a free surface) reach 2 ghost layers
        if size(typex, 2) >= 4
            if typex[ii,1] === :out
                if typex[ii,2] === :Dirichlet_tangent
                    MVx[ii,2] = muladd(2, bcx[ii,2], -Vx[ii,3])
                    MVx[ii,1] = muladd(2, bcx[ii,2], -Vx[ii,4])
                elseif typex[ii,2] === :Neumann_tangent
                    MVx[ii,2] = muladd(  -Δ.y, bcx[ii,2], Vx[ii,3])
                    MVx[ii,1] = muladd(-3*Δ.y, bcx[ii,2], Vx[ii,4])
                end
            end
            if typex[ii,end] === :out
                if typex[ii,end-1] === :Dirichlet_tangent
                    MVx[ii,end-1] = muladd(2, bcx[ii,end-1], -Vx[ii,end-2])
                    MVx[ii,end  ] = muladd(2, bcx[ii,end-1], -Vx[ii,end-3])
                elseif typex[ii,end-1] === :Neumann_tangent
                    MVx[ii,end-1] = muladd(  Δ.y, bcx[ii,end-1], Vx[ii,end-2])
                    MVx[ii,end  ] = muladd(3*Δ.y, bcx[ii,end-1], Vx[ii,end-3])
                end
            end
        end
    end
    # E/W
    for jj in axes(typex, 2)
        if typex[1,jj] === :Neumann_normal 
            MVx[1,jj] = muladd(-2, Δ.x*bcx[2,jj], Vx[3,jj])
        elseif typex[1,jj] === :normal_stress # linear extrapolation
            MVx[1,jj] = 2*Vx[2,jj] - Vx[3,jj]
        end
        if typex[end,jj] === :Neumann_normal 
            MVx[end,jj] = muladd(2, Δ.x*bcx[end-1,jj], Vx[end-2,jj])
        elseif typex[end,jj] === :normal_stress # linear extrapolation
            MVx[end,jj] = 2*Vx[end-1,jj] - Vx[end-2,jj]
        end
    end
    return SMatrix(MVx)
end

function SetBCVy1(Vy, typey, bcy, Δ)
    MVy = MMatrix(Vy)
    # E/W
    for jj in axes(typey, 2)
        if typey[1,jj] === :Dirichlet_tangent
            MVy[1,jj] = muladd(2, bcy[1,jj], -Vy[2,jj])
        elseif typey[1,jj] === :Neumann_tangent
            MVy[1,jj] = muladd(-Δ.x, bcy[1,jj], Vy[2,jj])
        end

        if typey[end,jj] === :Dirichlet_tangent
            MVy[end,jj] = muladd(2, bcy[end,jj], -Vy[end-1,jj])
        elseif typey[end,jj] === :Neumann_tangent
            MVy[end,jj] = muladd(Δ.x, bcy[end,jj], Vy[end-1,jj])
        end
        # Wide stencils (e.g. x-momentum at a free E/W boundary) reach 2 ghost layers
        if size(typey, 1) >= 4
            if typey[1,jj] === :out
                if typey[2,jj] === :Dirichlet_tangent
                    MVy[2,jj] = muladd(2, bcy[2,jj], -Vy[3,jj])
                    MVy[1,jj] = muladd(2, bcy[2,jj], -Vy[4,jj])
                elseif typey[2,jj] === :Neumann_tangent
                    MVy[2,jj] = muladd(  -Δ.x, bcy[2,jj], Vy[3,jj])
                    MVy[1,jj] = muladd(-3*Δ.x, bcy[2,jj], Vy[4,jj])
                end
            end
            if typey[end,jj] === :out
                if typey[end-1,jj] === :Dirichlet_tangent
                    MVy[end-1,jj] = muladd(2, bcy[end-1,jj], -Vy[end-2,jj])
                    MVy[end  ,jj] = muladd(2, bcy[end-1,jj], -Vy[end-3,jj])
                elseif typey[end-1,jj] === :Neumann_tangent
                    MVy[end-1,jj] = muladd(  Δ.x, bcy[end-1,jj], Vy[end-2,jj])
                    MVy[end  ,jj] = muladd(3*Δ.x, bcy[end-1,jj], Vy[end-3,jj])
                end
            end
        end
    end
    # N/S
    for ii in axes(typey, 1)
        if typey[ii,1] === :Neumann_normal
            MVy[ii,1] = muladd(-2, Δ.y*bcy[ii,2], Vy[ii,3])
        elseif typey[ii,1] === :normal_stress # linear extrapolation
            MVy[ii,1] = 2*Vy[ii,2] - Vy[ii,3]
        end
        if typey[ii,end] === :Neumann_normal
            MVy[ii,end] = muladd(2, Δ.y*bcy[ii,end-1], Vy[ii,end-2])
        elseif typey[ii,end] === :normal_stress # linear extrapolation
            MVy[ii,end] = 2*Vy[ii,end-1] - Vy[ii,end-2]
        end
    end
    return SMatrix(MVy)
end

# Ghost cells carry a BC type; interior cells are :in (or :periodic, :constant, ...)
@inline is_bc_ghost(t) = t === :Dirichlet || t === :Neumann || t === :no_flux
# A stencil row (column) is a S/N (W/E) ghost row (column) only if all its cells are BC ghosts.
# The cells of a S/N ghost row are flagged as BC too: they must not be mistaken for W/E ghosts (and vice versa),
# otherwise vertex averages next to a boundary pick the wrong neighbour.
@inline ghost_row(type, jj) = all(is_bc_ghost, type[:, jj])
@inline ghost_col(type, ii) = all(is_bc_ghost, type[ii, :])

function SetBCPt1(Pt, type, bc, Δ, ρtg  )

    MPt =  MMatrix(Pt)
    S, N = ghost_row(type, 1), ghost_row(type, size(type, 2))
    W, E = ghost_col(type, 1), ghost_col(type, size(type, 1))

    # E/W: rows that are not S/N ghosts
    for jj in axes(type, 2)
        ((jj == 1 && S) || (jj == size(type, 2) && N)) && continue
        # West
        if W && type[1,jj] === :Dirichlet
            MPt[1,jj] = muladd(2, bc[1,jj], - Pt[2,jj])
        elseif W && type[1,jj] === :Neumann
            MPt[1,jj] = Pt[2,jj]
        end

        # East
        if E && type[end,jj] === :Dirichlet
            MPt[end,jj] = muladd(2, bc[end,jj], - Pt[end-1,jj])
        elseif E && type[end,jj] === :Neumann
            MPt[end,jj] = Pt[end-1,jj]
        end
    end

    # N/S: all columns, from the E/W-corrected values (sets the corners)
    for ii in axes(type, 1)
        # South
        if S && type[ii,1] === :Dirichlet
            MPt[ii,1] = muladd(2, bc[ii,1], -MPt[ii,2])
        elseif S && type[ii,1] === :Neumann
            MPt[ii,1] = MPt[ii,2]
        end

        # North
        if N && type[ii,end] === :Dirichlet
            MPt[ii,end] = muladd(2, bc[ii,end], -MPt[ii,end-1])
        elseif N && type[ii,end] === :Neumann
            MPt[ii,end] = MPt[ii,end-1]
        end
    end

    return SMatrix(MPt)
end

function SetBCPf1(Pf, type, bc, Δ, ρfg  )

    MPf =  MMatrix(Pf)
    S, N = ghost_row(type, 1), ghost_row(type, size(type, 2))
    W, E = ghost_col(type, 1), ghost_col(type, size(type, 1))

    # E/W: rows that are not S/N ghosts
    for jj in axes(type, 2)
        ((jj == 1 && S) || (jj == size(type, 2) && N)) && continue
        # West
        if W && type[1,jj] === :Dirichlet
            MPf[1,jj] = muladd(2, bc[1,jj], - Pf[2,jj])
        elseif W && type[1,jj] === :Neumann
            MPf[1,jj] = muladd(Δ.x, bc[1,jj], Pf[2,jj])
        end

        # East
        if E && type[end,jj] === :Dirichlet
            MPf[end,jj] = muladd(2, bc[end,jj], - Pf[end-1,jj])
        elseif E && type[end,jj] === :Neumann
            MPf[end,jj] = muladd(-Δ.x, bc[end,jj], Pf[end-1,jj])
        end
    end

    # N/S: all columns, from the E/W-corrected values (sets the corners)
    for ii in axes(type, 1)
        # South
        if S && type[ii,1] === :Dirichlet
            MPf[ii,1] = muladd(2, bc[ii,1], -MPf[ii,2])
        elseif S && type[ii,1] === :Neumann 
            MPf[ii,1] = muladd(Δ.y, bc[ii,1], MPf[ii,2])
        elseif S && type[ii,1] === :no_flux
            MPf[ii,1] = MPf[ii,2] - ρfg[1]*Δ.y
        end

        # North
        if N && type[ii,end] === :Dirichlet
            MPf[ii,end] = muladd(2, bc[ii,end], -MPf[ii,end-1])
        elseif N && type[ii,end] === :Neumann
            MPf[ii,end] = muladd(-Δ.y, bc[ii,end], MPf[ii,end-1])
        elseif N && type[ii,end] === :no_flux
            MPf[ii,end] = MPf[ii,end-1] + ρfg[end]*Δ.y
        end
    end

    return SMatrix(MPf)
end














function SetBCVx!(Vx_loc, bcx_loc, bcv, Δ)

    for ii in axes(Vx_loc, 1)

        # Set Vx boundaries at S (this must be done 1st)
        if bcx_loc[ii,begin] === :Neumann 
            Vx_loc[ii,begin] =  Vx_loc[ii,begin+1] - Δ.y*bcv.∂Vx∂y_BC[ii,1]
        elseif bcx_loc[ii,begin] === :Dirichlet 
            Vx_loc[ii,begin] = -Vx_loc[ii,begin+1] + 2*bcv.Vx_BC[ii,1]
        end
        if bcx_loc[ii,begin] === :out 
            if bcx_loc[ii,begin+1] === :Neumann
                Vx_loc[ii,begin+1] =  Vx_loc[ii,begin+2] -   Δ.y*bcv.∂Vx∂y_BC[ii,1]
                Vx_loc[ii,begin]   =  Vx_loc[ii,begin+3] - 3*Δ.y*bcv.∂Vx∂y_BC[ii,1] 
            elseif bcx_loc[ii,begin+1] === :Dirichlet
                Vx_loc[ii,begin+1] = -Vx_loc[ii,begin+2] + 2*bcv.Vx_BC[ii,1]
                Vx_loc[ii,begin]   = -Vx_loc[ii,begin+3] + 2*bcv.Vx_BC[ii,1] 
            end
        end

        # Set Vx boundaries at N (this must be done 1st)
        if bcx_loc[ii,end] === :Neumann 
            Vx_loc[ii,end] =  Vx_loc[ii,end-1] + Δ.y*bcv.∂Vx∂y_BC[ii,2] 
        elseif bcx_loc[ii,end] === :Dirichlet 
            Vx_loc[ii,end] = -Vx_loc[ii,end-1] + 2*bcv.Vx_BC[ii,2]
        end
        if bcx_loc[ii,end] === :out
            if bcx_loc[ii,end-1] === :Neumann
                Vx_loc[ii,end-1] =  Vx_loc[ii,end-2] +   Δ.y*bcv.∂Vx∂y_BC[ii,2] 
                Vx_loc[ii,end]   =  Vx_loc[ii,end-3] + 3*Δ.y*bcv.∂Vx∂y_BC[ii,2]   
            elseif bcx_loc[ii,end-1] === :Dirichlet
                Vx_loc[ii,end-1] = -Vx_loc[ii,end-2] + 2*bcv.Vx_BC[ii,2] 
                Vx_loc[ii,end]   = -Vx_loc[ii,end-3] + 2*bcv.Vx_BC[ii,2]  
            end
        end
    end

    # for jj in axes(Vx_loc, 2)
    #     # Set Vx boundaries at W (this must be done 2nd)
    #     if bcx_loc[1,jj] === :out
    #         Vx_loc[1,jj] = Vx_loc[2,jj] - Δ.x*bcv.∂Vx∂x_BC[1,jj] 
    #     end
    #     # Set Vx boundaries at E (this must be done 2nd)
    #     if bcx_loc[3,jj] === :out
    #         Vx_loc[3,jj] = Vx_loc[2,jj] + Δ.x*bcv.∂Vx∂x_BC[2,jj] 
    #     end
    # end
end

function SetBCVy!(Vy_loc, bcy_loc, bcv, Δ)
    
    for jj in axes(Vy_loc, 2)

        # Set Vy boundaries at W (this must be done 1st)
        if bcy_loc[begin,jj] === :Neumann 
            Vy_loc[begin,jj] =  Vy_loc[begin+1,jj] - Δ.x*bcv.∂Vy∂x_BC[1,jj] 
        elseif bcy_loc[begin,jj] === :Dirichlet 
            Vy_loc[begin,jj] = -Vy_loc[begin+1,jj] + 2*bcv.Vy_BC[1,jj]
        end
        if bcy_loc[begin,jj] === :out
            if bcy_loc[begin+1,jj] === :Neumann 
                Vy_loc[begin+1,jj] = Vy_loc[begin+2,jj] -   Δ.x*bcv.∂Vy∂x_BC[1,jj] 
                Vy_loc[begin,jj]   = Vy_loc[begin+3,jj] - 3*Δ.x*bcv.∂Vy∂x_BC[1,jj] 
            elseif bcy_loc[begin+1,jj] === :Dirichlet
                Vy_loc[begin+1,jj] = -Vy_loc[begin+2,jj] + 2*bcv.Vy_BC[1,jj]
                Vy_loc[begin,jj]   = -Vy_loc[begin+3,jj] + 2*bcv.Vy_BC[1,jj]
            end 
        end

        # Set Vy boundaries at E (this must be done 1st)
        if bcy_loc[end,jj] === :Neumann 
            Vy_loc[end,jj] = Vy_loc[end-1,jj] + Δ.x*bcv.∂Vy∂x_BC[2,jj] 
        elseif bcy_loc[end,jj] === :Dirichlet 
            Vy_loc[end,jj] = -Vy_loc[end-1,jj] + 2*bcv.Vy_BC[2,jj]
        end
        if bcy_loc[end,jj] === :out
            if bcy_loc[end-1,jj] === :Neumann 
                Vy_loc[end-1,jj] = Vy_loc[end-2,jj] +   Δ.x*bcv.∂Vy∂x_BC[2,jj]
                Vy_loc[end,jj]   = Vy_loc[end-3,jj] + 3*Δ.x*bcv.∂Vy∂x_BC[2,jj]
            elseif bcy_loc[end-1,jj] === :Dirichlet 
                Vy_loc[end-1,jj] = -Vy_loc[end-2,jj] + 2*bcv.Vy_BC[2,jj]
                Vy_loc[end,jj]   = -Vy_loc[end-3,jj] + 2*bcv.Vy_BC[2,jj]
            end
        end
    end

    # for ii in axes(Vy_loc, 1)
    #     # Set Vy boundaries at S (this must be done 2nd)
    #     if bcy_loc[ii,1] === :out
    #         Vy_loc[ii,1] = Vy_loc[ii,2] - Δ.y*bcv.∂Vy∂y_BC[ii,1]
    #     end
    #     # Set Vy boundaries at S (this must be done 2nd)
    #     if bcy_loc[ii,3] === :out
    #         Vy_loc[ii,3] = Vy_loc[ii,2] + Δ.y*bcv.∂Vy∂y_BC[ii,2]
    #     end
    # end
end
