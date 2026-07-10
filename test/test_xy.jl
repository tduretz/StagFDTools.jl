using Test, StaticArrays, StagFDTools
using StagFDTools.Stokes: Ranges
using Statistics: mean

function SetBCVx!(Vx_loc, bcx_loc, bcv, Δ)

    for ii in axes(Vx_loc, 1)

        # Set Vx boundaries at S (this must be done 1st)
        if bcx_loc[ii,begin] == :Neumann 
            Vx_loc[ii,begin] =  Vx_loc[ii,begin+1] - Δ.y*bcv.∂Vx∂y_BC[ii,1]
        elseif bcx_loc[ii,begin] == :Dirichlet 
            Vx_loc[ii,begin] = -Vx_loc[ii,begin+1] + 2*bcv.Vx_BC[ii,1]
        end
        if bcx_loc[ii,begin] == :out 
            if bcx_loc[ii,begin+1] == :Neumann
                Vx_loc[ii,begin+1] =  Vx_loc[ii,begin+2] -   Δ.y*bcv.∂Vx∂y_BC[ii,1]
                Vx_loc[ii,begin]   =  Vx_loc[ii,begin+3] - 3*Δ.y*bcv.∂Vx∂y_BC[ii,1] 
            elseif bcx_loc[ii,begin+1] == :Dirichlet
                Vx_loc[ii,begin+1] = -Vx_loc[ii,begin+2] + 2*bcv.Vx_BC[ii,1]
                Vx_loc[ii,begin]   = -Vx_loc[ii,begin+3] + 2*bcv.Vx_BC[ii,1] 
            end
        end

        # Set Vx boundaries at N (this must be done 1st)
        if bcx_loc[ii,end] == :Neumann 
            Vx_loc[ii,end] =  Vx_loc[ii,end-1] + Δ.y*bcv.∂Vx∂y_BC[ii,2] 
        elseif bcx_loc[ii,end] == :Dirichlet 
            Vx_loc[ii,end] = -Vx_loc[ii,end-1] + 2*bcv.Vx_BC[ii,2]
        end
        if bcx_loc[ii,end] == :out
            if bcx_loc[ii,end-1] == :Neumann
                Vx_loc[ii,end-1] =  Vx_loc[ii,end-2] +   Δ.y*bcv.∂Vx∂y_BC[ii,2] 
                Vx_loc[ii,end]   =  Vx_loc[ii,end-3] + 3*Δ.y*bcv.∂Vx∂y_BC[ii,2]   
            elseif bcx_loc[ii,3] == :Dirichlet
                Vx_loc[ii,end-1] = -Vx_loc[ii,end-2] + 2*bcv.Vx_BC[ii,2] 
                Vx_loc[ii,end]   = -Vx_loc[ii,end-3] + 2*bcv.Vx_BC[ii,2]  
            end
        end
    end

    for jj in axes(Vx_loc, 2)
        # Set Vx boundaries at W (this must be done 2nd)
        if bcx_loc[1,jj] == :out
            Vx_loc[1,jj] = Vx_loc[2,jj] - Δ.x*bcv.∂Vx∂x_BC[1,jj] 
        end
        # Set Vx boundaries at E (this must be done 2nd)
        if bcx_loc[3,jj] == :out
            Vx_loc[3,jj] = Vx_loc[2,jj] + Δ.x*bcv.∂Vx∂x_BC[2,jj] 
        end
    end
end

function SetBCVy!(Vy_loc, bcy_loc, bcv, Δ)
    
    for jj in axes(Vy_loc, 2)

        # Set Vy boundaries at W (this must be done 1st)
        if bcy_loc[begin,jj] == :Neumann 
            Vy_loc[begin,jj] =  Vy_loc[begin+1,jj] - Δ.x*bcv.∂Vy∂x_BC[1,jj] 
        elseif bcy_loc[begin,jj] == :Dirichlet 
            Vy_loc[begin,jj] = -Vy_loc[begin+1,jj] + 2*bcv.Vy_BC[1,jj]
        end
        if bcy_loc[begin,jj] == :out
            if bcy_loc[begin+1,jj] == :Neumann 
                Vy_loc[begin+1,jj] = Vy_loc[begin+2,jj] -   Δ.y*bcv.∂Vy∂x_BC[1,jj] 
                Vy_loc[begin,jj]   = Vy_loc[begin+3,jj] - 3*Δ.y*bcv.∂Vy∂x_BC[1,jj] 
            elseif bcy_loc[begin+1,jj] == :Dirichlet
                Vy_loc[begin+1,jj] = -Vy_loc[begin+2,jj] + 2*bcv.Vy_BC[1,jj]
                Vy_loc[begin,jj]   = -Vy_loc[begin+3,jj] + 2*bcv.Vy_BC[1,jj]
            end 
        end

        # Set Vy boundaries at E (this must be done 1st)
        if bcy_loc[end,jj] == :Neumann 
            Vy_loc[end,jj] = Vy_loc[end-1,jj] + Δ.x*bcv.∂Vy∂x_BC[1,jj] 
        elseif bcy_loc[end,jj] == :Dirichlet 
            Vy_loc[end,jj] = -Vy_loc[end-1,jj] + 2*bcv.Vy_BC[2,jj]
        end
        if bcy_loc[end,jj] == :out
            if bcy_loc[end-1,jj] == :Neumann 
                Vy_loc[end-1,jj] = Vy_loc[end-2,jj] +   Δ.y*bcv.∂Vy∂x_BC[1,jj]
                Vy_loc[end,jj]   = Vy_loc[end-3,jj] + 3*Δ.y*bcv.∂Vy∂x_BC[1,jj]
            elseif bcy_loc[3,jj] == :Dirichlet 
                Vy_loc[end-1,jj] = -Vy_loc[end-2,jj] + 2*bcv.Vy_BC[2,jj]
                Vy_loc[end,jj]   = -Vy_loc[end-3,jj] + 2*bcv.Vy_BC[2,jj]
            end
        end
    end

    for ii in axes(Vy_loc, 1)
        # Set Vy boundaries at S (this must be done 2nd)
        if bcy_loc[ii,1] == :out
            Vy_loc[ii,1] = Vy_loc[ii,2] - Δ.y*bcv.∂Vy∂y_BC[ii,1]
        end
        # Set Vy boundaries at S (this must be done 2nd)
        if bcy_loc[ii,3] == :out
            Vy_loc[ii,3] = Vy_loc[ii,2] + Δ.y*bcv.∂Vy∂y_BC[ii,2]
        end
    end
end

function TestShearStrainRate(D_BC)
    #--------------------------------------------#
    # Resolution
    nc = (x = 10, y = 10)

    (; inx_Vx, iny_Vx, inx_Vy, iny_Vy, size_x, size_y) = Ranges(nc)

    #--------------------------------------------#
    # Boundary conditions

    # Define node types and set BC flags
    type = (
        Vx = fill(:out, (nc.x+3, nc.y+4)),
        Vy = fill(:out, (nc.x+4, nc.y+3)),
        Pt = fill(:out, (nc.x+2, nc.y+2)),
        xy = fill(:out, (nc.x+1, nc.y+1)),
    )

    type.xy                  .= :τxy 
    type.xy[2:end-1,2:end-1] .= :in 

    # -------- Vx -------- #
    type.Vx[inx_Vx,iny_Vx] .= :in       
    type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
    type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
    type.Vx[inx_Vx,2]       .= :Neumann
    type.Vx[inx_Vx,end-1]   .= :Neumann
    type.Vx[inx_Vx,2]       .= :Dirichlet
    type.Vx[inx_Vx,end-1]   .= :Dirichlet
    # -------- Vy -------- #
    type.Vy[inx_Vy,iny_Vy] .= :in       
    type.Vy[2,iny_Vy]       .= :Neumann
    type.Vy[end-1,iny_Vy]   .= :Neumann
    # type.Vy[2,iny_Vy]       .= :Dirichlet
    # type.Vy[end-1,iny_Vy]   .= :Dirichlet
    type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
    type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1] .= :in

    #--------------------------------------------#
    # Intialise field
    xmin, xmax = -1/2, 1/2
    ymin, ymax = -1/2, 1/2
    L  = (x=xmax-xmin, y=ymax-ymin)
    Δ  = (x=L.x/nc.x, y=L.y/nc.y)
    V  = (x=zeros(size_x...), y=zeros(size_y...))
    xv = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    xvx = LinRange(-L.x/2-Δ.x, L.x/2+Δ.x, nc.x+3)
    xvy = LinRange(-L.x/2-3Δ.x/2, L.x/2+3Δ.x/2, nc.x+4)
    yvy = LinRange(-L.y/2-Δ.y, L.y/2+Δ.y, nc.y+3)
    yvx = LinRange(-L.y/2-3Δ.y/2, L.y/2+3Δ.y/2, nc.y+4)

    # Velocity field
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'
    BC = (
        Vx    = zeros(size_x[1], 2),
        Vy    = zeros(2, size_y[2]),
        ∂Vx∂x = zeros(2, size_x[2]),
        ∂Vy∂y = zeros(size_y[1], 2),
        ∂Vx∂y = zeros(size_x[1], 2),
        ∂Vy∂x = zeros(2, size_y[2]),
    )
    BC.Vx[:,1] .= xvx .* D_BC[1,1] .+ ymin .* D_BC[1,2]
    BC.Vx[:,2] .= xvx .* D_BC[1,1] .+ ymax .* D_BC[1,2]
    BC.Vy[1,:] .= yvy .* D_BC[2,2] .+ xmin .* D_BC[2,1]
    BC.Vy[2,:] .= yvy .* D_BC[2,2] .+ xmax .* D_BC[2,1]
    BC.∂Vx∂x[1,:] .= D_BC[1,1]
    BC.∂Vx∂x[2,:] .= D_BC[1,1]
    BC.∂Vx∂y[:,1] .= D_BC[1,2]
    BC.∂Vx∂y[:,2] .= D_BC[1,2]
    BC.∂Vy∂x[1,:] .= D_BC[2,1]
    BC.∂Vy∂x[2,:] .= D_BC[2,1]
    BC.∂Vy∂y[:,1] .= D_BC[2,2]
    BC.∂Vy∂y[:,2] .= D_BC[2,2]

    # Global array
    ε̇xy     = zeros(nc.x+1, nc.y+1)

    # Local arrays
    ε̇x̄ȳ_loc  = MMatrix{2,2}(zeros(2,2))
    ε̇xy_loc  = MMatrix{3,3}(zeros(3,3))

    for i in 1:nc.x+1, j in 1:nc.y+1

        #########################
        bcx_loc    = SMatrix{3,4}(type.Vx[i:i+2,j:j+3])
        bcy_loc    = SMatrix{4,3}(type.Vy[i:i+3,j:j+2])
        Vx_loc     = MMatrix{3,4}(V.x[i:i+2,j:j+3])
        Vy_loc     = MMatrix{4,3}(V.y[i:i+3,j:j+2])
        Vx_BC      = SMatrix{3,2}(BC.Vx[i:i+2,:])
        Vy_BC      = SMatrix{2,3}(BC.Vy[:,j:j+2])
        ∂Vx∂x_BC   = SMatrix{2,4}(BC.∂Vx∂x[:,j:j+3])
        ∂Vx∂y_BC   = SMatrix{3,2}(BC.∂Vx∂y[i:i+2,:])
        ∂Vy∂x_BC   = SMatrix{2,3}(BC.∂Vy∂x[:,j:j+2])
        ∂Vy∂y_BC   = SMatrix{4,2}(BC.∂Vy∂y[i:i+3,:])
        bcv_Vx     = (Vx_BC=Vx_BC, ∂Vx∂x_BC=∂Vx∂x_BC, ∂Vx∂y_BC=∂Vx∂y_BC)
        bcv_Vy     = (Vy_BC=Vy_BC, ∂Vy∂x_BC=∂Vy∂x_BC, ∂Vy∂y_BC=∂Vy∂y_BC)

        #########################
        SetBCVx!(Vx_loc, bcx_loc, bcv_Vx, Δ)

        #########################
        SetBCVy!(Vy_loc, bcy_loc, bcv_Vy, Δ)

        # ########################
        ε̇xy_loc .= 1/2* ( diff(Vx_loc, dims=2)/Δ.y + diff(Vy_loc, dims=1)/Δ.x ) 
        ε̇x̄ȳ_loc .= 1/4*(ε̇xy_loc[1:end-1,1:end-1] + ε̇xy_loc[2:end-0,1:end-1] + ε̇xy_loc[1:end-1,2:end-0] + ε̇xy_loc[2:end-0,2:end-0])
        ε̇xy[i,j] = 1/4*(ε̇x̄ȳ_loc[1:end-1,1:end-1] + ε̇x̄ȳ_loc[2:end-0,1:end-1] + ε̇x̄ȳ_loc[1:end-1,2:end-0] + ε̇x̄ȳ_loc[2:end-0,2:end-0])[1]

    end
     
    #--------------------------------------------#
    # p1 = heatmap(xv, yv, ε̇xy', aspect_ratio=1, xlim=extrema(xc))
    # display(plot(p1))

    printxy(ε̇xy)

    return mean(ε̇xy)    end

let
    # Pure Shear
    D_BC = [1  0;
    0 -1]
    TestShearStrainRate(D_BC) ≈ 0.5*(D_BC[1,2] + D_BC[2,1])
end

@testset "Shear strain rate" verbose=true begin

    # Pure Shear
    D_BC = [1  0;
            0 -1]
    @test TestShearStrainRate(D_BC) ≈ 0.5*(D_BC[1,2] + D_BC[2,1])

    # SimpleShearXY
    D_BC = [0 -1;
            0  0]
    @test TestShearStrainRate(D_BC) ≈ 0.5*(D_BC[1,2] + D_BC[2,1])

    # SimpleShearYX
    D_BC = [0  0;
            1  0]
    @test TestShearStrainRate(D_BC) ≈ 0.5*(D_BC[1,2] + D_BC[2,1])

    # AxialXX
    D_BC = [1  0;
            0  0]
    @test TestShearStrainRate(D_BC) ≈ 0.5*(D_BC[1,2] + D_BC[2,1])

    # AxialYY
    D_BC = [0  0;
            0  1]
    @test TestShearStrainRate(D_BC) ≈ 0.5*(D_BC[1,2] + D_BC[2,1])

end