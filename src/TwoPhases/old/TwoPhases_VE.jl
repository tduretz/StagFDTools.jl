function Continuity_VE(Vx, Vy, Pt, Pt0, Pf, Pf0, ηΦ, Kd, α, ϕ, type_loc, bcv_loc, Δ)
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y
    # fp = (Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy + (Pt[1] - Pf[2,2])/((1-ϕ)*ηΦ) + ((Pt[1]-Pt0[1])/Δ.t - α*(Pf[2,2]-Pf0[1])/Δ.t)/Kd
    
    
    # fp = (Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy  + (Pt[1] - Pf[2,2])/((1-ϕ)*ηΦ) + ((Pt[1]-Pt0[1])/Δ.t - α*(Pf[2,2]-Pf0[1])/Δ.t)/Kd

    fp = (Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy  + ((Pt[1]-Pt0[1]) - (Pf[2,2]-Pf0[1]))/Kd/Δ.t

    # fp *= η/(Δ.x+Δ.y)
    return fp
end

function FluidContinuity_VE(Vx, Vy, Pt, Pt0, Pf, Pf0, ηΦ, Kd, α, B, ϕ, kμ, type_loc, bcv_loc, Δ)
    
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

    qxW = -kμ.xx[1]*(PfC - PfW)/Δ.x
    qxE = -kμ.xx[2]*(PfE - PfC)/Δ.x
    qyS = -kμ.yy[1]*(PfC - PfS)/Δ.y
    qyN = -kμ.yy[2]*(PfN - PfC)/Δ.y
    # F   = (qxE - qxW)/Δ.x + (qyN - qyS)/Δ.y - (Pt[1]-Pf[2,2])/((1-ϕ)*ηΦ) - α/Kd*((Pt[1]-Pt0[1])/Δ.t - (Pf[2,2]-Pf0[1])/Δ.t/B)

    F   = (qxE - qxW)/Δ.x + (qyN - qyS)/Δ.y - 0*((Pt[1]-Pt0[1]) - (Pf[2,2]-Pf0[1]))/Kd/Δ.t


    # F   = (qxE - qxW)/Δ.x + (qyN - qyS)/Δ.y

    return F
end

function ResidualContinuity2D_VE!(R, V, P, P0, rheo, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    # (; bc_val, type, pattern, num) = numbering
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Pf_loc     = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf0        = MMatrix{1,1}(     P0.f[ii,jj] for ii in i:i, jj in j:j)
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcx_loc    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        bcy_loc    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        typex_loc  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        typey_loc  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        R.pt[i,j]  = Continuity_VE(Vx_loc, Vy_loc, P.t[i,j], P0.t[i,j], Pf_loc, Pf0, rheo.ηΦ[i,j], rheo.Kd[i,j], rheo.α[i,j], rheo.ϕ[i,j], type_loc, bcv_loc, Δ)
    end
    return nothing
end

function AssembleContinuity2D_VE!(K, V, P, P0, rheo, num, pattern, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    ∂R∂Vx = @MMatrix zeros(3,2)
    ∂R∂Vy = @MMatrix zeros(2,3)
    ∂R∂Pt = @MMatrix zeros(1,1)
    ∂R∂Pf = @MMatrix zeros(3,3)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Pt_loc     = MMatrix{1,1}(      P.t[ii,jj] for ii in i:i, jj in j:j)
        Pt0        = MMatrix{1,1}(     P0.t[ii,jj] for ii in i:i, jj in j:j)
        Pf_loc     = MMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf0        = MMatrix{1,1}(     P0.f[ii,jj] for ii in i:i, jj in j:j)
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
        ∂Vx, ∂Vy, ∂Pt, ∂Pf = ad_partial_gradients(Continuity_VE, (Vx_loc, Vy_loc, Pt_loc, Pf_loc), Pt0, Pf0, rheo.ηΦ[i,j], rheo.Kd[i,j], rheo.α[i,j], rheo.ϕ[i,j], type_loc, bcv_loc, Δ)
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

function ResidualFluidContinuity2D_VE!(R, V, P, P0, rheo, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        if type.Pf[i,j] !== :constant 
            Pf_loc     = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Pf0        = MMatrix{1,1}(     P0.f[ii,jj] for ii in i:i, jj in j:j)
            type_loc   = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcv_loc    = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
            Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
            k_loc_xx   = @SVector [rheo.kμf.x[i,j+1], rheo.kμf.x[i+1,j+1]]
            k_loc_yy   = @SVector [rheo.kμf.y[i+1,j], rheo.kμf.y[i+1,j+1]]
            k_loc      = (xx = k_loc_xx,    xy = 0.,
                          yx = 0.,          yy = k_loc_yy)
            R.pf[i,j]  = FluidContinuity_VE(Vx_loc, Vy_loc, P.t[i,j], P0.t[i,j], Pf_loc, Pf0, rheo.ηΦ[i,j], rheo.Kd[i,j], rheo.α[i,j], rheo.B[i,j], rheo.ϕ[i,j], k_loc, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleFluidContinuity2D_VE!(K, V, P, P0, rheo, num, pattern, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    ∂R∂Vx = @MMatrix zeros(3,2)
    ∂R∂Vy = @MMatrix zeros(2,3)
    ∂R∂Pt = @MMatrix zeros(1,1)
    ∂R∂Pf = @MMatrix zeros(3,3)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Pt_loc     = MMatrix{1,1}(      P.t[ii,jj] for ii in i:i, jj in j:j)
        Pt0        = MMatrix{1,1}(     P0.t[ii,jj] for ii in i:i, jj in j:j)
        Pf_loc     = MMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf0        = MMatrix{1,1}(     P0.f[ii,jj] for ii in i:i, jj in j:j)
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
        ∂Vx, ∂Vy, ∂Pt, ∂Pf = ad_partial_gradients(FluidContinuity_VE, (Vx_loc, Vy_loc, Pt_loc, Pf_loc), Pt0, Pf0, rheo.ηΦ[i,j], rheo.Kd[i,j], rheo.α[i,j], rheo.B[i,j], rheo.ϕ[i,j], k_loc, type_loc, bcv_loc, Δ)
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
