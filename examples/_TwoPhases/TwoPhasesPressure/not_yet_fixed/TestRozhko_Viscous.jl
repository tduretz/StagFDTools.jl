using StagFDTools.TwoPhases, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf, JLD2
import Statistics:mean
using DifferentiationInterface
function Rozhko2008(rho, phi, r1, rc, P0, dPf, m, G, ν)
    eta   = (1-2*ν)/(1-ν)/2
    kappa = 3-4*ν
    if rho < r1
        Pf   = dPf
        Ux   = 0.
        Uy   = 0.
        Ur   = 0.
        Ut   = 0.
        Pt   = 0.
        Sxx  = 0.  
        Syy  = 0.  
        Sxy  = 0.  
    else
        Srr = (eta*rho^2*dPf*m^3*cos(2*phi)*log(1/(rho^32))+eta*rho^2*dPf*m^4*log(1/(rc^8))+eta*dPf*m^4*log(rho^8*rc^8)+eta*rho^6*dPf*log(rc^8)+eta*m^2*rho^4*P0*log(1/(rho^32))+eta*m^2*rho^4*dPf*log(rho^32)+eta*m*rho^6*P0*cos(2*phi)*log(rho^32)+eta*m^2*rho^2*dPf*log(1/(rc^8))+eta*rho^2*P0*m^4*log(rc^8)+eta*rho^8*dPf*log(1/rc^8*rho^8)+eta*rho^8*P0*log(1/rho^8*rc^8)+8*eta*P0*m^4+24*eta*m^2*rho^2*dPf+16*eta*m^2*rho^4*P0-16*eta*m^2*rho^4*dPf-8*eta*rho^6*dPf*m^2+8*eta*rho^6*P0*m^2-8*eta*dPf*m^4-24*eta*m^2*rho^2*P0+eta*rho^6*dPf*m^2*log(rc^8)+8*eta*rho^2*dPf*m^4-8*eta*rho^2*P0*m^4+8*eta*rho^2*dPf*m^3*cos(2*phi)-8*eta*m^3*dPf*cos(2*phi)+8*eta*m^3*P0*cos(2*phi)+eta*P0*m^4*log(1/(rho^8*rc^8))+eta*rho^6*P0*log(1/(rc^8))+24*eta*m*rho^4*P0*cos(2*phi)-8*eta*m^2*rho^2*P0*cos(4*phi)+8*eta*m^2*rho^4*P0*cos(4*phi)-24*eta*m*rho^6*P0*cos(2*phi)+8*eta*m^2*rho^2*dPf*cos(4*phi)-8*eta*m^2*rho^4*dPf*cos(4*phi)-8*eta*rho^2*P0*m^3*cos(2*phi)+eta*m^2*rho^4*dPf*log(rho^16)*cos(4*phi)+eta*m^2*rho^4*P0*log(1/(rho^16))*cos(4*phi)-24*eta*m*rho^4*dPf*cos(2*phi)+eta*rho^2*P0*m^3*cos(2*phi)*log(rho^32)+24*eta*m*rho^6*dPf*cos(2*phi)+eta*m^2*rho^2*P0*log(rc^8)+eta*rho^6*P0*m^2*log(1/(rc^8))+eta*m*rho^6*dPf*cos(2*phi)*log(1/(rho^32)))/(m^2*rho^4*cos(4*phi)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*phi)*log(1/(rc^32))+rho^2*m^3*cos(2*phi)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
        Stt = (eta*dPf*m^4*log(rho^8*rc^8)+eta*m^2*rho^4*P0*log(1/(rho^32))+eta*m^2*rho^4*dPf*log(rho^32)+eta*m^2*rho^2*dPf*log(rc^8)+eta*rho^6*P0*m^2*log(rc^8)+eta*rho^2*dPf*m^4*log(rc^8)+eta*m^2*rho^2*P0*log(1/(rc^8))+eta*rho^8*dPf*log(1/rc^8*rho^8)+eta*rho^8*P0*log(1/rho^8*rc^8)+16*eta*P0*m^4-24*eta*m^2*rho^2*dPf+16*eta*m^2*rho^4*P0-16*eta*m^2*rho^4*dPf+8*eta*rho^6*dPf*m^2-8*eta*rho^6*P0*m^2-16*eta*dPf*m^4+24*eta*m^2*rho^2*P0-8*eta*rho^2*dPf*m^4+8*eta*rho^2*P0*m^4+56*eta*rho^2*dPf*m^3*cos(2*phi)+8*eta*m^3*dPf*cos(2*phi)-8*eta*m^3*P0*cos(2*phi)+eta*P0*m^4*log(1/(rho^8*rc^8))-24*eta*m*rho^4*P0*cos(2*phi)+8*eta*m^2*rho^2*P0*cos(4*phi)+8*eta*m^2*rho^4*P0*cos(4*phi)+24*eta*m*rho^6*P0*cos(2*phi)-8*eta*m^2*rho^2*dPf*cos(4*phi)-8*eta*m^2*rho^4*dPf*cos(4*phi)-56*eta*rho^2*P0*m^3*cos(2*phi)+eta*m^2*rho^4*dPf*log(rho^16)*cos(4*phi)+eta*m^2*rho^4*P0*log(1/(rho^16))*cos(4*phi)+24*eta*m*rho^4*dPf*cos(2*phi)-24*eta*m*rho^6*dPf*cos(2*phi)+eta*rho^6*dPf*log(1/(rc^8))+eta*rho^6*dPf*m^2*log(1/(rc^8))+eta*rho^2*P0*m^3*cos(2*phi)*log(rho^32*rc^32)+eta*rho^6*P0*log(rc^8)+eta*rho^2*P0*m^4*log(1/(rc^8))+eta*m*rho^6*P0*cos(2*phi)*log(1/rc^32*rho^32)+8*eta*rho^8*dPf-8*eta*rho^8*P0+eta*m*rho^6*dPf*cos(2*phi)*log(1/rho^32*rc^32)+eta*rho^2*dPf*m^3*cos(2*phi)*log(1/(rho^32*rc^32)))/(m^2*rho^4*cos(4*phi)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*phi)*log(1/(rc^32))+rho^2*m^3*cos(2*phi)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
        Srt = eta*m*sin(2*phi)*(-2*rho^6*dPf*log(rc)+2*rho^2*log(rc)*P0*m^2-2*rho^4*log(rc)*P0*m^2+2*rho^4*dPf*log(rc)*m^2+2*m*dPf*rho^2*cos(2*phi)-2*m*P0*rho^2*cos(2*phi)-2*rho^4*dPf*m^2+2*rho^6*log(rc)*P0+2*rho^4*P0*m^2-2*m*rho^4*dPf*cos(2*phi)-m^2*dPf+m^2*P0-3*rho^2*P0*m^2+3*rho^4*P0-3*rho^4*dPf+3*rho^2*dPf*m^2-3*rho^6*P0+3*rho^6*dPf+2*m*rho^4*P0*cos(2*phi)+2*rho^4*dPf*log(rc)-2*rho^2*dPf*log(rc)*m^2-2*rho^4*log(rc)*P0)/log(rc)/(4*m^2*rho^4*cos(2*phi)^2+2*m^2*rho^4-4*rho^6*m*cos(2*phi)-4*rho^2*m^3*cos(2*phi)+rho^8+m^4);
                
        Ux  = -1/8*eta*r1*cos(phi)*(11*m*rho^4*dPf-11*m*rho^4*P0+kappa*rho^6*dPf+4*m^3*log(rho)*P0-4*m^3*log(rho)*dPf+5*rho^2*P0*m^2-kappa*rho^6*P0+4*rho^2*P0*m^3-3*kappa*m^3*dPf+3*kappa*m^3*P0-4*rho^2*dPf*m^3+12*m*P0*rho^2-12*rho^2*m*dPf-4*rho^2*m*log(rc)*P0+2*kappa*log(rc)*rho^6*P0+4*rho^2*dPf*log(rc)*m^3+4*kappa*log(rc)*m^3*dPf-20*rho^4*m*dPf*cos(phi)^2+6*rho^4*m*log(rc)*P0+20*m*P0*rho^4*cos(phi)^2+12*dPf*m^2*cos(phi)^2*rho^2-16*m*P0*cos(phi)^2*rho^2-12*P0*m^2*cos(phi)^2*rho^2+16*m*dPf*cos(phi)^2*rho^2+dPf*m^3+4*P0*m^2-rho^6*P0+rho^6*dPf-5*rho^2*dPf*m^2-8*kappa*log(rc)*m*P0*rho^4*cos(phi)^2+8*kappa*log(rc)*m^2*dPf*rho^2+2*kappa*log(rc)*m*P0*rho^4-2*kappa*log(rc)*m^2*P0*rho^2-4*kappa*rho^4*dPf*m*cos(phi)^2+4*kappa*log(rc)*m*dPf*rho^4+4*kappa*m*P0*rho^4*cos(phi)^2-8*rho^4*m*log(rc)*P0*cos(phi)^2+16*cos(phi)^2*dPf*rho^2*log(rho)*m^2+4*rho^4*dPf*m^2-4*rho^4*P0*m^2-5*kappa*rho^2*dPf*m^2+5*kappa*m^2*P0*rho^2+16*cos(phi)^2*m*log(rho)*dPf*rho^4-16*cos(phi)^2*rho^2*P0*log(rho)*m^2-16*cos(phi)^2*m*log(rho)*P0*rho^4+8*kappa*log(rc)*m^2*P0*cos(phi)^2*rho^2-16*kappa*log(rc)*m^2*dPf*cos(phi)^2*rho^2-12*kappa*m^2*P0*cos(phi)^2*rho^2-16*log(rc)*m^2*dPf*cos(phi)^2*rho^2+4*rho^6*dPf*log(rc)+4*rho^4*log(rc)*P0-4*rho^4*dPf*log(rc)+8*log(rc)*m^2*P0*cos(phi)^2*rho^2+12*kappa*m^2*dPf*cos(phi)^2*rho^2-2*rho^6*log(rc)*P0-4*dPf*m^2-4*rho^4*dPf*log(rc)*m^2+4*rho^4*log(rc)*P0*m^2+12*rho^2*dPf*log(rc)*m^2-6*rho^2*log(rc)*P0*m^2-P0*m^3+kappa*m*P0*rho^4-kappa*rho^4*dPf*m-12*dPf*rho^2*log(rho)*m^2+12*rho^2*P0*log(rho)*m^2-4*rho^2*log(rc)*P0*m^3-12*m*log(rho)*dPf*rho^4+12*m*log(rho)*P0*rho^4-2*kappa*log(rc)*m^3*P0+4*rho^2*m*dPf*log(rc)+4*rho^6*P0*log(rho)-4*dPf*rho^6*log(rho)+2*log(rc)*m^3*P0)/rho/log(rc)/G/(-m^2+4*m*rho^2*cos(phi)^2-2*m*rho^2-rho^4);       
        Uy  = -1/8*eta*r1*sin(phi)*(-9*m*rho^4*dPf+9*m*rho^4*P0-kappa*rho^6*dPf+4*m^3*log(rho)*P0-4*m^3*log(rho)*dPf+7*rho^2*P0*m^2+kappa*rho^6*P0+4*rho^2*P0*m^3-3*kappa*m^3*dPf+3*kappa*m^3*P0-4*rho^2*dPf*m^3-4*m*P0*rho^2+4*rho^2*m*dPf-4*rho^2*m*log(rc)*P0-2*kappa*log(rc)*rho^6*P0+4*rho^2*dPf*log(rc)*m^3+4*kappa*log(rc)*m^3*dPf+20*rho^4*m*dPf*cos(phi)^2-2*rho^4*m*log(rc)*P0-20*m*P0*rho^4*cos(phi)^2+12*dPf*m^2*cos(phi)^2*rho^2+16*m*P0*cos(phi)^2*rho^2-12*P0*m^2*cos(phi)^2*rho^2-16*m*dPf*cos(phi)^2*rho^2+dPf*m^3-4*P0*m^2+rho^6*P0-rho^6*dPf-7*rho^2*dPf*m^2+8*kappa*log(rc)*m*P0*rho^4*cos(phi)^2+8*kappa*log(rc)*m^2*dPf*rho^2-6*kappa*log(rc)*m*P0*rho^4-6*kappa*log(rc)*m^2*P0*rho^2+4*kappa*rho^4*dPf*m*cos(phi)^2+4*kappa*log(rc)*m*dPf*rho^4-4*kappa*m*P0*rho^4*cos(phi)^2+8*rho^4*m*log(rc)*P0*cos(phi)^2+16*cos(phi)^2*dPf*rho^2*log(rho)*m^2-4*rho^4*dPf*m^2+4*rho^4*P0*m^2-7*kappa*rho^2*dPf*m^2+7*kappa*m^2*P0*rho^2-16*cos(phi)^2*m*log(rho)*dPf*rho^4-16*cos(phi)^2*rho^2*P0*log(rho)*m^2+16*cos(phi)^2*m*log(rho)*P0*rho^4+8*kappa*log(rc)*m^2*P0*cos(phi)^2*rho^2-16*kappa*log(rc)*m^2*dPf*cos(phi)^2*rho^2-12*kappa*m^2*P0*cos(phi)^2*rho^2-16*log(rc)*m^2*dPf*cos(phi)^2*rho^2-4*rho^6*dPf*log(rc)-4*rho^4*log(rc)*P0+4*rho^4*dPf*log(rc)+8*log(rc)*m^2*P0*cos(phi)^2*rho^2+12*kappa*m^2*dPf*cos(phi)^2*rho^2+2*rho^6*log(rc)*P0+4*dPf*m^2+4*rho^4*dPf*log(rc)*m^2-4*rho^4*log(rc)*P0*m^2+4*rho^2*dPf*log(rc)*m^2-2*rho^2*log(rc)*P0*m^2-P0*m^3+5*kappa*m*P0*rho^4-5*kappa*rho^4*dPf*m-4*dPf*rho^2*log(rho)*m^2+4*rho^2*P0*log(rho)*m^2-4*rho^2*log(rc)*P0*m^3+4*m*log(rho)*dPf*rho^4-4*m*log(rho)*P0*rho^4-2*kappa*log(rc)*m^3*P0+4*rho^2*m*dPf*log(rc)-4*rho^6*P0*log(rho)+4*dPf*rho^6*log(rho)+2*log(rc)*m^3*P0)/rho/log(rc)/G/(m^2-4*m*rho^2*cos(phi)^2+2*m*rho^2+rho^4);

        Ur  =  1/8*r1*eta*(-4*rho^2*dPf*log(rc)*m^2+4*m^2*log(rho)*dPf-4*m^2*log(rho)*P0-4*rho^2*log(rc)*m*P0*cos(2*phi)-2*rho^4*log(rc)*P0+4*rho^4*dPf*log(rc)+4*rho^2*log(rc)*P0*m^2+4*kappa*log(rc)*m*dPf*rho^2*cos(2*phi)+4*rho^2*log(rc)*m*dPf*cos(2*phi)+2*kappa*log(rc)*m^2*P0-4*kappa*log(rc)*m^2*dPf+2*kappa*log(rc)*rho^4*P0-4*m*P0*cos(2*phi)-4*kappa*rho^2*dPf*m*cos(2*phi)+4*kappa*rho^2*P0*m*cos(2*phi)-2*log(rc)*m^2*P0-3*kappa*P0*m^2+kappa*rho^4*dPf+4*rho^2*log(rc)*P0+3*kappa*dPf*m^2-4*dPf*rho^2*log(rc)-kappa*rho^4*P0+4*m*dPf*cos(2*phi)+P0*m^2-dPf*m^2-8*m*dPf*rho^2*cos(2*phi)+8*m*P0*rho^2*cos(2*phi)-4*rho^4*dPf*log(rho)+4*rho^4*P0*log(rho)-4*kappa*log(rc)*rho^2*P0*m*cos(2*phi)-rho^4*P0+rho^4*dPf-4*rho^2*P0*m^2+4*rho^2*dPf*m^2)/(-2*m*rho^2*cos(2*phi)+rho^4+m^2)^(1/2)/rho/G/log(rc);
        Ut  = -1/4*r1*eta*m*sin(2*phi)*(2*dPf*rho^2*log(rc)-kappa*rho^2*dPf+rho^2*kappa*P0+2*kappa*log(rc)*dPf*rho^2-rho^2*P0+rho^2*dPf-2*dPf+2*P0-4*rho^2*dPf*log(rho)+4*rho^2*P0*log(rho))/(-2*m*rho^2*cos(2*phi)+rho^4+m^2)^(1/2)/rho/G/log(rc);

        Pf  = P0 + dPf - dPf*log(rho)/log(rc);
        Sxx =  1/2*(((-2*rho.^2+1+rho.^4).*Srr+(-2*rho.^2-1-rho.^4).*Stt).*cos(2*phi)+(-2*rho.^2+1+rho.^4).*Srr+(2*rho.^2+1+rho.^4).*Stt+(-2*rho.^4 .*sin(2*phi)+2*sin(2*phi)).*Srt)./(-2*rho.^2 .*cos(2*phi)+rho.^4+1);
        Syy = -1/2*(((2*rho.^2+1+rho.^4).*Srr+(-rho.^4+2*rho.^2-1).*Stt).*cos(2*phi)+(-2*rho.^2-1-rho.^4).*Srr+(-rho.^4+2*rho.^2-1).*Stt+(-2*rho.^4 .*sin(2*phi)+2*sin(2*phi)).*Srt)./(-2*rho.^2 .*cos(2*phi)+rho.^4+1);
        Sxy =  1/2*((2+2*rho.^4).*Srt.*cos(2*phi)+(-sin(2*phi)+rho.^4 .*sin(2*phi)).*Srr+(sin(2*phi)-rho.^4 .*sin(2*phi)).*Stt-4*Srt.*rho.^2)./(-2*rho.^2 .*cos(2*phi)+rho.^4+1);
        Pt  = -1/2*(Sxx + Syy) 
    end   
    return (ux=Ux, uy=Uy, ur=Ur, ut=Ut, pt=Pt, pf=Pf, sxx=Sxx, syy=Syy, sxy=Sxy)
end

@views function main(nc, Ωl, Ωη)

    # Independant
    η0    = 1e0              # Shear viscosity
    ηsi    = 1.0       # Inclusion shear viscosity
    len    = 10.              # Box size
    ϕ0     = 1e-3
    # Dependant
    ν        = 0.25
    @show ηb0      = 3*η0*(1-2ν)/(2*(1+ν)*(1-2ν))

    @show ηb0      = 40*η0       # Bulk viscosity
    ηΦi      = ηb0*20000
    ηΦo      = ηb0/20000
    k_ηf0    = 1e0 # Permeability / fluid viscosity
    kμfi     = 1e1
    r_in     = 1.        # Inclusion radius 
    r_out    = 3.5*r_in
    ε̇        = 0.0    # Background strain rate
    G0,  Gi  = 1e25, 1e25*1e-3
    ν        = 0.49
    K        = 1*3*G0*(1-2ν)/(2*(1+ν)*(1-2ν))
    Ks0, Ksi = 1e20*K, 1e20*K*1e-3
    KΦ0, KΦi = 1e20*K, 1e20*K*1e-3
    Kf0, Kfi = 1e20*K, 1e20*K*1e-3

    # Set Rozhko values for fluid pressure
    G_anal = 1.0
    ν_anal = 0.49

    Pf_out = 0.    # Fluid pressure on external boundary, Pa
    # nondim 
    m      = 0.0   # 0 - circle, 0.5 - ellipse, 1 - cut 
    # dependent scales
    dPf   = 1.0   # Fluid pressure on cavity - Po    
    Δt0   = 1e0

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇ 0; 0 -ε̇] )
    
    # Resolution
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c = Ranges(nc)
    
    # Intialise field
    L   = (x=len, y=len)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    R   = (x=zeros(size_x...), y=zeros(size_y...), pt=zeros(size_c...), pf=zeros(size_c...))
    V   = (x=zeros(size_x...), y=zeros(size_y...))
    η   = (x= η0.*ones(size_x...), y= η0.*ones(size_y...), p = η0.*ones(size_c...) )
    ϕ   = ϕ0.*ones(size_c...) 
    ηΦ  = ηb0./(1. .-ϕ ).*ones(size_c...) 
    KΦ  = KΦ0.*ones(size_c...) 
    Ks  = Ks0.*ones(size_c...) 
    Kf  = Kf0.*ones(size_c...) 
    G   = (x= G0.*ones(size_x...), y= G0.*ones(size_y...), p = G0.*ones(size_c...) )

    kμf = (x= k_ηf0.*ones(size_x...), y= k_ηf0.*ones(size_y...))
    P   = (t=zeros(size_c...), f=zeros(size_c...))
    P0  = (t=zeros(size_c...), f=zeros(size_c...))
    xv  = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv  = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc  = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc  = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    xce = LinRange(-L.x/2-Δ.x/2, L.x/2+Δ.x/2, nc.x+2)
    yce = LinRange(-L.y/2-Δ.y/2, L.y/2+Δ.y/2, nc.y+2)
    xvx = LinRange(-L.x/2-Δ.x, L.x/2+Δ.x, nc.x+3)
    xvy = LinRange(-L.x/2-3Δ.x/2, L.x/2+3Δ.x/2, nc.x+4)
    yvy = LinRange(-L.y/2-Δ.y, L.y/2+Δ.y, nc.y+3)
    yvx = LinRange(-L.y/2-3Δ.y/2, L.y/2+3Δ.y/2, nc.y+4)

    # Define node types and set BC flags
    type = Fields(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    # -------- Vx -------- #
    type.Vx[inx_Vx,iny_Vx]  .= :in       
    type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
    type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
    type.Vx[inx_Vx,2]       .= :Dirichlet_tangent
    type.Vx[inx_Vx,end-1]   .= :Dirichlet_tangent
    # -------- Vy -------- #
    type.Vy[inx_Vy,iny_Vy]  .= :in       
    type.Vy[2,iny_Vy]       .= :Dirichlet_tangent
    type.Vy[end-1,iny_Vy]   .= :Dirichlet_tangent
    type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
    type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1] .= :in
    # -------- Pf -------- #
    type.Pf[2:end-1,2:end-1] .= :in
    type.Pf[1,:]             .= :Dirichlet 
    type.Pf[end,:]           .= :Dirichlet 
    type.Pf[:,1]             .= :Dirichlet
    type.Pf[:,end]           .= :Dirichlet
    # Add a constrant pressure within a circular region
    @views type.Pf[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= r_in^2] .= :constant
    @views type.Pf[inx_c, iny_c][(xc.^2 .+ (yc').^2) .>= r_out^2] .= :constant
    
    # @views type.Pt[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= r_in^2] .= :constant
    # @views type.Pt[inx_c, iny_c][(xc.^2 .+ (yc').^2) .>= r_out^2] .= :constant
    

    # Equation Fields
    number = Fields(
        fill(0, (nc.x+3, nc.y+4)),
        fill(0, (nc.x+4, nc.y+3)),
        fill(0, (nc.x+2, nc.y+2)),
        fill(0, (nc.x+2, nc.y+2)),
    )
    Numbering!(number, type, nc)

    # Stencil extent for each block matrix
    pattern = Fields(
        Fields(@SMatrix([0 1 0; 1 1 1; 0 1 0]),                 @SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]), @SMatrix([0 1 0;  0 1 0]),        @SMatrix([0 1 0;  0 1 0])), 
        Fields(@SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]),  @SMatrix([0 1 0; 1 1 1; 0 1 0]),                @SMatrix([0 0; 1 1; 0 0]),        @SMatrix([0 0; 1 1; 0 0])),
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]),                    @SMatrix([1])),
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]),                    @SMatrix([1 1 1; 1 1 1; 1 1 1])),
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    nPf   = maximum(number.Pf)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt), ExtendableSparseMatrix(nVx, nPf)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt), ExtendableSparseMatrix(nVy, nPf)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt), ExtendableSparseMatrix(nPt, nPf)),
        Fields(ExtendableSparseMatrix(nPf, nVx), ExtendableSparseMatrix(nPf, nVy), ExtendableSparseMatrix(nPf, nPt), ExtendableSparseMatrix(nPf, nPf)),
    )

    #--------------------------------------------#

    # Initial configuration
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'
    P.f[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= r_in^2]  .= dPf
    P.f[inx_c, iny_c][(xc.^2 .+ (yc').^2) .>= r_out^2] .= Pf_out

    KΦ[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= r_in^2] .= KΦi
    Ks[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= r_in^2] .= Ksi
    Kf[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= r_in^2] .= Kfi
    η.y[(xvy.^2 .+ (yvy').^2) .<= r_in^2] .= ηsi
    η.x[(xvx.^2 .+ (yvx').^2) .<= r_in^2] .= ηsi 
    G.y[(xvy.^2 .+ (yvy').^2) .<= r_in^2] .= Gi
    G.x[(xvx.^2 .+ (yvx').^2) .<= r_in^2] .= Gi 
    ηΦ[(xce.^2 .+ (yce').^2) .<= r_in^2]  .= ηΦi



    # kμf.x[(xvx.^2 .+ (yvx').^2) .<= r_in^2] .= kμfi
    # kμf.y[(xvy.^2 .+ (yvy').^2) .<= r_in^2] .= kμfi

    Ks[inx_c, iny_c][(xc.^2 .+ (yc').^2) .>= r_out^2] .= Ksi
    Kf[inx_c, iny_c][(xc.^2 .+ (yc').^2) .>= r_out^2] .= Kfi
    KΦ[inx_c, iny_c][(xc.^2 .+ (yc').^2) .>= r_out^2] .= KΦi
    η.y[(xvy.^2 .+ (yvy').^2) .>= r_out^2] .= ηsi
    η.x[(xvx.^2 .+ (yvx').^2) .>= r_out^2] .= ηsi 
    G.y[(xvy.^2 .+ (yvy').^2) .>= r_out^2] .= Gi
    G.x[(xvx.^2 .+ (yvx').^2) .>= r_out^2] .= Gi
    ηΦ[(xce.^2 .+ (yce').^2) .>= r_out^2]  .= ηΦo

    η.y .= 1 ./ (1. ./ η.y .+ 1. ./ (G.y*Δ.t))
    η.x .= 1 ./ (1. ./ η.x .+ 1. ./ (G.x*Δ.t))

    @show extrema(η.y)
    
    η.p .= 0.25.*(η.x[1:end-1,2:end-1].+η.x[2:end-0,2:end-1].+η.y[2:end-1,1:end-1].+η.y[2:end-1,2:end-0])
    Kd = (1-ϕ0) .* ( 1 ./ KΦ + 1 ./ Ks).^-1
    α  = 1 .- Kd ./ Ks
    B  = @. (Kd.^-1 - Ks.^-1) ./ (Kd.^-1 - Ks.^-1 + ϕ.*(Kf.^-1 .- Ks.^-1))

    rheo = (η=η, ηΦ=ηΦ, kμf=kμf, ϕ=ϕ, B=B, α=α, Kd=Kd)


    @show extrema(Ks)
    @show extrema(KΦ)
    @show extrema(Kd)
    @show extrema(α)
    @show extrema(B)
    @show extrema(η.p)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), Pf = zeros(size_c...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[1]  )
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[1]   .+ D_BC[2,2]*yv)
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[end] .+ D_BC[2,2]*yv)
    

    # preprocessing 
    xce  = LinRange(-L.x/2-Δ.x/2, L.x/2+Δ.x/2, nc.x+2)
    yce  = LinRange(-L.y/2-Δ.y/2, L.y/2+Δ.y/2, nc.y+2)

    Ur_ana = zero(BC.Pf)
    Ut_ana = zero(BC.Pf)
    Pf_ana = zero(BC.Pf)
    Pt_ana = zero(BC.Pf)
    ϵ_Ur   = zero(BC.Pf)
    ϵ_Pf   = zero(BC.Pf)

    for i=1:size(BC.Pf,1), j=1:size(BC.Pf,2)
        # coordinate transform
        ro  = sqrt(xce[i]^2 + yce[j]^2)
        phi = atan(yce[j], xce[i])
        sol = Rozhko2008(ro, phi, r_in, r_out, Pf_out, dPf, m, G_anal, ν_anal)
        BC.Pf[i,j]  = sol.pf
        Pf_ana[i,j] = sol.pf
        Pt_ana[i,j] = sol.pf
        Ur_ana[i,j] = sol.ur
        Ut_ana[i,j] = sol.ut
    end

    xce = LinRange(-L.x/2-Δ.x, L.x/2+Δ.x, nc.x+3)# nc.x+3, nc.y+4
    yce  = LinRange(-L.y/2-3*Δ.y/2, L.y/2+3*Δ.y/2, nc.y+4)
    for i=1:size(BC.Vx,1), j=1:size(BC.Vx,2)
        # coordinate transform
        ro  = sqrt(xce[i]^2 + yce[j]^2)
        phi = atan(yce[j], xce[i])
        sol = Rozhko2008(ro, phi, r_in, r_out, Pf_out, dPf, m, G_anal, ν_anal)
        # BC.Vx[i,j] = sol.ux
        # V.x[i,j]   = sol.ux
    end

    xce = LinRange(-L.x/2-3*Δ.x/2, L.x/2+3*Δ.x/2, nc.x+4)# nc.x+3, nc.y+4
    yce  = LinRange(-L.y/2-Δ.y, L.y/2+Δ.y, nc.y+3)
    for i=1:size(BC.Vy,1), j=1:size(BC.Vy,2)
        # coordinate transform
        ro  = sqrt(xce[i]^2 + yce[j]^2)
        phi = atan(yce[j], xce[i])
        sol = Rozhko2008(ro, phi, r_in, r_out, Pf_out, dPf, m, G_anal, ν_anal)
        # BC.Vy[i,j] = sol.uy
        # V.y[i,j]   = sol.uy
    end

    #--------------------------------------------#
    # Residual check
    ResidualMomentum2D_x!(R, V, P, rheo, number, type, BC, nc, Δ)
    ResidualMomentum2D_y!(R, V, P, rheo, number, type, BC, nc, Δ)

    ResidualContinuity2D_VE!(R, V, P, P0, rheo, number, type, BC, nc, Δ) 
    ResidualFluidContinuity2D_VE!(R, V, P, P0, rheo, number, type, BC, nc, Δ) 

    @info "Residuals"
    @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
    @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
    @show norm(R.pt[inx_c,iny_c])/sqrt(nPt)
    @show norm(R.pf[inx_c,iny_c])/sqrt(nPf)

    # Set global residual vector
    r = zeros(nVx + nVy + nPt + nPf)
    SetRHS!(r, R, number, type, nc)

    #--------------------------------------------#
    # Assembly
    @info "Assembly, ndof  = $(nVx + nVy + nPt + nPf)"
    AssembleMomentum2D_x!(M, V, P, rheo, number, pattern, type, BC, nc, Δ)
    AssembleMomentum2D_y!(M, V, P, rheo, number, pattern, type, BC, nc, Δ)
    
    AssembleContinuity2D_VE!(M, V, P, P0, rheo, number, pattern, type, BC, nc, Δ)
    AssembleFluidContinuity2D_VE!(M, V, P, P0, rheo, number, pattern, type, BC, nc, Δ)

    # Two-phases operator as block matrix
    𝑀 = [
        M.Vx.Vx M.Vx.Vy M.Vx.Pt M.Vx.Pf;
        M.Vy.Vx M.Vy.Vy M.Vy.Pt M.Vy.Pf;
        M.Pt.Vx M.Pt.Vy M.Pt.Pt M.Pt.Pf;
        M.Pf.Vx M.Pf.Vy M.Pf.Pt M.Pf.Pf;
    ]

    @info "System symmetry"
    𝑀diff = 𝑀 - 𝑀'
    dropzeros!(𝑀diff)
    @show norm(𝑀diff)

    #--------------------------------------------#
    # Direct solver 
    @time dx = - 𝑀 \ r

    #--------------------------------------------#
    UpdateSolution!(V, P, dx, number, type, nc)

    #--------------------------------------------#
    # Residual check
    ResidualMomentum2D_x!(R, V, P, rheo, number, type, BC, nc, Δ)
    ResidualMomentum2D_y!(R, V, P, rheo, number, type, BC, nc, Δ)
    ResidualContinuity2D_VE!(R, V, P, P0, rheo, number, type, BC, nc, Δ) 
    ResidualFluidContinuity2D_VE!(R, V, P, P0, rheo, number, type, BC, nc, Δ) 
    
    @info "Residuals"
    @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
    @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
    @show norm(R.pt[inx_c,iny_c])/sqrt(nPt)
    @show norm(R.pf[inx_c,iny_c])/sqrt(nPf)

    #--------------------------------------------#

    Vxsc = 0.5*(V.x[1:end-1,2:end-1] + V.x[2:end,2:end-1])
    Vysc = 0.5*(V.y[2:end-1,1:end-1] + V.y[2:end-1,2:end])

    Vs   = sqrt.( Vxsc.^2 .+ Vysc.^2)

    Vxf  = -k_ηf0*diff(P.f, dims=1)/Δ.x
    Vyf  = -k_ηf0*diff(P.f, dims=2)/Δ.y
    Vyfc = 0.5*(Vyf[1:end-1,:] .+ Vyf[2:end,:])
    Vxfc = 0.5*(Vxf[:,1:end-1] .+ Vxf[:,2:end])


    Vf = sqrt.( Vxfc.^2 .+ Vyfc.^2)

    # P.t .= P.t .- mean(P.t)

    xce = LinRange(-L.x/2-Δ.x/2, L.x/2+Δ.x/2, nc.x+2)
    yce = LinRange(-L.y/2-Δ.y/2, L.y/2+Δ.y/2, nc.y+2)
    P.t[(xce.^2 .+ (yce').^2) .<= 1^2] .= 0.


    Vr  = zero(Vxsc)
    Vt  = zero(Vxsc)

    for i in 1:length(xce), j in 1:length(yce)

        r = sqrt.(xce[i].^2 .+ yce[j].^2)
        t = atan.(yce[j], xce[i])

        J = [cos(t) sin(t);    
             -sin(t) cos(t)]
        V_cart = [Vxsc[i,j]; Vysc[i,j]]
        V_pol  =  J*V_cart

        Vr[i,j] = V_pol[1]
        Vt[i,j] = V_pol[2]

        if (xce[i].^2 .+ yce[j].^2) <= r_in^2 ||  (xce[i].^2 .+ yce[j].^2) >= r_out^2
            Vr[i,j]     = NaN
            Vt[i,j]     = NaN
            P.f[i,j]    = NaN
            P.t[i,j]    = NaN
            # Ur_ana[i,j] = NaN
            # Ut_ana[i,j] = NaN
        else
            ϵ_Ur[i,j] = abs(Ur_ana[i,j] - Vr[i,j] )
            ϵ_Pf[i,j] = abs(Pf_ana[i,j] - P.f[i,j])
        end
        
    end

    @show mean(ϵ_Ur)
    @show mean(ϵ_Pf)

    # p1 = heatmap(xc, yc, Vs[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc), title="Vs")
    # p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, title="Ux", xlims=(-5,5), ylims=(-5,5))
    # p2 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, title="Uy", xlims=(-5,5), ylims=(-5,5))
    p1 = heatmap(xce, yce, Vr', aspect_ratio=1, title="Ur", c=:jet)
    p2 = heatmap(xce, yce, Vt', aspect_ratio=1, title="Ut", c=:jet)
    p3 = heatmap(xc, yc, P.t[inx_c,iny_c]',   aspect_ratio=1, title="Pt", c=:jet)
    p4 = heatmap(xc, yc, P.f[inx_c,iny_c]',   aspect_ratio=1, title="Pf", c=:jet)
    display(plot(p4, p3, p1, p2))

    ymid = Int64(floor(nc.y/2))
    p5 = plot(xlabel="x", ylabel="Pf")
    p5 = scatter!(xc, P.f[2:end-1, ymid], label="numerics")
    p5 = plot!(xc, BC.Pf[2:end-1, ymid], label="analytics")
    p6 = plot(xlabel="x", ylabel="Ur")
    p6 = scatter!(xc, Vr[2:end-1, ymid], label="numerics")
    p6 = plot!(xc, Ur_ana[2:end-1, ymid], label="analytics")
    display(plot(p5, p6))

    #--------------------------------------------#

    return P, Δ, (c=xc, v=xv), (c=yc, v=yv)
end

##################################
function Run()

    nc = (x=100, y=100)

    # Mode 0   
    Ωl = 0.1
    Ωη = 10.
    main(nc,  Ωl, Ωη);

end

Run()
