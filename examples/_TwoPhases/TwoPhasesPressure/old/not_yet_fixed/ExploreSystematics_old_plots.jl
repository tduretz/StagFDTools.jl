using JLD2, Plots

function main()

    data = load("./examples/_TwoPhases/TwoPhasesPressure/Systematics_Large.jld2") 
    data = load("./examples/_TwoPhases/TwoPhasesPressure/Systematics_Zoom.jld2") 


    Ωη  = data["Ωη"]
    Ωl  = data["Ωl"]
    
    ΔPt    = data["ΔPt"]
    ΔPf    = data["ΔPf"]
    ΔPe    = data["ΔPe"]
    Pe     = data["Pe"]
    Pt     = data["Pt"]
    Pf     = data["Pf"]
    ∇Pe    = data["∇Pe"] 
    ∇Pt    = data["∇Pt"]
    ∇Pf    = data["∇Pf"]

    display(ΔPf)
    @show sum(isnan.(ΔPf))

    p1 = heatmap(log10.(Ωl), log10.(Ωη), ΔPt', xlabel="Ωl", ylabel="Ωη", title="ΔPt", clim=(0, 6), aspect_ratio=1, xlim=extrema(log10.(Ωl)))
    # p1 = scatter!([-2.5 -1.5 -1 0], [-2.5 -1.5 -1 0], label=:none, c=:white)
    
    p2 = heatmap(log10.(Ωl), log10.(Ωη), ΔPf', xlabel="Ωl", ylabel="Ωη", title="ΔPf", clim=(0, 6), aspect_ratio=1, xlim=extrema(log10.(Ωl)))
    # p2 = scatter!([-2.5 -1.5 -1 0], [-2.5 -1.5 -1 0], label=:none, c=:white)

    p3 = contour(log10.(Ωl), log10.(Ωη), ΔPe', xlabel="Ωl", ylabel="Ωη", title="ΔPe", clim=(0, 6), aspect_ratio=1, xlim=extrema(log10.(Ωl)))
    # p3 = scatter!([-2.5 -1.5 -1 0], [-2.5 -1.5 -1 0], label=:none, c=:white)
 
    p4 = contour(log10.(Ωl), log10.(Ωη), (ΔPt.-ΔPf)',  xlabel="Ωl", ylabel="Ωη", title="ΔPt.-ΔPf", clim=(0, 6), aspect_ratio=1, xlim=extrema(log10.(Ωl)))
    # p4 = scatter!([-2.5 -1.5 -1 0], [-2.5 -1.5 -1 0], label=:none, c=:white)


    # p1 = heatmap(log10.(Ωl), log10.(Ωη), Pt', xlabel="Ωl", ylabel="Ωη", title="Pt", clim=(0, 1), aspect_ratio=1, xlim=extrema(log10.(Ωl)))
    # p1 = scatter!([-2.5 -1.5 -1 0], [-2.5 -1.5 -1 0], label=:none, c=:white)
    
    # p2 = heatmap(log10.(Ωl), log10.(Ωη), Pf', xlabel="Ωl", ylabel="Ωη", title="Pf", clim=(0, 1), aspect_ratio=1, xlim=extrema(log10.(Ωl)))
    # p2 = scatter!([-2.5 -1.5 -1 0], [-2.5 -1.5 -1 0], label=:none, c=:white)

    # p3 = heatmap(log10.(Ωl), log10.(Ωη), Pe', xlabel="Ωl", ylabel="Ωη", title="Pe", clim=(0, 1), aspect_ratio=1, xlim=extrema(log10.(Ωl)))
    # p3 = scatter!([-2.5 -1.5 -1 0], [-2.5 -1.5 -1 0], label=:none, c=:white)
 
    # p4 = heatmap(log10.(Ωl), log10.(Ωη), NaN.*Pe',  xlabel="Ωl", ylabel="Ωη", title="Pe", clim=(0, 50), aspect_ratio=1, xlim=extrema(log10.(Ωl)))
    # p4 = scatter!([-2.5 -1.5 -1 0], [-2.5 -1.5 -1 0], label=:none, c=:white)


    # p1 = heatmap(log10.(Ωl), log10.(Ωη), ∇Pt', xlabel="Ωl", ylabel="Ωη", title="∇Pt", clim=(0, 1), aspect_ratio=1, xlim=extrema(log10.(Ωl)))
    # p1 = scatter!([-2.5 -1.5 -1 0], [-2.5 -1.5 -1 0], label=:none, c=:white)
    
    # p2 = heatmap(log10.(Ωl), log10.(Ωη), ∇Pf', xlabel="Ωl", ylabel="Ωη", title="∇Pf", clim=(0, 1), aspect_ratio=1, xlim=extrema(log10.(Ωl)))
    # p2 = scatter!([-2.5 -1.5 -1 0], [-2.5 -1.5 -1 0], label=:none, c=:white)

    # p3 = heatmap(log10.(Ωl), log10.(Ωη), ∇Pe', xlabel="Ωl", ylabel="Ωη", title="∇Pe", clim=(0, 1), aspect_ratio=1, xlim=extrema(log10.(Ωl)))
    # p3 = scatter!([-2.5 -1.5 -1 0], [-2.5 -1.5 -1 0], label=:none, c=:white)
 
    # p4 = heatmap(log10.(Ωl), log10.(Ωη), NaN.*Pe',  xlabel="Ωl", ylabel="Ωη", title="Pe", clim=(0, 50), aspect_ratio=1, xlim=extrema(log10.(Ωl)))
    # p4 = scatter!([-2.5 -1.5 -1 0], [-2.5 -1.5 -1 0], label=:none, c=:white)



    # p5 = heatmap(log10.(Ωl), log10.(Ωη), Pe_Pfc', clim=(0, 50), aspect_ratio=1)
    plot(p1, p2, p3, p4)

    # scatter(log10.(Ωl), ΔPt[:,end-1])
    # @show Ωη[end-1]
    # heatmap(log10.(Ωl), log10.(Ωη), (ΔPt.-ΔPf)',  xlabel="Ωl", ylabel="Ωη", title="Pe", clim=(-1, 6), aspect_ratio=1, xlim=extrema(log10.(Ωl)))
    # scatter!([-2.5 -1.5 -1 0], [-2.5 -1.5 -1 0], label=:none, c=:white)

end

main()