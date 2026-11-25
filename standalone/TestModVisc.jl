using ModVisc

function main()

model = ViscousModel(
    1.0,
    MeltFraction(Tube(0.0), 0.5),
    MeltFraction(Spheroid(1.0), 0.5),
)

@time  result = viscosity(model, 0.05)

# Porosity range
Φ = 10.0 .^ range(-6, log10(0.5); length=1000)

@time  profile = viscosity_profile(model, Φ)

# Equivalent collection dispatch:
@time profile = viscosity(model, Φ)

#############################

elastic_model = ElasticModel(
    66e9,
    40e9,
    20e9,
    PoreFraction(Spheroid(0.1), 0.0, 1.0);
    matrix_density=3300.0,
    density_contrast=300.0,
)
@time properties = elasticity(elastic_model, 0.01)

end

main()

# using CairoMakie
# using ModVisc

# CairoMakie.activate!()

# const OUTPUT_DIR = joinpath(@__DIR__, "output")
# const CUTOFF = 1e-3

# mkpath(OUTPUT_DIR)

# pure(geometry) = ViscousModel(1.0, MeltFraction(geometry, 1.0))

# function visible_curve!(axis, porosity, values; kwargs...)
#     keep = @. isfinite(values) & (values >= CUTOFF)
#     lines!(axis, porosity[keep], values[keep]; kwargs...)
# end

# function model_lines!(axis, porosity, model; label, kwargs...)
#     curve = viscosity_profile(model, porosity)
#     visible_curve!(axis, porosity, curve.bulk; label, kwargs...)
#     visible_curve!(axis, porosity, curve.shear; kwargs...)
# end

# function figure2()
#     porosity = collect(range(1e-6, 0.52; length=1200))
#     fig = Figure(size=(900, 650), fontsize=18)
#     ax = Axis(
#         fig[1, 1];
#         xlabel="Melt fraction",
#         ylabel="Effective viscosity / matrix shear viscosity",
#         limits=((0, 0.5), (0, 4)),
#     )

#     colors = Makie.wong_colors()
#     for (i, alpha) in enumerate((0.01, 0.03, 0.1, 0.3, 1.0))
#         model_lines!(ax, porosity, pure(Spheroid(alpha));
#                      color=colors[mod1(i, length(colors))], label="Spheroid alpha=$alpha")
#     end
#     model_lines!(ax, porosity, pure(Tube(0.0));
#                  color=:red3, linewidth=3, label="Tube, tapered")
#     model_lines!(ax, porosity, pure(Tube(Inf));
#                  color=:red3, linestyle=:dot, label="Tube, circular")

#     sphere_tube = ViscousModel(
#         1.0,
#         MeltFraction(Spheroid(1.0), 0.5),
#         MeltFraction(Tube(0.0), 0.5),
#     )
#     sphere_spheroid = ViscousModel(
#         1.0,
#         MeltFraction(Spheroid(1.0), 0.5),
#         MeltFraction(Spheroid(0.1), 0.5),
#     )
#     model_lines!(ax, porosity, sphere_tube;
#                  color=:dodgerblue3, linestyle=:dash,
#                  label="50% sphere + 50% tapered tube")
#     model_lines!(ax, porosity, sphere_spheroid;
#                  color=:dodgerblue3,
#                  label="50% sphere + 50% spheroid alpha=0.1")

#     text!(ax, 0.05, 2.6; text="Bulk viscosity")
#     text!(ax, 0.05, 0.78; text="Shear viscosity")
#     axislegend(ax; position=:rt, labelsize=12)
#     save(joinpath(OUTPUT_DIR, "figure2.png"), fig; px_per_unit=2)
#     # save(joinpath(OUTPUT_DIR, "figure2.pdf"), fig)
#     return fig
# end

# function figure3()
#     porosity = 10.0 .^ range(-3, log10(0.6); length=1200)
#     fig = Figure(size=(900, 650), fontsize=18)
#     ax = Axis(
#         fig[1, 1];
#         xscale=log10,
#         yscale=log10,
#         xlabel="Melt fraction",
#         ylabel="Effective viscosity / matrix shear viscosity",
#         limits=((1e-3, 0.6), (1e-3, 1e3)),
#     )

#     for (i, alpha) in enumerate((0.01, 0.03, 0.1, 0.3, 1.0))
#         color = Makie.wong_colors()[mod1(i, length(Makie.wong_colors()))]
#         curve = viscosity_profile(pure(Spheroid(alpha)), porosity)
#         fits = spheroid_fit.(porosity, alpha)
#         fit_bulk = getproperty.(fits, :bulk)
#         fit_shear = getproperty.(fits, :shear)

#         visible_curve!(ax, porosity, curve.bulk;
#                        color, label="alpha=$alpha numerical")
#         visible_curve!(ax, porosity, curve.shear; color)
#         visible_curve!(ax, porosity, fit_bulk;
#                        color, linestyle=:dot, label="alpha=$alpha fit")
#         visible_curve!(ax, porosity, fit_shear; color, linestyle=:dot)
#     end

#     text!(ax, 2e-3, 35; text="Bulk viscosity")
#     text!(ax, 2e-3, 1.25; text="Shear viscosity")
#     axislegend(ax; position=:rt, labelsize=11)
#     save(joinpath(OUTPUT_DIR, "figure3.png"), fig; px_per_unit=2)
#     # save(joinpath(OUTPUT_DIR, "figure3.pdf"), fig)
#     return fig
# end

# figure2()
# figure3()