module StagFDTools

using StaticArrays, ExtendableSparse, StaticArrays, Printf, LinearAlgebra
using DifferentiationInterface, ForwardDiff

include("AD.jl")
export ad_gradient, ad_value_and_gradient, ad_derivative, ad_value_and_derivative
export ad_jacobian, ad_value_and_jacobian, ad_partial_gradients, ad_value_and_jacobian_first
export Const, Duplicated, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian

include("operators.jl")
export inn, inn_x, inn_y, av, avx, avy, harm, ∂x, ∂y, ∂x_inn, ∂y_inn, ∂kk
export deviatoric_strain_rate, effective_strain_rate

include("Utils.jl")
export GenerateGrid, printxy, av2D, Plot_Tangent_Operator

include("Solvers.jl")
export DecoupledSolver, KSP_GCR_Stokes!, mechanical_solver!, linear_tol, two_phases_mechanical_solver!, KSP_GCR_TwoPhases_setup, KSP_GCR_TwoPhases_opt! 

include("BCs.jl")
export SetBCPf1, SetBCPt1, SetBCVx1, SetBCVy1

include("materials.jl")
export Materials, Materials_TwoPhases, preprocess!, preprocess
export AbstractPlasticity, VonMises, DruckerPrager, DruckerPrager1, DruckerHyperbolic, DruckerAniso, Golchin2021, Kiss2023, Tensile, NoPlasticity
export initialize_materials, initialize_materials_TwoPhases

# module markers
#     include("markers.jl")
#     export PhaseRatios, ...
# end
module Rheology
using StaticArrays, StagFDTools, LinearAlgebra
include("Rheology.jl")
export LocalRheology, StressVector!
export LocalRheology_div, StressVector_div!
export Yield, Potential
end

module Poisson
using StaticArrays, ExtendableSparse, StaticArrays
include("Poisson.jl")
export Fields, Ranges, Numbering!, SparsityPattern!
end
module Stokes
using LinearAlgebra, StaticArrays, ExtendableSparse, StaticArrays, StagFDTools, StagFDTools.Rheology, DifferentiationInterface
include("Stokes.jl")
export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx!, SetBCVy!, set_boundaries_template!, SetBCVx1, SetBCVy1
export Continuity, SMomentum_x_Generic, SMomentum_y_Generic
export ResidualContinuity2D!, ResidualMomentum2D_x!, ResidualMomentum2D_y!
export AssembleContinuity2D!, AssembleMomentum2D_x!, AssembleMomentum2D_y!
export TangentOperator!, LineSearch!
include("Markers.jl")
export InitialiseMarkerField, InitialisePhaseRatios, SetPhaseRatios!, compute_grid_fields!
end
module StokesDeformed
using LinearAlgebra, StaticArrays, ExtendableSparse, StaticArrays, StagFDTools, StagFDTools.Rheology
include("StokesDeformed.jl")
export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx!, SetBCVy!, set_boundaries_template!, SetBCVx1, SetBCVy1
export Continuity, SMomentum_x_Generic, SMomentum_y_Generic
export ResidualContinuity2D!, ResidualMomentum2D_x!, ResidualMomentum2D_y!
export AssembleContinuity2D!, AssembleMomentum2D_x!, AssembleMomentum2D_y!
export TangentOperator!
export LineSearch!
end

module StokesFSG
using StaticArrays, ExtendableSparse, StaticArrays, StagFDTools
include("StokesFSG.jl")
export FSG_Array, Fields, Ranges, Numbering!#, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx!, SetBCVy!
export AllocateSparseMatrix, Patterns
export AssembleContinuity2D_1!, AssembleContinuity2D_2!, ResidualContinuity2D_1!, ResidualContinuity2D_2!
export SetRHS!, UpdateSolution!, SetRHSSG1!, UpdateSolutionSG1!, SetRHSSG2!, UpdateSolutionSG2!
end

module ThermoMechanics
using StagFDTools, StaticArrays, ExtendableSparse, StaticArrays, LinearAlgebra, MineralEoS
include("ThermoMechanics/ThermoMechanics.jl")
export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx1, SetBCVy1
export AssembleHeatDiffusion2D!, ResidualHeatDiffusion2D!, HeatDiffusion
export AssembleContinuity2D!, ResidualContinuity2D!, Continuity
export AssembleMomentum2D_y!, ResidualMomentum2D_y!, Momentum_y
export AssembleMomentum2D_x!, ResidualMomentum2D_x!, Momentum_x
export LineSearch!
include("ThermoMechanics/ThermoMechanics_Rheology.jl")
export LocalRheology, StressVector!, TangentOperator!
end

module TwoPhases
using StagFDTools, StaticArrays, ExtendableSparse, StaticArrays, LinearAlgebra
# Material produced before 09/25 wew done with this
# include("TwoPhases/TwoPhases_v2.jl")
# Now this one is preferred because it fully accounts for porosity evolution
include("TwoPhases/TwoPhases_v3.jl")
export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx1, SetBCVy1, SetBCPf1
export AssembleFluidContinuity2D!, ResidualFluidContinuity2D!, FluidContinuity
export LineSearch!, BackTrackingLineSearch!
export AssembleContinuity2D!, ResidualContinuity2D!, Continuity, ResidualPorosity2D!, UpdatePorosity2D!
export AssembleMomentum2D_y!, ResidualMomentum2D_y!, Momentum_y
export AssembleMomentum2D_x!, ResidualMomentum2D_x!, Momentum_x
export reduce_sparse_matrix!, reset_parallel_storage
# export AssembleFluidContinuity2D_VE!, ResidualFluidContinuity2D_VE!, FluidContinuity_VE
# export AssembleContinuity2D_VE!, ResidualContinuity2D_VE!, Continuity_VE
# include("TwoPhases.jl")
# export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx1, SetBCVy1
# export AssembleFluidContinuity2D!, ResidualFluidContinuity2D!, FluidContinuity
# export AssembleContinuity2D!, ResidualContinuity2D!, Continuity
# export AssembleMomentum2D_y!, ResidualMomentum2D_y!, Momentum_y
# export AssembleMomentum2D_x!, ResidualMomentum2D_x!, Momentum_x
# include("TwoPhases_VE.jl")
# export AssembleFluidContinuity2D_VE!, ResidualFluidContinuity2D_VE!, FluidContinuity_VE
# export AssembleContinuity2D_VE!, ResidualContinuity2D_VE!, Continuity_VE
include("TwoPhases/TwoPhases_Rheology_Trial_P.jl")
export TangentOperator!, Porosity
include("TwoPhases/TwoPhases_Rheology_Common.jl")
export invII, StrainRateTrial, F
include("Markers.jl")
export InitialiseMarkerField, InitialisePhaseRatios, SetPhaseRatios!, compute_grid_fields_two_phases!
end

module TwoPhases_v1
using StagFDTools, StaticArrays, ExtendableSparse, StaticArrays
include("TwoPhases/TwoPhases_v1.jl")
export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx1, SetBCVy1
export AssembleFluidContinuity2D!, ResidualFluidContinuity2D!, FluidContinuity
export AssembleContinuity2D!, ResidualContinuity2D!, Continuity
export AssembleMomentum2D_y!, ResidualMomentum2D_y!, Momentum_y
export AssembleMomentum2D_x!, ResidualMomentum2D_x!, Momentum_x
include("TwoPhases/TwoPhases_VE.jl")
export AssembleFluidContinuity2D_VE!, ResidualFluidContinuity2D_VE!, FluidContinuity_VE
export AssembleContinuity2D_VE!, ResidualContinuity2D_VE!, Continuity_VE
end

end # module StagFDTools
