using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, CairoMakie, MathTeXEngine
Makie.update_theme!(fonts=(regular=texfont(), bold=texfont(:bold), italic=texfont(:italic)))
import Statistics: mean
using JustPIC, JustPIC._2D
import JustPIC.@index
const backend = JustPIC.CPUBackend
using DifferentiationInterface
using TimerOutputs, GridGeometryUtils