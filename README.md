# StagFDTools.jl

[![CI](https://img.shields.io/github/actions/workflow/status/tduretz/StagFDTools.jl/ci.yml?label=CPU%20Unit%20Tests)](https://github.com/tduretz/StagFDTools.jl/actions/workflows/ci.yml/badge.svg)

This package aims at generating flexible FD stencils that can interoperate with AD tools for Jacobian generation. In particular, the aims are:
- generic treatment of BCs
- generation of equation numbering
- generation of sparsity patterns
- flexible treatment of multi-physics

# Get started

0) Make sure you have Julia and `VSCode` or `VSCodium` working together. *Recommended way: (1) Install Julia using [`juliaup`](https://github.com/JuliaLang/juliaup), (2) install [`VSCodium`](https://vscodium.com), (3) install `Julia Language Support` in `VSCode` extensions.*

1) In your general Julia environment, make sure you have plotting packages installed.
*How: (1) In Julia's terminal (REPL), type: `]` to go to package mode (`(@v1.11) pkg>`). (2) type `add Plots` and then `add Makie`.*

2) Clone this repository somewhere on your computer.

3) In `VSCodium`, click on `File`, then `New Window`. Click on `Open...` and select the `StagFDTools` folder. *Achtung: At this point, you should agree to select `StagFDTools`'s environment*.

4) Go to package mode: type `]`. It should clearly indicate that you're working in the correct environment (`(StagFDTools) pkg>`). If not, repeat step 3. If yes, type `instantiate`.

5) Now, you should be able to run the scripts in the `example/` folder.

****

## Single phase mechanics

### Power-law Stokes (standard staggered grid - asymmetric operator)

![image](./results/PowerLaw.gif)
<!-- 
![image](https://github.com/user-attachments/assets/e29d72a5-93cf-4cc5-84a1-c353a05a4edb) -->

### Shear banding benchmark ([Duretz et al., 2018 - test 1](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018GC007877))

see code [here](examples/_Stokes_VEP_SSG/ShearBanding_Duretz2018/Example_EP_Duretz2018.jl)

![image](./results/EP_Duretz2018.png)

### Shear banding with compressibility and dilatancy
![image](./results/ShearBanding.gif)

### Pressurized hole with mixed-mode plasticity (see Kiss et al., 2023)
![image](./results/PressurizedHole.gif)

### Host-inclusion decompression with Drucker-Prager plasticity
![image](./results/HostInclusion_DruckerPrager.gif)

### Host-inclusion decompression with tensile plasticity
![image](./results/HostInclusion_tensile.gif)

### Power-law Stokes (full staggered grid - symmetric operator)
![image](https://github.com/user-attachments/assets/9c1e02d5-6b7f-4764-a99d-12a87e28ea21)

### Anisotropic Stokes (full staggered grid - symmetric operator)
![image](https://github.com/user-attachments/assets/3df8215a-0eca-4e3e-b01a-85a501a4bacb)

****

## Two-phase mechanics
![image](https://github.com/user-attachments/assets/e5606f59-1a56-43e8-84d9-25381318eb0c)

****

### Poisson / Diffusion and sparsity examples

#### Diffusion (backward Euler)
![image](results/Diffusion_BackwardEuler.svg)

#### Diffusion (Crank-Nicolson)
![image](results/Diffusion_CrankNicolson.svg)

####  Flag boundary nodes and constant nodes (e.g. inner BC or free surface)
```julia
[ Info: Node types
6×5 Matrix{Symbol}:
 :Neumannn    :Neumannn    :Neumannn    :Neumannn    :Neumannn
 :periodic   :in         :in         :in         :periodic
 :periodic   :in         :in         :in         :periodic
 :periodic   :in         :in         :in         :periodic
 :periodic   :in         :in         :in         :periodic
 :Dirichlet  :Dirichlet  :Dirichlet  :Dirichlet  :Dirichlet
```

#### Generation of a 5-point stencil & check for symmetry
```julia 
[ Info: 5-point stencil
12×12 ExtendableSparseMatrixCSC{Float64, Int64} with 54 stored entries:
 1.0  1.0  1.0  1.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
 1.0  1.0  1.0   ⋅   1.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
 1.0  1.0  1.0   ⋅    ⋅   1.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
 1.0   ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅   1.0   ⋅   1.0  1.0  1.0   ⋅   1.0   ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅   1.0   ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅   1.0   ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅   1.0   ⋅   1.0  1.0  1.0   ⋅   1.0   ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅   1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   1.0   ⋅    ⋅   1.0  1.0  1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   1.0   ⋅   1.0  1.0  1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   1.0  1.0  1.0  1.0
12×12 SparseArrays.SparseMatrixCSC{Float64, Int64} with 0 stored entries:
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
```

#### Generation of a 9-point stencil & check for symmetry
```julia
[ Info: 9-point stencil
12×12 ExtendableSparseMatrixCSC{Float64, Int64} with 54 stored entries:
 1.0  1.0  1.0  1.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
 1.0  1.0  1.0   ⋅   1.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
 1.0  1.0  1.0   ⋅    ⋅   1.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
 1.0   ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅   1.0   ⋅   1.0  1.0  1.0   ⋅   1.0   ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅   1.0   ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅   1.0   ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅   1.0   ⋅   1.0  1.0  1.0   ⋅   1.0   ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅   1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   1.0   ⋅    ⋅   1.0  1.0  1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   1.0   ⋅   1.0  1.0  1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   1.0  1.0  1.0  1.0
12×12 SparseArrays.SparseMatrixCSC{Float64, Int64} with 0 stored entries:
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
```
