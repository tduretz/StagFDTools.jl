using Base.Threads, ExtendableSparse, LinearAlgebra

function merge_sparse(A, B)
    IA, JA, VA = findnz(A)
    IB, JB, VB = findnz(B)

    return sparse(
        vcat(IA, IB),
        vcat(JA, JB),
        vcat(VA, VB),
        size(A,1),
        size(A,2)
    )
end


function merge_COO(I, J, V, A)
    IA, JA, VA = findnz(A)
    append!(I, IA)
    append!(J, JA)
    append!(V, VA)
end

@views function serial_assembly!(M, n)
    for i=1:n
        if i==1
            M[i,i  ] =  1.0  
            M[i,i+1] = -1.0
        elseif i==n
            M[i,i-1] = -1.0  
            M[i,i  ] =  1.0
        else
            M[i,i-1] = -1.0  
            M[i,i  ] =  2.0
            M[i,i+1] = -1.0
        end
    end
end

@views function threaded_assembly!(M, M_loc, n)
    Threads.@threads for i=1:n
        tid = threadid()
        if i==1
            M_loc[tid-1][i,i  ] =  1.0  
            M_loc[tid-1][i,i+1] = -1.0
        elseif i==n
            M_loc[tid-1][i,i-1] = -1.0  
            M_loc[tid-1][i,i  ] =  1.0
        else
            M_loc[tid-1][i,i-1] = -1.0  
            M_loc[tid-1][i,i  ] =  2.0
            M_loc[tid-1][i,i+1] = -1.0
        end
    end
end

let 
    n  = 10_000_000

    # Serial
    M1 = ExtendableSparseMatrix(n,n)
    @time serial_assembly!(M1, n)

    # Threaded - version 1
    M2 = ExtendableSparseMatrix(n,n)
    M2_loc = [ExtendableSparseMatrix(n,n) for i=1:nthreads()]
    @time threaded_assembly!(M2, M2_loc, n)
    # Reduce. Looks like a conversion to sparse is needed to allow for .+
    # This makes things a slightly faster
    @time for k=1:nthreads()
        M2 .= sparse(M2) .+ sparse(M2_loc[k])
    end

    M3 = ExtendableSparseMatrix(n,n)
    M3_loc = [ExtendableSparseMatrix(n,n) for i=1:nthreads()]
    @time threaded_assembly!(M3, M3_loc, n)
    # Reduce. 
    @time begin 
        I = Int[]
        J = Int[]
        V = Float64[]
        for k=1:nthreads()
            I_loc, J_loc, V_loc = findnz(M2_loc[k])
            append!(I, I_loc)
            append!(J, J_loc)
            append!(V, V_loc)
        end
        M3 = sparse(I, J, V)
    end

    # Check
    @show norm(M1 .- M2)
    @show norm(M1 .- M3)
end