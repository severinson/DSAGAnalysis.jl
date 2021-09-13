using HDF5, H5Sparse, SparseArrays, DSAG
using DataFrames, CSV, Glob, Dates, Random, LinearAlgebra

"""

Create a from the output file of a particular job. `nrows=2504` and `ncolumns=81271767` is correct 
for the 1000 Genomes dataset.
"""
function df_from_fid(fid, nrows=2504, ncolumns=81271767)
    rv = DataFrame()
    row = Dict{String, Any}()
    row["mse"] = missing # initialize to missing, it's computed later

    # store job parameters
    if "parameters" in keys(fid) && typeof(fid["parameters"]) <: HDF5.Group
        g = fid["parameters"]
        for key in keys(g)
            value = g[key][]
            row[key] = value
        end
    end

    # default values for nrows and ncolumns
    # (previous versions of the distributed solver would not store these in the output file)
    if !haskey(row, "nrows")
        row["nrows"] = nrows
    end
    if !haskey(row, "ncolumns")
        row["ncolumns"] = ncolumns
    end    

    # add benchmark data
    niterations = Int(row["niterations"])
    nworkers = Int(row["nworkers"])
    for i in 1:niterations
        row["iteration"] = i
        row["t_compute"] = fid["benchmark/t_compute"][i]
        row["t_update"] = fid["benchmark/t_update"][i]
        for j in 1:nworkers # worker response epochs
            row["repoch_worker_$j"] = fid["benchmark/responded"][j, i]
        end
        if "latency" in keys(fid["benchmark"]) # worker latency
            for j in 1:nworkers
                row["latency_worker_$j"] = fid["benchmark/latency"][j, i]
            end
        end            
        if "compute_latency" in keys(fid["benchmark"]) # worker compute latency
            for j in 1:nworkers
                row["compute_latency_worker_$j"] = fid["benchmark/compute_latency"][j, i]
            end
        end
        if "loadbalanced" in keys(fid["benchmark"])                                 
            row["loadbalanced"] = fid["benchmark/loadbalanced"][i]
        end
        push!(rv, row, cols=:union)
    end
    rv
end

"""

Compute the explained variance (here referred to as mse) for PCA. `Xnorm=104444.37027911078` is 
correct for the 1000 Genomes dataset.
"""
function compute_mse_pca!(mses, iterates, Xs; mseiterations=20, Xnorm=104444.37027911078)
    if iszero(mseiterations)
        return mses
    end
    niterations = size(iterates, 3)
    is = unique(round.(Int, exp.(range(log(1), log(niterations), length=mseiterations))))
    for k in 1:length(is)
        i = is[k]
        if !ismissing(mses[i])
            continue
        end
        norms = zeros(length(Xs))
        t = @elapsed begin
            Threads.@threads for j in 1:length(Xs)
                norms[j] = norm(view(iterates, :, :, i)'*Xs[j])
            end
        end
        mses[i] = (sum(norms) / Xnorm)^2
        println("Iteration $i finished in $t s")
    end
    GC.gc()
    mses
end

function compute_mse_pca!(mses, iterates, iterateindices, Xs; Xnorm=104444.37027911078)
    size(iterates, 3) == length(iterateindices) || throw(DimensionMismatch("iterates has dimensions $(size(iterates)), but iterateindices has dimension $(length(iterateindices))"))
    for k in 1:iterateindices
        i = iterateindices[k]
        if !ismissing(mses[i])
            continue
        end
        norms = zeros(length(Xs))
        t = @elapsed begin
            Threads.@threads for j in 1:length(Xs)
                norms[j] = norm(view(iterates, :, :, k)'*Xs[j])
            end
        end
        mses[i] = (sum(norms) / Xnorm)^2
        println("Iteration $i finished in $t s")
    end
    GC.gc()
    mses
end

function compute_mse_logreg!(mses, iterates, data; mseiterations)
    if iszero(mseiterations)
        return mses
    end    
    Xs, bs, λ = data
    prob = LogRegProblem(Xs, bs, λ)
    niterations = length(mses)
    is = unique(round.(Int, exp.(range(log(1), log(niterations), length=mseiterations))))
    for k in 1:length(is)
        i = is[k]
        if !ismissing(mses[i])
            continue
        end
        v = view(iterates, :, i)
        t = @elapsed begin
            mses[i] = loss(v, prob) + loss(v, regularizer(prob))
        end
        @info "Computed MSE $(mses[i]) for iteration $i in $t seconds"
    end
    mses
end

function compute_mse_logreg!(mses, iterates, iterateindices, data)
    size(iterates, 2) == length(iterateindices) || throw(DimensionMismatch("iterates has dimensions $(size(iterates)), but iterateindices has dimension $(length(iterateindices))"))
    @info "Computing log. reg. MSE for iterations $iterateindices"
    Xs, bs, λ = data
    prob = LogRegProblem(Xs, bs, λ)
    for k in 1:length(iterateindices)
        i = iterateindices[k]
        if !ismissing(mses[i])
            continue
        end        
        v = view(iterates, :, k)
        t = @elapsed begin
            mses[i] = loss(v, prob) + loss(v, regularizer(prob))
        end
        @info "Computed MSE $(mses[i]) for iteration $i in $t seconds"        
    end
    mses
end

function compute_mse!(df, fid, prob; mseiterations, reparse)
    mses = Vector{Union{Float64,Missing}}(df.mse)
    if reparse
        mses .= missing
    end
    algo = df.algorithm[1]    
    if algo == "pca.jl"
        select!(df, Not(:mse))
        if "savediterateindices" in keys(fid)
            iterateindices = fid["savediterateindices"][:]
            df.mse = compute_mse_pca!(mses, fid["iterates"][:, :, :], iterateindices, prob)            
        else
            df.mse = compute_mse_pca!(mses, fid["iterates"][:, :, :], prob; mseiterations)
        end        
    elseif algo == "logreg.jl"
        Xs, bs = prob
        if length(unique(df.lambda)) != 1
            error("λ must be unique within each DataFrame")
        end
        λ = df.lambda[1]
        select!(df, Not(:mse))
        if "savediterateindices" in keys(fid)
            iterateindices = fid["savediterateindices"][:]
            df.mse = compute_mse_logreg!(mses, fid["iterates"][:, :], iterateindices, (Xs, bs, λ))
        else
            df.mse = compute_mse_logreg!(mses, fid["iterates"][:, :], (Xs, bs, λ); mseiterations)
        end
    else
        raise(ArgumentError("unknown algorithm $algo"))
    end
    df
end

"""
    partition(n::Integer, p::Integer, i::Integer)

Divide the integers from `1` to `n` into `p` evenly sized partitions, and return a `UnitRange` 
making up the integers of the `i`-th partition.
"""
function partition(n::Integer, p::Integer, i::Integer)
    0 < n || throw(ArgumentError("n must be positive, but is $n"))
    0 < p <= n || throw(ArgumentError("p must be in [1, $n], but is $p"))    
    0 < i <= p || throw(ArgumentError("i must be in [1, $p], but is $i"))
    (div((i-1)*n, p)+1):div(i*n, p)
end

"""

Read the sparse matrix stored in dataset with `name` in `filename` and partitions it column-wise 
into `nblocks` partitions.
"""
function load_inputmatrix(filename, name::AbstractString="X"; nblocks=Threads.nthreads())
    X = H5SparseMatrixCSC(filename, name)
    m, n = size(X)
    [sparse(X[:, partition(n, nblocks, i)]) for i in 1:nblocks]
end

"""

Return a vector composed of the number of flops performed by each worker and iteration. The density
of the 1000 Genomes data matrix is `0.05360388070027386`.
"""
function worker_flops_from_df_pca(df; density=0.05360388070027386)
    nflops = float.(df.nrows)
    nflops ./= df.nworkers
    nflops .*= df.nreplicas
    nflops ./= Missings.replace(df.nsubpartitions, 1.0)
    nflops .*= 2 .* df.ncolumns .* df.ncomponents
    nflops .*= density
end

"""

The density of the rcv1full dataset is `0.0015492979884363385`.
"""
function worker_flops_from_df_logreg(df; density=0.0015492979884363385)
    2density .* df.nrows .* df.ncolumns ./ df.npartitions .* df.nreplicas
end

"""
    df_from_output_file(filename::AbstractString, Xs::Vector{<:AbstractMatrix}; df_filename::AbstractString=replace(filename, ".h5"=>".csv"), mseiterations=0, reparse=false)

Parse the .h5 file `filename` (resulting from a run of the PCA kernel) into a DataFrame, which is 
written to disk as a .csv file with name `df_filename`. If `reparse = false` and `df_filename` 
already exists, then the existing `.csv` file is read from disk and returned, otherwise the `.h5`
file is parsed again.

To compute the explained variance of each iteration (here referred to as mse), the data matrix `X`
must be provided as a vector `Xs`, corresponding to a horizontal partitioning of the data matrix, 
i.e., `X = hcat(Xs, ...)`. Explained variance is computed for of up to `mseiterations` different 
iterations.
"""
function df_from_output_file(filename::AbstractString; prob=nothing, df_filename::AbstractString=replace(filename, ".h5"=>".csv"), mseiterations=0, reparse=false)
    if !reparse && isfile(df_filename)
        return DataFrame(CSV.File(df_filename))
    end
    if !HDF5.ishdf5(filename)
        println("skipping (not a HDF5 file): $filename")
        return DataFrame()
    end
    h5open(filename) do fid
        df = isfile(df_filename) ? DataFrame(CSV.File(df_filename)) : df_from_fid(fid)
        df = df[.!ismissing.(df.iteration), :]
        sort!(df, :iteration)        
        if size(df, 1) == 0
            return df
        end
        if "iterates" in keys(fid) && mseiterations > 0 && !isnothing(prob)
            compute_mse!(df, fid, prob; mseiterations, reparse)
        end
        CSV.write(df_filename, df)
        return df
    end
end

"""

Aggregate all DataFrames in `dir` into a single DataFrame.
"""
function aggregate_dataframes(dir::AbstractString; outputdir::AbstractString=dir, prefix::AbstractString="output", dfname::AbstractString="df.csv")
    filenames = glob("$(prefix)*.csv", dir)
    println("Aggregating $(length(filenames)) files")
    dfs = [DataFrame(CSV.File(filename)) for filename in filenames]
    if length(dfs) == 0
        return DataFrame()
    end
    for (i, df) in enumerate(dfs)
        df[!, :jobid] .= i # store a unique ID for each file read
    end
    for (filename, df) in zip(filenames, dfs)
        df[!, :filename] .= filename # store the filename
    end
    df = vcat(dfs..., cols=:union)
    df = clean_df(df)
    CSV.write(joinpath(outputdir, dfname), df)
    df
end

function load_logreg_problem(filename="/home/albin/rcv1/rcv1_shuffled.h5"; nblocks=Threads.nthreads(), name="X", labelname="b")
    h5open(filename) do fid
        X = H5SparseMatrixCSC(fid, name)
        m, n = size(X)
        Xs = [sparse(X[:, partition(n, nblocks, i)]) for i in 1:nblocks]
        bs = [fid[labelname][partition(n, nblocks, i)] for i in 1:nblocks]
        return Xs, bs
    end
end

"""

Cleanup
"""
function clean_df(df::DataFrame)
    df = df[.!ismissing.(df.nworkers), :]
    df = df[.!ismissing.(df.iteration), :]
    df[!, :nostale] .= Missings.replace(df.nostale, false)
    df[!, :kickstart] .= Missings.replace(df.kickstart, false)
    df = df[df.kickstart .== false, :]
    select!(df, Not(:kickstart)) # drop the kickstart column
    df.npartitions = df.nworkers .* df.nsubpartitions
    rename!(df, :t_compute => :latency)
    rename!(df, :t_update => :update_latency)
        
    algo = df.algorithm[1]
    length(unique(df.algorithm)) == 1 || error("Mixing multiple algorithms within the same DataFrame is not allowed")
    if algo == "pca.jl"
        df[!, :worker_flops] = worker_flops_from_df_pca(df)    
        df[!, :nbytes] = df.nrows .* df.ncomponents .* 4 # Float32 entries => 4 bytes per entry        
    elseif algo == "logreg.jl"
        df[!, :worker_flops] = worker_flops_from_df_logreg(df)
        df[!, :nbytes] = (df.nrows .+ 1) .* 4 # Float32 entries => 4 bytes per entry
    else
        error("algorithm is $algo, but must be either of pca.jl or logreg.jl")
    end
    sort!(df, [:jobid, :iteration])
    df.time = combine(groupby(df, :jobid), :latency => cumsum => :time).time # cumulative time since the start of the computation
    df.time .+= combine(groupby(df, :jobid), :update_latency => cumsum => :time).time
    df
end

function parse_output_file(filename::AbstractString; reparse=false, prob=nothing, mseiterations=20)
    try
        return df_from_output_file(filename; prob, mseiterations, reparse)
    catch e
        printstyled(stderr,"ERROR: ", bold=true, color=:red)
        printstyled(stderr,sprint(showerror,e), color=:light_red)
        println(stderr)
    end    
end

"""

Read all output files from `dir` and write summary statistics (e.g., iteration time and convergence) to DataFrames.
"""
function parse_output_files(dir::AbstractString; prefix="output", dfname="df.csv", reparse=false, prob=nothing, mseiterations=20)

    # process output files
    filenames = glob("$(prefix)*.h5", dir)
    shuffle!(filenames) # randomize the order to minimize overlap when using multiple concurrent processes
    for (i, filename) in enumerate(filenames)
        t = now()
        println("[$i / $(length(filenames)), $(Dates.format(now(), "HH:MM"))] parsing $filename")
        parse_output_file(filename; reparse, prob, mseiterations)
        GC.gc()
    end
    aggregate_dataframes(dir; prefix, dfname)
end

function parse_loop(args...; kwargs...)
    while true
        parse_output_files(args...; kwargs...)
        println("Sleeping for 60s")
        sleep(60)
    end
    return
end