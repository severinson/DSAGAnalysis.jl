module DSAGAnalysis

using LinearAlgebra, SparseArrays
using CSV, DataFrames, PyPlot
using Random, StatsBase, Statistics, Distributions
using DataStructures
using HDF5, H5Sparse

include("Stats.jl")
include("LIBSVM.jl")
include("eventdriven.jl")

### utilities

"""

Write a table composed of two columns to file, where the elements of `xs` and `ys` make up the 
first and second column, respectively. For each row, the elements are separated by `sep`. The
vectors `xs` and `ys` must have equal length. At most `nsamples` rows are written, and, if `xs` and
`ys` contain more than `nsamples` entries, the entries to write are chosen such that they are 
uniformly spaced out over the two vectors.
"""
function write_table(xs::AbstractVector, ys::AbstractVector, filename::AbstractString; prefix="./results", sep=" ", nsamples=min(100, length(xs)))
    length(xs) == length(ys) || throw(DimensionMismatch("xs has dimension $(length(xs)), but ys has dimension $(length(ys))"))
    path = joinpath(prefix, filename)    
    mkpath(dirname(path))
    open(path, "w") do io
        for i in round.(Int, range(1, length(xs), length=nsamples))
            write(io, "$(xs[i])$(sep)$(ys[i])\n")
        end
    end
    return
end

"""

Return a vector composed of the subset of the elements of `vs` that lie between the `p1`-th and 
`p2`-th percentiles, where `p1 < p2`.
"""
function in_percentiles(vs::AbstractVector, p1::Real=0.01, p2::Real=1-p1)
    0 <= p1 < p2 || throw(ArgumentError("p1 must be positive, and p2 must be greater than p1, but got p1=$p1 and p2=$p2"))
    q1, q2 = quantile(vs, p1), quantile(vs, p2)
    filter((x)->q1<=x<=q2, vs)
end

"""

Return a `DataFrame` mapping each unique combination of the entries of `cols` to the number of 
jobs in `df` with that particular combination.
"""
njobs_df(df; cols=[:nworkers, :nwait, :worker_flops, :nbytes, :nsubpartitions]) = sort!(combine(groupby(df, cols), :jobid => ((x)->length(unique(x))) => :njobs), cols)

### pre-processing

"""

Update the worker indices such that the fastest worker has index `1` and so on.
"""
function reindex_workers_by_order!(df)
    maxworkers = maximum(df.nworkers)
    latencies = zeros(maxworkers)
    repochs = zeros(Int, maxworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    repoch_columns = ["repoch_worker_$i" for i in 1:maxworkers]
    for i in 1:size(df, 1)
        latencies .= Inf
        nworkers = df.nworkers[i]
        for j in 1:nworkers
            repoch = df[i, repoch_columns[j]]
            isstraggler = ismissing(repoch) || repoch < df.iteration[i]
            latencies[j] = isstraggler ? Inf : df[i, latency_columns[j]]
        end
        latencies_view = view(latencies, 1:nworkers)
        p = sortperm(latencies_view)
        for j in 1:nworkers
            df[i, latency_columns[j]] = latencies[p[j]]
            df[i, repoch_columns[j]] = repochs[p[j]]
        end
    end
    df
end

"""

Remove columns not necessary for analysis (to save space).
"""
function strip_columns!(df)
    for column in ["iteratedataset", "saveiterates", "outputdataset", "inputdataset", "inputfile", "outputfile"]
        if column in names(df)
            select!(df, Not(column))
        end
    end
    df
end

"""

For each job and worker, to remove initialization latency (e.g., due to compilation), replace the 
latency of the first iteration by the average latency of the remaining iterations. For the update
latency, replace the latency of the 2 first iterations by the mean computed over the remaining 
iterations, since both of the 2 first iterations require compilation.
"""
function remove_initialization_latency!(df)
    sort!(df, [:jobid, :iteration])
    maxworkers = maximum(df.nworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    compute_latency_columns = ["compute_latency_worker_$i" for i in 1:maxworkers]
    for dfi in groupby(df, :jobid)
        nworkers = dfi.nworkers[1]
        for j in 1:nworkers
            dfi[1, latency_columns[j]] = mean(dfi[2:end, latency_columns[j]])
        end
        if compute_latency_columns[1] in names(df) && !ismissing(dfi[1, compute_latency_columns[1]])
            for j in 1:nworkers
                dfi[1, compute_latency_columns[j]] = mean(dfi[2:end, compute_latency_columns[j]])
            end
        end
        dfi[1, :latency] = mean(dfi[2:end, :latency])
        dfi[1:2, :update_latency] .= mean(dfi[3:end, :update_latency])
    end
    df
end

"""

Fix the value of `nwait` when `nwaitschedule < 1`.
"""
function fix_nwaitschedule!(df)
    if !("nwaitschedule" in names(df))
        return df
    end
    df.nwaitschedule .= Missings.replace(df.nwaitschedule, 1.0)
    df.nwait .= min.(max.(1, ceil.(Int, df.nwait .* df.nwaitschedule.^df.iteration)), df.nworkers)
    df
end

"""

Set the overall iteration latency equal to the latency of the `nwait`-th fastest worker, and move 
the difference between the overall recorded latency to `update_latency`. The motivation for doing 
so is that the difference between the latency of the `nwait`-th fastest worker and the overall 
recorded latency is due to delays at the coordinator, which should be counted as part of the 
update latency.
"""
function fix_update_latency!(df)
    maxworkers = maximum(df.nworkers)
    latencies = zeros(maxworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    repoch_columns = ["repoch_worker_$i" for i in 1:maxworkers]    
    for i in 1:size(df, 1)
        latencies .= Inf
        nworkers = df.nworkers[i]
        nwait = df.nwait[i]
        for j in 1:nworkers
            repoch = df[i, repoch_columns[j]]
            isstraggler = ismissing(repoch) || repoch < df.iteration[i]
            latencies[j] = isstraggler ? Inf : df[i, latency_columns[j]]
        end
        partialsort!(latencies, nwait)
        nwait_latency = latencies[nwait]
        df[i, :update_latency] += df[i, :latency] - nwait_latency
        df[i, :latency] = nwait_latency
    end
    df
end

function compute_cumulative_time!(df)
    sort!(df, [:jobid, :iteration])
    df.time .= combine(groupby(df, :jobid), :latency => cumsum => :time).time
    # df.time .+= combine(groupby(df, :jobid), :update_latency => cumsum => :time).time
    df
end

function vcat_job_dfs(args...)
    jobid = 0
    dfs = copy.(args)
    for df in dfs
        df[:, :jobid] .+= jobid
        jobid = maximum(df.jobid)        
    end
    vcat(dfs..., cols=:union)
end

### timeseries plots

"""

Plot the iteration latency of workers with indices in `worker_indices` of job `jobid`.
"""
function plot_timeseries(df, jobid, worker_indices; separate=true, cumulative=false)
    println("Plotting per-worker iteration latency for workers: $worker_indices of jobid: $jobid")
    df = filter(:jobid => (x)->x==jobid, df)
    sort!(df, :iteration)
    plt.figure()
    for worker in worker_indices
        xs = df.iteration
        if separate
            # compute
            ys = df[:, "compute_latency_worker_$worker"]
            ys = cumulative ? cumsum(ys) : ys
            plt.plot(xs, ys, label="Worker $worker (comp.)")
            # write_table(xs[1:100], ys[1:100], "timeseries_compute_$(jobid)_$(worker).csv")

            # communication
            ys = df[:, "latency_worker_$worker"] .- df[:, "compute_latency_worker_$worker"]
            ys = cumulative ? cumsum(ys) : ys
            plt.plot(xs, ys, label="Worker $worker (comm.)")
            # write_table(xs[1:100], ys[1:100], "timeseries_communication_$(jobid)_$(worker).csv")
        else
            ys = df[:, "latency_worker_$worker"]
            ys = cumulative ? cumsum(ys) : ys
            plt.plot(xs, ys, label="Worker $worker (total)")
            # write_table(xs[1:100], ys[1001:1100], "timeseries_$(jobid)_$(worker).csv", nsamples=600)            
        end
    end    
    plt.grid()
    plt.legend()
    plt.title("Job $jobid")
    plt.xlabel("Iteration")
    plt.ylabel("Per-worker iteration latency [s]")
    plt.tight_layout()
    return
end

"""

Plot the total iteration latency for job `jobid`.
"""
function plot_timeseries(df, jobid; cumulative=false, withupdate=true)
    println("Plotting total iteration latency for jobid: $jobid")
    df = filter(:jobid => (x)->x==jobid, df)
    sort!(df, :iteration)    

    xs = df.iteration
    ys = df.latency
    if withupdate
        ys .+= df.update_latency
    end
    ys = cumulative ? cumsum(ys) : ys
    plt.figure()    
    plt.plot(xs, ys)
    # write_table(xs, ys, "cumulative_time_$(jobid).csv")

    plt.grid()
    plt.legend()
    plt.title("Job $jobid")
    plt.xlabel("Iteration")
    plt.ylabel("Total iteration latency [s]")
    plt.tight_layout()
    return
end

### per-worker latency distribution

"""

Plot the latency distribution of individual workers.
"""
function plot_worker_latency_distribution(df, jobid, worker_indices=[1, 2]; dist=Gamma, prune=0.01)
    df = filter(:jobid => (x)->x==jobid, df)
    worker_flops = df.worker_flops[1]
    nbytes = df.nbytes[1]    
    nworkers = df.nworkers[1]    
    plt.figure()
    plt.title("job $jobid ($(round(worker_flops, sigdigits=3)) flops, $nbytes bytes, sparse matrix)")    
    println("Plotting per-worker latency distribution of workers $worker_indices for job $jobid") 
    println("nbytes: $nbytes, nflops: $worker_flops, niterations: $(size(df, 1))")

    latency_columns = ["latency_worker_$i" for i in 1:nworkers]
    compute_latency_columns = ["compute_latency_worker_$i" for i in 1:nworkers]

    # overall
    plt.subplot(3, 1, 1)
    for i in worker_indices        
        xs = sort(df[:, latency_columns[i]])
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys, label="Worker $i")
        # write_table(xs, ys, "cdf_$(jobid)_$(i).csv")
        d = Distributions.fit(dist, in_percentiles(xs, prune))
        xs = range(quantile(d, 0.000001), quantile(d, 0.999999), length=100)
        ys = cdf.(d, xs)
        # write_table(xs, ys, "cdf_fit_$(jobid)_$(i).csv")
        plt.plot(xs, ys, "k--")
    end
    plt.xlabel("Overall per-worker iteration latency [s]")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()    

    # communication    
    plt.subplot(3, 1, 2)
    for i in worker_indices        
        xs = df[:, latency_columns[i]] .- df[:, compute_latency_columns[i]]
        sort!(xs)
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys, label="Worker $i")
        # write_table(xs, ys, "cdf_communication_$(jobid)_$(i).csv")
        d = Distributions.fit(dist, in_percentiles(xs, prune))
        xs = range(quantile(d, 0.000001), quantile(d, 0.999999), length=100)
        ys = cdf.(d, xs)
        plt.plot(xs, ys, "k--")
    end
    plt.xlabel("Per-worker comm. latency [s]")
    plt.ylabel("CDF")
    plt.grid()        

    # computation
    plt.subplot(3, 1, 3)
    for i in worker_indices
        xs = sort(df[:, compute_latency_columns[i]])
        ys = range(0, 1, length=length(xs))
        plt.plot(xs, ys, label="Worker $i")
        # write_table(xs, ys, "cdf_compute_$(jobid)_$(i).csv")
        d = Distributions.fit(dist, in_percentiles(xs, prune))
        xs = range(quantile(d, 0.000001), quantile(d, 0.999999), length=100)
        ys = cdf.(d, xs)
        # write_table(xs, ys, "cdf_fit_compute_$(jobid)_$(i).csv")
        plt.plot(xs, ys, "k--")
    end
    plt.xlabel("Per-worker comp. latency [s]")
    plt.ylabel("CDF")
    plt.grid()

    plt.tight_layout()
    return
end

### distribution of the mean and variance of the per-worker latency

"""

Return a `DataFrame` composed of the mean and variance of the per-iteration latency for each job 
and worker, where each row of the `DataFrame` corresponds to a particular job and worker. Ignores
jobs with less than `miniterations` iterations.
"""
function per_worker_statistics_df(df; miniterations=100, prune=0.01)
    df = filter([:nwait, :nworkers] => (x,y)->x==y, df)
    rv = DataFrame()
    row = Dict{String, Any}()
    maxworkers = maximum(df.nworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]    
    compute_latency_columns = ["compute_latency_worker_$i" for i in 1:maxworkers]    
    for jobid in unique(df.jobid)
        dfi = filter(:jobid => (x)->x==jobid, df)
        if size(dfi, 1) < miniterations
            continue
        end
        nworkers = dfi.nworkers[1]
        row["nworkers"] = nworkers
        row["worker_flops"] = dfi.worker_flops[1]
        row["nbytes"] = dfi.nbytes[1]
        row["nsamples"] = size(dfi, 1)
        row["jobid"] = jobid
        for i in 1:nworkers

            if "compute_latency_worker_1" in names(dfi) && !ismissing(dfi[1, compute_latency_columns[1]])

                # compute latency                
                ys = in_percentiles(float.(dfi[:, compute_latency_columns[i]]), prune)
                row["comp_mean"] = mean(ys)
                row["comp_var"] = var(ys)

                # communication latency
                ys = in_percentiles(float.(dfi[:, latency_columns[i]] .- dfi[:, compute_latency_columns[i]]), prune)      
                row["comm_mean"] = mean(ys)
                row["comm_var"] = var(ys)
            else
                row["comp_mean"] = missing
                row["comp_var"] = missing
                row["comm_mean"] = missing
                row["comm_var"] = missing
            end

            # overall latency
            ys = in_percentiles(float.(dfi[:, latency_columns[i]]), prune)
            row["mean"] = mean(ys)
            row["var"] = var(ys)

            # record
            row["worker_index"] = i
            push!(rv, row, cols=:union)
        end
    end
    rv
end

"""

Plot the distribution of the mean and variance of the per-worker latency.
"""
function plot_worker_mean_var_distribution(dfg; nflops, nbytes=30048, prune=0.01)

    plt.figure()

    # total latency
    dfi = dfg
    dfi = filter(:worker_flops => (x)->isapprox(x, nflops, rtol=1e-2), dfi)
    dfi = filter(:nbytes => (x)->x==nbytes, dfi)

    ## mean cdf
    plt.subplot(3, 3, 1)    

    xs = sort(dfi.mean)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys, label="nbytes: $nbytes")
    # write_table(xs, ys, "cdf_comm_mean_$(nbytes).csv")

    # fitted distribution
    if size(dfi, 1) >= 100
        d = Distributions.fit(LogNormal, in_percentiles(xs, prune)) 
        xs = 10.0.^range(log10(quantile(d, 0.0001)), log10(quantile(d, 0.9999)), length=100)
        ys = cdf.(d, xs)
        plt.plot(xs, ys, "k--")
        # write_table(xs, ys, "cdf_comm_mean_fit_$(nbytes).csv")
    end
    
    plt.ylabel("CDF")
    plt.xlabel("Mean (total latency)")

    ## var cdf
    plt.subplot(3, 3, 2)
    xs = sort(dfi.var)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys, label="nbytes: $nbytes")
    # write_table(xs, ys, "cdf_comm_var_$(nbytes).csv")

    # fitted distribution
    if size(dfi, 1) >= 100
        d = Distributions.fit(LogNormal, in_percentiles(xs, prune))
        xs = 10.0.^range(log10(quantile(d, 0.0001)), log10(quantile(d, 0.9999)), length=100)            
        ys = cdf.(d, xs)
        plt.plot(xs, ys, "k--")
        # write_table(xs, ys, "cdf_comm_var_fit_$(nbytes).csv")
    end

    plt.ylabel("CDF")
    plt.xlabel("Variance (total latency)")
    plt.xscale("log")

    # mean-var scatter
    plt.subplot(3, 3, 3)
    xs = dfi.mean
    ys = dfi.var
    plt.plot(xs, ys, ".", label="nbytes: $nbytes")
    # write_table(xs, ys, "scatter_comm_$(nbytes).csv", nsamples=200)

    plt.xlabel("Mean (total latency)")
    plt.ylabel("Variance (total latency)")
    plt.yscale("log")

    # communication latency
    dfi = dfg
    dfi = filter(:nbytes => (x)->x==nbytes, dfi)

    # choose only fast or slow workers (for AWS c5.xlarge)
    # dfi = filter(:comm_mean => (x)->x<0.00026, dfi)
    # dfi = filter(:comm_var => (x)->x<1e-9, dfi)

    ## mean cdf
    plt.subplot(3, 3, 4)
    xs = sort(dfi.comm_mean)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys, label="nbytes: $nbytes")
    # write_table(xs, ys, "cdf_comm_mean_$(nbytes).csv")

    # fitted distribution
    if size(dfi, 1) >= 100
        d = Distributions.fit(LogNormal, in_percentiles(xs, prune))
        xs = 10.0.^range(log10(quantile(d, 0.0001)), log10(quantile(d, 0.9999)), length=100)
        ys = cdf.(d, xs)
        plt.plot(xs, ys, "k--")
        # write_table(xs, ys, "cdf_comm_mean_fit_$(nbytes).csv")
    end
    plt.ylabel("CDF")
    plt.xlabel("Avg. comm. latency")

    ## var cdf
    plt.subplot(3, 3, 5)
    xs = sort(dfi.comm_var)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys, label="nbytes: $nbytes")
    # write_table(xs, ys, "cdf_comm_var_$(nbytes).csv")

    # fitted distribution
    if size(dfi, 1) >= 100
        d = Distributions.fit(LogNormal, in_percentiles(xs, prune))
        xs = 10.0.^range(log10(quantile(d, 0.0001)), log10(quantile(d, 0.9999)), length=100)            
        ys = cdf.(d, xs)
        plt.plot(xs, ys, "k--")            
        # write_table(xs, ys, "cdf_comm_var_fit_$(nbytes).csv")
    end
    plt.ylabel("CDF")
    plt.xlabel("Comm. latency var")
    plt.xscale("log")

    # mean-var scatter
    plt.subplot(3, 3, 6)
    xs = dfi.comm_mean
    ys = dfi.comm_var
    plt.plot(xs, ys, ".", label="nbytes: $nbytes")
    # write_table(xs, ys, "scatter_comm_$(nbytes).csv", nsamples=200)

    plt.xlabel("Mean (comm. latency)")
    plt.ylabel("Variance (comm. latency)")
    plt.yscale("log")    

    # compute latency
    dfi = dfg
    dfi = filter(:worker_flops => (x)->isapprox(x, nflops, rtol=1e-2), dfi)

    ## mean cdf
    plt.subplot(3, 3, 7)
    xs = sort(dfi.comp_mean)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys, label="nflops: $(round(nflops, sigdigits=3))")
    # write_table(xs, ys, "cdf_comp_mean_$(round(nflops, sigdigits=3)).csv")

    # fitted distribution
    if size(dfi, 1) >= 100
        d = Distributions.fit(LogNormal, in_percentiles(xs, prune))
        xs = 10.0.^range(log10(quantile(d, 0.0001)), log10(quantile(d, 0.99999)), length=100)            
        ys = cdf.(d, xs)
        plt.plot(xs, ys, "k--")                        
        # write_table(xs, ys, "cdf_comp_mean_fit_$(round(nflops, sigdigits=3)).csv")
    end
    plt.ylabel("CDF")
    plt.xlabel("Mean (comp. latency)")
    plt.xscale("log")
    plt.legend()

    ## var cdf
    plt.subplot(3, 3, 8)
    xs = sort(dfi.comp_var)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys, label="nflops: $nflops")
    # write_table(xs, ys, "cdf_comp_var_$(round(nflops, sigdigits=3)).csv")

    # fitted distribution
    if size(dfi, 1) >= 100
        i = round(Int, 0.01*length(xs))
        xs = xs[1:end-i]
        d = Distributions.fit(LogNormal, in_percentiles(xs, prune))
        xs = 10.0.^range(log10(quantile(d, 0.0001)), log10(quantile(d, 0.99999)), length=100)            
        ys = cdf.(d, xs)
        plt.plot(xs, ys, "k--")
        # write_table(xs, ys, "cdf_comp_var_fit_$(round(nflops, sigdigits=3)).csv")
    end
    plt.ylabel("CDF")
    plt.xlabel("Variance (comp. latency)")
    plt.xscale("log")

    # mean-var scatter
    plt.subplot(3, 3, 9)
    xs = dfi.comp_mean
    ys = dfi.comp_var
    plt.plot(xs, ys, ".", label="nflops: $nflops")
    # write_table(xs, ys, "scatter_comp_$(round(nflops, sigdigits=3)).csv", nsamples=200)
    plt.xlabel("Mean (comp. latency)")
    plt.ylabel("Variance (comp. latency)")
    plt.xscale("log")
    plt.yscale("log")        
end

### order statistics

"""

Plot order statistics latency for a particular job, and predicted order statistics latency.
"""
function plot_orderstats(df, jobid; maxiterations=Inf, p1=0.01, p2=1-p1)
    df = filter(:jobid => (x)->x==jobid, df)
    df = filter(:iteration => (x)->x<=maxiterations, df)
    sort!(df, :iteration)
    nbytes, nflops = df.nbytes[1], df.worker_flops[1]
    println("nbytes: $nbytes, nflops: $nflops, niterations: $(size(df, 1))")
    nworkers = df.nworkers[1]
    niterations = floor(Int, min(df.niterations[1], maxiterations))
    orderstats = zeros(nworkers)
    buffer = zeros(nworkers)
    latency_columns = ["latency_worker_$(i)" for i in 1:nworkers]    
    compute_latency_columns = ["compute_latency_worker_$(i)" for i in 1:nworkers]    

    plt.figure() 

    # empirical orderstats
    for i in 1:niterations
        for j in 1:nworkers
            buffer[j] = df[i, latency_columns[j]]
        end
        sort!(buffer)
        orderstats += buffer
    end
    orderstats ./= niterations
    plt.plot(1:nworkers, orderstats, label="Empirical")
    # write_table(1:nworkers, orderstats, "orderstats_$(jobid).csv")

    # compute the global mean and variance
    m_comm, v_comm = 0.0, 0.0
    m_comp, v_comp = 0.0, 0.0
    for j in 1:nworkers
        vs = in_percentiles(df[:, latency_columns[j]] .- df[:, compute_latency_columns[j]], p1, p2)
        m_comm += mean(vs)
        v_comm += var(vs)
        vs = in_percentiles(df[:, compute_latency_columns[j]], p1, p2)
        m_comp += mean(vs)
        v_comp += var(vs)
    end
    m_comm /= nworkers
    v_comm /= nworkers
    m_comp /= nworkers
    v_comp /= nworkers

    # global gamma
    θ = v_comm / m_comm
    α = m_comm / θ
    d_comm = Gamma(α, θ)
    θ = v_comp / m_comp
    α = m_comp / θ
    d_comp = Gamma(α, θ)    
    orderstats .= 0
    for i in 1:niterations
        for j in 1:nworkers
            buffer[j] = rand(d_comm) + rand(d_comp)
        end
        sort!(buffer)
        orderstats += buffer
    end    
    orderstats ./= niterations
    plt.plot(1:nworkers, orderstats, "--", label="Global Gamma")
    # write_table(1:nworkers, orderstats, "orderstats_global_gamma_$(jobid).csv")

    # global shiftexp
    θ = sqrt(v_comm)
    s = m_comm - θ
    d_comm = ShiftedExponential(s, θ)
    θ = sqrt(v_comp)
    s = m_comp - θ
    d_comp = ShiftedExponential(s, θ)
    orderstats .= 0
    for i in 1:niterations
        for j in 1:nworkers
            buffer[j] = rand(d_comm) + rand(d_comp)
        end        
        sort!(buffer)
        orderstats += buffer
    end    
    orderstats ./= niterations
    plt.plot(1:nworkers, orderstats, "--", label="Global ShiftExp")
    # write_table(1:nworkers, orderstats, "orderstats_global_shiftexp_$(jobid).csv")

    # independent orderstats (gamma)
    ds = [Distributions.fit(Gamma, in_percentiles(df[:, latency_columns[j]], p1, p2)) for j in 1:nworkers]
    orderstats .= 0
    for _ in 1:niterations
        for j in 1:nworkers
            buffer[j] = rand(ds[j])
        end
        sort!(buffer)
        orderstats += buffer
    end
    orderstats ./= niterations    
    plt.plot(1:nworkers, orderstats, "k--", label="Gamma")

    # independent orderstats (shiftexp)
    ds = [Distributions.fit(ShiftedExponential, in_percentiles(df[:, latency_columns[j]], p1, p2)) for j in 1:nworkers]
    orderstats .= 0
    for _ in 1:niterations
        for j in 1:nworkers
            buffer[j] = rand(ds[j])
        end
        sort!(buffer)
        orderstats += buffer
    end
    orderstats ./= niterations    
    plt.plot(1:nworkers, orderstats, "r--", label="ShiftExp")    

    # independent orderstats w. separate communication and compute, both of which are gamma
    ds_comm = [Distributions.fit(Gamma, in_percentiles(df[:, latency_columns[j]] .- df[:, compute_latency_columns[j]], p1, p2)) for j in 1:nworkers]
    ds_comp = [Distributions.fit(Gamma, in_percentiles(df[:, compute_latency_columns[j]], p1, p2)) for j in 1:nworkers]            
    orderstats .= 0
    for _ in 1:niterations
        for j in 1:nworkers
            buffer[j] = rand(ds_comm[j]) + rand(ds_comp[j])
        end
        sort!(buffer)
        orderstats += buffer
    end
    orderstats ./= niterations
    plt.plot(1:nworkers, orderstats, "c.", label="Gamma-Gamma (ind., sep.)")    
    # write_table(1:nworkers, orderstats, "orderstats_gamma_gamma_$(jobid).csv")

    # dependent orderstats (using a Normal copula)
    # first, compute the empirical correlation matrix
    μ = zeros(nworkers)
    Σ = Matrix(1.0.*I, nworkers, nworkers)
    for i in 1:nworkers
        vsi = df[:, latency_columns[i]] .- df[:, compute_latency_columns[i]]
        for j in (i+1):nworkers
            vsj = df[:, latency_columns[j]] .- df[:, compute_latency_columns[j]]
            Σ[i, j] = cor(vsi, vsj)
        end
    end

    # the resulting correlation matrix may be non-positive-definite, and,
    # if so, we replace it with the "closest" positive-definite matrix
    if !isposdef(Symmetric(Σ))
        F = eigen(Symmetric(Σ))
        replace!((x)->max(sqrt(eps(Float64)), x), F.values)
        Σ = F.vectors*Diagonal(F.values)*F.vectors'
    end

    # copula
    ds_comm = [Distributions.fit(Gamma, in_percentiles(df[:, latency_columns[j]] .- df[:, compute_latency_columns[j]], p1, p2)) for j in 1:nworkers]
    ds_comp = [Distributions.fit(Gamma, in_percentiles(df[:, compute_latency_columns[j]], p1, p2)) for j in 1:nworkers]         
    copula = MvNormal(μ, Symmetric(Σ))
    normal = Normal() # standard normal
    sample = zeros(nworkers)
    orderstats .= 0
    for _ in 1:niterations
        Distributions.rand!(copula, sample) # sample from the MvNormal
        for j in 1:nworkers
            buffer[j] = quantile(ds_comm[j], cdf(normal, sample[j])) # Convert to uniform and then to the correct marginal
            buffer[j] += rand(ds_comp[j]) # add compute latency
        end
        sort!(buffer)
        orderstats += buffer
    end
    orderstats ./= niterations
    plt.plot(1:nworkers, orderstats, "m--", label="Simulated (copula)")
    # write_table(1:nworkers, orderstats, "orderstats_gamma_gamma_copula_$(jobid).csv")

    plt.legend()
    plt.grid()
    plt.xlabel("Order")
    plt.ylabel("Latency [s]")
    return
end

### autocorrelation and cross-correlation

"""

Plot the latency auto-correlation function averaged over many realizations of the process.
"""
function plot_autocorrelation(df; nflops, nbytes=30048, maxlag=100, latency="total")
    df = filter(:worker_flops => (x)->isapprox(x, nflops, rtol=1e-2), df)
    df = filter(:nbytes => (x)->x==nbytes, df)
    sort!(df, [:jobid, :iteration])    
    maxworkers = maximum(df.nworkers)
    ys = zeros(maxlag)
    nsamples = zeros(Int, maxlag)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    compute_latency_columns = ["compute_latency_worker_$i" for i in 1:maxworkers]
    for jobid in unique(df.jobid)
        dfi = filter(:jobid => (x)->x==jobid, df)
        nworkers = dfi.nworkers[1]
        lags = 0:min(maxlag-1, size(dfi, 1)-1)
        for i in 1:nworkers
            if latency == "total"
                vs = float.(dfi[:, latency_columns[i]]) # total latency
            elseif latency == "communication"
                vs = float.(dfi[:, latency_columns[i]] .- dfi[:, compute_latency_columns[i]]) # communication latency
            elseif latency == "compute"
                vs = float.(dfi[:, compute_latency_columns[i]]) # compute latency
            else
                error("latency must be one of [total, communication, compute]")
            end
            ys[lags.+1] .+= autocor(vs, lags)
            nsamples[lags.+1] .+= 1
        end
    end
    ys ./= nsamples
    plt.figure()
    xs = 0:(maxlag-1)
    plt.plot(xs, ys)
    # write_table(xs, ys, "ac_$(latency)_$(round(nflops, sigdigits=3))_$(nbytes).csv", nsamples=maxlag)
    plt.xlabel("Lag (iterations)")
    plt.ylabel("Auto-correlation")
    return
end

"""

Plot the CDF of the correlation between pairs of workers.
"""
function plot_worker_latency_cov_cdf(df; nflops, nbytes=30048, maxworkers=108, latency="total", minsamples=10)
    df = filter(:worker_flops => (x)->isapprox(x, nflops, rtol=1e-2), df)
    df = filter(:nbytes => (x)->x==nbytes, df)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    compute_latency_columns = ["compute_latency_worker_$i" for i in 1:maxworkers]
    xs = zeros(0)
    xsr = zeros(0)
    for jobid in unique(df.jobid)
        dfi = filter(:jobid => (x)->x==jobid, df)
        if size(dfi, 1) < minsamples
            continue
        end
        nworkers = min(maxworkers, dfi.nworkers[1])
        for i in 1:nworkers
            if latency == "total"
                xsi = float.(dfi[:, latency_columns[i]])
            elseif latency == "communication"
                xsi = dfi[:, latency_columns[i]] .- dfi[:, compute_latency_columns[i]]            
            elseif latency == "compute"
                xsi = dfi[:, compute_latency_columns[i]]
            else
                error("latency must be in [total, communication, compute]")
            end
            if minimum(xsi) <= 0
                continue
            end
            try
                rvi = Distributions.fit(Gamma, xsi)
            catch e
                return xsi
            end
            rvi = Distributions.fit(Gamma, xsi)
            rsamplesi = rand(rvi, size(dfi, 1))
            for j in 1:nworkers
                if j == i
                    continue
                end
                if latency == "total"
                    xsj = float.(dfi[:, latency_columns[j]])
                elseif latency == "communication"
                    xsj = dfi[:, latency_columns[j]] .- dfi[:, compute_latency_columns[j]]            
                elseif latency == "compute"
                    xsj = dfi[:, compute_latency_columns[j]]
                else
                    error("latency must be in [total, communication, compute]")
                end
                if minimum(xsj) <= 0
                    continue
                end
                push!(xs, cor(xsi, xsj))
                
                rvj = Distributions.fit(Gamma, xsj)
                rsamplesj = rand(rvj, size(dfi, 1))
                push!(xsr, cor(rsamplesi, rsamplesj))
            end
        end
    end

    # empirical cdf
    plt.figure()    
    sort!(xs)
    ys = range(0, 1, length=length(xs))
    plt.plot(xs, ys)
    # write_table(xs, ys, "cov_cdf_$(latency)_$(round(nflops, sigdigits=3))_$(nbytes).csv")

    # independent cdf
    sort!(xsr)
    plt.plot(xsr, ys, "k--")
    # write_table(xsr, ys, "cov_cdf_ind_$(latency)_$(round(nflops, sigdigits=3))_$(nbytes).csv")
    return
end

### latency model code

"""

Return distributions of type `d` (either `Gamma` or `ShiftedExponential`) fit to the communication,
compute, and total latency of each worker in the job with id `jobid`.
"""
function fit_worker_distributions(dfg, jobid; d=Gamma)
    d == Gamma || d == ShiftedExponential || throw(ArgumentError("d must be either Gamma or ShiftedExponential"))
    dfg = filter(:jobid => (x)->x==jobid, dfg)
    size(dfg, 1) > 0 || error("job $jobid doesn't exist")
    sort!(dfg, :worker_index)
    nworkers = dfg.nworkers[1]
    if ismissing(dfg.comp_mean[1])
        ds_comm = [nothing for _ in 1:nworkers]
        ds_comp = [nothing for _ in 1:nworkers]
    else
        ds_comm = distribution_from_mean_variance.(d, dfg.comm_mean, dfg.comm_var)
        ds_comp = distribution_from_mean_variance.(d, dfg.comp_mean, dfg.comp_var)
    end
    ds_total = distribution_from_mean_variance.(d, dfg.mean, dfg.var)
    return ds_comm, ds_comp, ds_total
end


### prior distribution code

"""

Fit distributions to the mean and variance of the per-worker latency, and compute the correlation 
between the two, for each value of `nbytes` and `nflops`.
"""
function prior_statistics_df(dfg; prune=0.01, minsamples=10)

    # communication
    df_comm = DataFrame()
    row = Dict{String,Any}()
    nbytes_all = sort!(unique(dfg.nbytes))
    for nbytes in nbytes_all    
        dfi = filter(:nbytes => (x)->x==nbytes, dfg)

        # filter out extreme values
        q1, q2 = quantile(dfi.mean, prune), quantile(dfi.mean, 1-prune)
        dfi = filter(:mean => (x)->q1<=x<=q2, dfi)
        if size(dfi, 1) < minsamples
            continue
        end

        row["nbytes"] = nbytes
        row["mean_mean"] = mean(dfi.comm_mean)
        row["mean_var"] = var(dfi.comm_mean)
        row["mean_μ"], row["mean_σ"] = params(Distributions.fit(LogNormal, dfi.comm_mean))
        row["var_mean"] = mean(dfi.comm_var)
        row["var_var"] = var(dfi.comm_var)
        row["var_μ"], row["var_σ"] = params(Distributions.fit(LogNormal, dfi.comm_var))        
        row["cor"] = max.(cor(dfi.comm_mean, dfi.comm_var), 0)
        row["nsamples"] = size(dfi, 1)
        push!(df_comm, row, cols=:union)
    end
    row["nbytes"] = 0
    row["mean_mean"] = 0
    row["mean_var"] = 0
    row["mean_μ"], row["mean_σ"] = 0, 0
    row["var_mean"] = 0
    row["var_var"] = 0
    row["var_μ"], row["var_σ"] = 0, 0
    row["cor"] = 0
    row["nsamples"] = 1
    push!(df_comm, row, cols=:union)
    sort!(df_comm, :nbytes)

    # computation
    df_comp = DataFrame()
    row = Dict{String,Any}()
    nflops_all = sort!(unique(dfg.worker_flops))
    for nflops in nflops_all
        dfi = filter(:worker_flops => (x)->x==nflops, dfg)

        # filter out extreme values
        q1, q2 = quantile(dfi.mean, prune), quantile(dfi.mean, 1-prune)
        dfi = filter(:mean => (x)->q1<=x<=q2, dfi)
        if size(dfi, 1) < minsamples
            continue
        end        

        row["nflops"] = nflops
        row["mean_mean"] = mean(dfi.comp_mean)
        row["mean_var"] = var(dfi.comp_mean)
        row["mean_μ"], row["mean_σ"] = params(Distributions.fit(LogNormal, dfi.comp_mean))
        row["var_mean"] = mean(dfi.comp_var)
        row["var_var"] = var(dfi.comp_var)
        row["var_μ"], row["var_σ"] = params(Distributions.fit(LogNormal, dfi.comp_var))        
        row["cor"] = max.(cor(dfi.comp_mean, dfi.comp_var), 0)
        row["nsamples"] = size(dfi, 1)
        push!(df_comp, row, cols=:union)
    end
    row["nflops"] = 0
    row["mean_mean"] = 0
    row["mean_var"] = 0
    row["mean_μ"], row["mean_σ"] = 0, 0
    row["var_mean"] = 0
    row["var_var"] = 0
    row["var_μ"], row["var_σ"] = 0, 0
    row["cor"] = 0
    row["nsamples"] = 1
    push!(df_comp, row, cols=:union)    
    sort!(df_comp, :nflops)

    # total
    df_total = DataFrame()
    row = Dict{String,Any}()
    for dfi in groupby(dfg, [:nbytes, :worker_flops])
        nbytes = dfi.nbytes[1]
        worker_flops = dfi.worker_flops[1]

        # filter out extreme values
        q1, q2 = quantile(dfi.mean, prune), quantile(dfi.mean, 1-prune)
        dfi = filter(:mean => (x)->q1<=x<=q2, dfi)
        if size(dfi, 1) < minsamples
            continue
        end

        row["nbytes"] = nbytes
        row["nflops"] = worker_flops
        row["mean_mean"] = mean(dfi.mean)
        row["mean_var"] = var(dfi.mean)
        row["mean_μ"], row["mean_σ"] = params(Distributions.fit(LogNormal, dfi.mean))
        row["var_mean"] = mean(dfi.var)
        row["var_var"] = var(dfi.var)
        row["var_μ"], row["var_σ"] = params(Distributions.fit(LogNormal, dfi.var))        
        row["cor"] = max.(cor(dfi.mean, dfi.var), 0)
        row["nsamples"] = size(dfi, 1)
        push!(df_total, row, cols=:union)
    end
    row["nbytes"] = 0
    row["nflops"] = 0
    row["mean_mean"] = 0
    row["mean_var"] = 0
    row["mean_μ"], row["mean_σ"] = 0, 0
    row["var_mean"] = 0
    row["var_var"] = 0
    row["var_μ"], row["var_σ"] = 0, 0
    row["cor"] = 0
    row["nsamples"] = 1
    push!(df_total, row, cols=:union)
    sort!(df_total, [:nflops, :nbytes])

    df_comm, df_comp, df_total
end

"""

Interpolate between rows of the DataFrame `df`.
"""
function interpolate_df(df, x; key=:nflops)
    sort!(df, key)    
    if x in df[:, key]
        i = searchsortedfirst(df[:, key], x)
        return Dict(pairs(df[i, :]))
    end
    size(df, 1) > 1 || error("Need at least 2 samples for interpolation")

    # select the two closest points for which there is data
    j = searchsortedfirst(df[:, key], x)
    if j > size(df, 1)
        j = size(df, 1)  
        i = j - 1
    elseif j == 1
        j = 2
        i = 1
    else
        i = j - 1
    end

    # interpolate between, or extrapolate from, those points to x
    rv = Dict{Symbol,Any}()
    for name in names(df)
        slope = (df[j, name] - df[i, name]) / (df[j, key] - df[i, key])
        intercept = df[j, name] - df[j, key]*slope
        rv[Symbol(name)] = intercept + slope*x
    end
    rv
end

"""

Sample the latency disribution (either communication or computation) of a worker.
"""
function sample_worker_distribution(dfc, x; key, dist=Gamma, meanscale=1.0, varscale=1.0)
    row = interpolate_df(dfc, x; key)
    
    # mean-distribution
    μ = row[:mean_μ]
    σ = row[:mean_σ]
    d_mean = LogNormal(μ, σ)

    # var-distribution
    μ = row[:var_μ]
    σ = row[:var_σ]
    d_var = LogNormal(μ, σ)

    # copula
    c = row[:cor]
    Σ = [1 c; c 1]
    d = MvNormal(zeros(2), Σ)

    # sample the mean and variance of the per-iteration latency of this worker
    p = rand(d)
    m = quantile(d_mean, cdf(Normal(), p[1])) * meanscale
    v = quantile(d_var, cdf(Normal(), p[2])) * varscale
    
    # return a distribution of type d with the samples mean and variance
    distribution_from_mean_variance(dist, m, v)
end

sample_worker_comm_distribution(dfc_comm, nbytes) = sample_worker_distribution(dfc_comm, nbytes; key=:nbytes)
sample_worker_comp_distribution(dfc_comp, nflops) = sample_worker_distribution(dfc_comp, nflops; key=:nflops)

### code for simulating latency without accounting for interaction between iterations

function simulate_latency!(buffer, ds_comm, ds_comp; nsamples=1000)
    length(ds_comm) == length(ds_comp) || throw(DimensionMismatch("ds_comm has dimension $(length(ds_comm)), but ds_comp has dimension $(length(ds_comp))"))
    length(buffer) == length(ds_comp) || throw(DimensionMismatch("buffer has dimension $(length(buffer)), but ds_comp has dimension $(length(ds_comp))"))
    rv = zeros(length(ds_comm))
    for _ in 1:nsamples
        for i in 1:nworkers
            buffer[i] = 0
            if !isnothing(ds_comm)
                buffer[i] += rand(ds_comm[i])
            end
            if !isnothing(ds_comp)
                buffer[i] += rand(ds_comp[i])
            end
        end
        sort!(buffer)
        rv += buffer
    end
    rv ./= nsamples
end

"""

Simulate order statistics latency without accounting for the interaction between iterations, i.e.,
we assume that the next iteration doesn't start before the current iteration has finished, for a
particular set of per-worker latency distributions.
"""
simulate_latency(ds_comm, ds_comp) = simulate_latency!(zeros(length(ds_comm)), ds_comm, ds_comp)

"""

Simulate order statistics latency without accounting for the interaction between iterations, i.e.,
we assume that the next iteration doesn't start before the current iteration has finished.
"""
function simulate_latency(nbytes, nflops, nworkers; nsamples=1000, df_comm=nothing, df_comp=nothing)
    rv = zeros(nworkers)
    buffer = zeros(nworkers)
    for _ in 1:nsamples
        for i in 1:nworkers
            buffer[i] = 0
            if !isnothing(df_comm)
                buffer[i] += rand(sample_worker_comm_distribution(df_comm, nbytes))
            end
            if !isnothing(df_comp)
                buffer[i] += rand(sample_worker_comp_distribution(df_comp, nflops))
            end
        end
        sort!(buffer)
        rv += buffer
    end
    rv ./= nsamples
end

### code for simulating latency when the interaction between iterations is accounted for





"""

Simulate `niterations` iterations of the computation for `nruns` realizations of the set of workers.
"""
function simulate_iterations(nbytes::Real, nflops::Real; nruns=10, niterations=100, nworkers, nwait, df_comm, df_comp, update_latency, balanced=false)
    dfs = Vector{DataFrame}()
    for i in 1:nruns
        if !isnothing(df_comm)
            if !balanced
                ds_comm = [sample_worker_comm_distribution(df_comm, nbytes) for _ in 1:nworkers]
            else
                d = sample_worker_comm_distribution(df_comm, nbytes)
                ds_comm = [d for _ in 1:nworkers]
            end
        else
            ds_comm = [nothing for _ in 1:nworkers]
        end
        if !isnothing(df_comp)
            if !balanced
                ds_comp = [sample_worker_comp_distribution(df_comp, nflops) for _ in 1:nworkers]
            else
                d = sample_worker_comp_distribution(df_comp, nflops)
                ds_comp = [d for _ in 1:nworkers]
            end
        else
            ds_comp = [nothing for _ in 1:nworkers]
        end
        df = simulate_iterations(;nwait, niterations, ds_comm, ds_comp, update_latency)
        df[!, :jobid] .= i
        push!(dfs, df)
    end
    df = vcat(dfs...)
    df = combine(
        groupby(df, :iteration),
        :time => mean => :time,
        :update_latency => mean => :update_latency,
        :latency => mean => :latency,
        :idle_time => mean => :idle_time,
        :fresh_time => mean => :fresh_time,
        :stale_time => mean => :stale_time,
    )
    df[!, :nworkers] .= nworkers        
    df[!, :nwait] .= nwait
    df[!, :nbytes] .= nbytes
    df[!, :worker_flops] .= nflops
    df
end

### plots for comparing simulated and empirical latency

"""

Plot the average iteration latency (across realizations of the set of workers) vs. the number of workers.
"""
function plot_latency_vs_nworkers(;minworkers=10, maxworkers=500, nbytes::Real=30048, nflops0::Real=6.545178710898845e10/80, ϕ=1, df_comm, df_comp, update_latency=0.5e-3, df=nothing)

    plt.figure()
    # empirical iteration latency
    if !isnothing(df)
        df = filter([:worker_flops, :nworkers] => (x, y)->isapprox(x*y, nflops0, rtol=1e-2), df)
        df = filter(:nbytes => (x)->x==nbytes, df)
        df = filter([:nwait, :nworkers] => (x, y)->x==round(Int, ϕ*y), df) 
        xs = zeros(Int, 0)
        ys = zeros(0)
        for nworkers in unique(df.nworkers)
            dfi = filter(:nworkers => (x)->x==nworkers, df)
            dfi = combine(groupby(dfi, :jobid), :latency => mean => :latency)
            push!(xs, nworkers)
            push!(ys, mean(dfi.latency))
        end
        plt.plot(xs, ys, "s", label="Empiric")
        # write_table(xs, ys, "latency_vs_nworkers_$(round(nflops0, sigdigits=3))_$(round(ϕ, sigdigits=3)).csv")
    end

    # simulated iteration latency
    nworkerss = round.(Int, 10 .^ range(log10(minworkers), log10(maxworkers), length=10))
    latencies = zeros(length(nworkerss))
    # Threads.@threads 
    for i in 1:length(nworkerss)
        nworkers = nworkerss[i]
        nflops = nflops0 / nworkers
        nwait = max(1, round(Int, ϕ*nworkers))
        println("nworkers: $nworkers, nwait: $nwait, nflops: $(round(nflops, sigdigits=3))")
        df = simulate_iterations(nbytes, nflops; nworkers, nwait, df_comm, df_comp, update_latency)
        latencies[i] = mean(df.latency)
    end    
    plt.plot(nworkerss, latencies, "-", label="Predicted")
    # write_table(nworkerss, latencies, "latency_vs_nworkers_sim_$(round(nflops0, sigdigits=3))_$(round(ϕ, sigdigits=3)).csv")

    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of workers")
    plt.ylabel("Time [s]")
    return
end

function plot_time_vs_npartitions(;nbytes::Real=30048, nflops0::Real=6.545178710898845e10, nworkers, nwait, dfc_comm, dfc_comp, update_latency=0.5e-3)
    ps = 10 .^ range(log10(1), log10(320), length=10) .* nworkers
    idle_times = zeros(length(ps))
    fresh_times = zeros(length(ps))
    stale_times = zeros(length(ps))
    for (i, p) in enumerate(ps)
        df = simulate_iterations(nbytes, nflops0/p; nworkers, nwait, dfc_comm, dfc_comp, update_latency)
        idle_times[i] = mean(df.idle_time)
        fresh_times[i] = mean(df.fresh_time)
        stale_times[i] = mean(df.stale_time)
        total = idle_times[i] + fresh_times[i] + stale_times[i]
        idle_times[i] /= total
        fresh_times[i] /= total
        stale_times[i] /= total
    end
    
    plt.figure()
    plt.plot(ps, idle_times, ".-", label="Idle")
    plt.plot(ps, fresh_times, ".-", label="Fresh")
    plt.plot(ps, stale_times, ".-", label="Stale")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of data partitions")
    plt.ylabel("Time [s]")
    return
end

### prior order statistics

"""

Plot the average orderstats of the `iter`-th iteration computed over worker realizations.
"""
function plot_prior_orderstats(df; nworkers, nwait=nworkers, nbytes=30048, nflops, iters=10, model=nothing)
    df = filter(:nworkers => (x)->x==nworkers, df)
    df = filter(:nwait => (x)->x==nwait, df)
    df = filter(:nbytes => (x)->x==nbytes, df)    
    df = filter(:worker_flops => (x)->isapprox(x, nflops, rtol=1e-2), df)
    if size(df, 1) == 0
        error("no rows match nbytes: $nbytes and nflops: $nflops")
    end
    latency_columns = ["latency_worker_$(i)" for i in 1:maximum(df.nworkers)]
    repoch_columns = ["repoch_worker_$(i)" for i in 1:maximum(df.nworkers)]

    # plot empirical prior orderstats when all workers are available
    buffer = zeros(nworkers)
    orderstats = zeros(nwait)
    jobids = unique(df.jobid)
    nsamples = 0
    println("Computing orderstats over $(length(jobids)) jobs")
    for jobid in jobids
        dfi = filter(:jobid=>(x)->x==jobid, df)
        sort!(dfi, :iteration)
        for i in iters
            if i > size(dfi, 1)
                continue
            end
            for j in 1:nworkers
                if dfi[i, repoch_columns[j]] == dfi[i, :iteration]
                    buffer[j] = dfi[i, latency_columns[j]]
                else
                    buffer[j] = Inf
                end
            end
            sort!(buffer)
            orderstats += view(buffer, 1:nwait)
            nsamples += 1
        end
    end    
    orderstats ./= nsamples
    xs = 1:nwait
    plt.figure()
    plt.plot(xs, orderstats, "-o")
    # write_table(xs, orderstats, "prior_orderstats_$(nworkers)_$(nbytes)_$(round(nflops, sigdigits=3)).csv")

    # latency predicted by the model
    if !isnothing(model)
        df_comm, df_comp = model
        xs = 1:nworkers        
        ys = simulate_latency(nbytes, nflops, nworkers; df_comm, df_comp)
        plt.plot(xs, ys, "c--")
        # write_table(xs, ys, "prior_orderstats_niidm_$(nworkers)_$(nbytes)_$(round(nflops, sigdigits=3)).csv")
    end

    plt.grid()
    return    
end


### convergence plots

# function foo(dfj, v, p)
#     # I want the time at which the prob. of the loss being less than v is equal to p
#     # That means that 
#     # Let's convert the loss to a boolean (at most v)
#     # Next, take the average of that boolean 

#     # Split time into bins
#     # Convert the loss into a boolean (true if at most v)
#     # Count the number of samples in each bin
# end

"""

Plot the rate of convergence over time for DSAG, SAG, SGD, and coded computing. Let 
`latency=empirical` to plot against empirical latency, or let `latency=c5xlarge` to plot against 
latency computed by the model, fitted to traces recorded on `c5xlarge` instances.

rcv1full optimal value: 0.08294410910152755
"""
function plot_convergence(df, nworkers, opt=minimum(skipmissing(df.mse)); latency="empirical", niidm=nothing)
    df = filter(:nworkers => (x)->x==nworkers, df)
    df = filter(:nreplicas => (x)->x==1, df)
    df = filter(:mse => (x)->!ismissing(x), df)
    # sort!(df, [:jobid, :iteration])
    sort!(df, :iteration)
    println("nworkers: $nworkers, opt: $opt")

    # parameters are recorded as a tuple (nwait, nsubpartitions, stepsize)
    if nworkers == 10
        
        # varying the number of partitions
        # stepsize = 10.0
        # nwait = 10
        # params = [
        #     (nwait, 1, stepsize),
        #     (nwait, 2, stepsize),            
        #     (nwait, 5, stepsize),                        
        #     (nwait, 40, stepsize),            
        #     (nwait, 80, stepsize),         
        #     (nwait, 120, stepsize),         
        #     (nwait, 160, stepsize),         
        #     (nwait, 240, stepsize),
        #     (nwait, 320, stepsize),            
        # ]        

        # varying nwait
        stepsize = 10.0
        nwaits = [1, 2, 5, 9, 10]
        # nwaits = [10]
        # nwaits = collect(1:10)
        nsubpartitions = 240  
        params = [(nwait, nsubpartitions, stepsize) for nwait in nwaits]      

    elseif nworkers == 36

        # # varying npartitions
        # nwait = 3
        # params = [
        #     # (nwait, 10, 0.9),            
        #     # (nwait, 40, 0.9),
        #     # (nwait, 80, 0.9),
        #     (nwait, 160, 0.9),
        # ]

        # varying nwait      
        nsubpartitions = 160
        params = [
            (1, nsubpartitions, 0.9),
            (3, nsubpartitions, 0.9),            
            (6, nsubpartitions, 0.9),                        
            (9, nsubpartitions, 0.9),            
        ]
    elseif nworkers == 72

        # log. reg.
        nsubpartitions = 30
        nwaits = [1, 2, 5, 10, 36, 60, 65, 68, 70, 72]
        stepsize = 10
        nwaitschedule = 1.0
        params = [(nwait, nwaitschedule, nsubpartitions, stepsize) for nwait in nwaits]

        # # log. reg. w. nwaitschedule
        # nwaitschedules = [0.9330329915368074, 0.9930924954370359, 0.9993070929904525, 0.9999306876841536]
        # nwait = 72
        # stepsize = 10
        # nsubpartitions = 30
        # params = [(nwait, nwaitschedule, nsubpartitions, stepsize) for nwaitschedule in nwaitschedules]        

        # nsubpartitions = 160
        # params = [
        #     (1, nsubpartitions, 0.9),            
        #     (3, nsubpartitions, 0.9),            
        #     (6, nsubpartitions, 0.9),                        
        #     (9, nsubpartitions, 0.9),
        # ]

        # nwait = 9
        # params = [
        #     (nwait, 120, 0.9),            
        #     (nwait, 160, 0.9),            
        #     # (nwait, nsubpartitions, 0.9),                        
        #     # (nwait, nsubpartitions, 0.9),
        # ]   
    elseif nworkers == 108
        nsubpartitions = 160
        params = [
            (1, nsubpartitions, 0.9),            
            (3, nsubpartitions, 0.9),            
            (6, nsubpartitions, 0.9),                        
            (9, nsubpartitions, 0.9),
        ]      
        # nwait = 3
        # params = [
        #     (nwait, 120, 0.9),            
        #     (nwait, 160, 0.9),            
        #     (nwait, 240, 0.9),            
        #     (nwait, 320, 0.9),            
        #     (nwait, 640, 0.9),            
        # ]  
    else
        error("parameters not defined")
    end

    # plt.figure()    

    upscale = 1

    for (nwait, nwaitschedule, nsubpartitions, stepsize) in params
        
        dfi = df
        if nwaitschedule == 1
            dfi = filter(:nwait => (x)->x==nwait, dfi)
        end
        dfi = filter(:nwaitschedule => (x)->isapprox(x, nwaitschedule), dfi)        
        dfi = filter(:nsubpartitions => (x)->x==nsubpartitions, dfi)
        dfi = filter(:stepsize => (x)->isapprox(x, stepsize, rtol=1e-2), dfi)
        println("nwait: $nwait, nwaitschedule: $nwaitschedule, nsubpartitions: $nsubpartitions, stepsize: $stepsize")

        ### DSAG
        dfj = dfi
        dfj = dfj[dfj.variancereduced .== true, :]
        if nwait < nworkers # for nwait = nworkers, DSAG and SAG are the same
            dfj = dfj[dfj.nostale .== false, :]
        end
        # for simulations

        println("DSAG: $(length(unique(dfj.jobid))) jobs")
        # dfk = dfj
        # for dfj in groupby(dfk, :jobid)
        if size(dfj, 1) > 0
            dfj = combine(groupby(dfj, :iteration), :mse => mean => :mse, :time => mean => :time)
            if latency == "empirical"
                println("Plotting DSAG with empirical latency")
            else
                # dfj.time .= predict_latency(nwait, mean(dfi.worker_flops), nworkers; type=latency) .* dfj.iteration
                df_comm, df_comp = niidm
                nbytes = dfi.nbytes[1]
                nflops = dfi.worker_flops[1]
                update_latency = mean(dfi.update_latency)
                dfs = simulate_iterations(nbytes, nflops/upscale; niterations=maximum(dfj.iteration), nworkers=nworkers*upscale, nwait=nwait*upscale, df_comm, df_comp, update_latency)
                dfj.time .= dfs.time[dfj.iteration]
                println("Plotting DSAG with model latency for $latency")
            end
            xs = dfj.time
            # ys = opt.-dfj.mse
            ys = dfj.mse .- opt

            # xs = xs[2:end]
            # ys = diff(ys)

            plt.semilogy(xs, ys, "s--", label="DSAG w=$nwait, s: $nwaitschedule, p=$nsubpartitions")
            filename = "dsag_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"            
            write_table(xs, ys, filename)
        end
        println()
        # end

        # simulated latency (using the event-driven model)
        if !isnothing(niidm) && nwait == 1

            # # latency predicted by the new non-iid model
            # t_iter = predict_latency_niid(nbytes, nflops, nworkers; niidm)[nwait]
            # # t_iter = predict_latency(Nw, worker_flops, nworkers)
            # xs = (t_iter + 0.0022031946363636366) .* dfj.iteration
            # ys = opt .- dfj.mse
            # plt.plot(xs, ys, "k.")

            df_comm, df_comp = niidm
            dfs = simulate_iterations(nbytes, nflops; niterations=maximum(dfj.iteration), nworkers, nwait, df_comm, df_comp, update_latency=0.0022031946363636366)
            ys = opt .- dfj.mse
            xs = dfs.time[dfj.iteration]
            plt.plot(xs, ys, "k.")
            filename = "event_driven_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"
            write_table(xs, ys, filename)
        end

        ### SAG        
        dfj = dfi
        dfj = dfj[dfj.variancereduced .== true, :]
        dfj = dfj[dfj.nostale .== true, :]
        println("SAG: $(length(unique(dfj.jobid))) jobs")
        if size(dfj, 1) > 0
            dfj = combine(groupby(dfj, :iteration), :mse => mean => :mse, :time => mean => :time)
            if latency == "empirical"
                println("Plotting SAG with empirical latency")
            else
                # TODO: not checked
                # df_comm, df_comp = niidm
                # dfs = simulate_iterations(nbytes, nflops/upscale; niterations=maximum(dfj.iteration), nworkers=nworkers*upscale, nwait=nwait*upscale, df_comm, df_comp, update_latency=0.0022031946363636366)
                # ys = opt .- dfj.mse
                # dfj.time .= dfs.time[dfj.iteration]
                # println("Plotting DSAG with model latency for $latency")
            end
            xs = dfj.time
            ys = opt.-dfj.mse
            plt.semilogy(xs, ys, ".-", label="SAG w=$nwait, p=$nsubpartitions")
            filename = "sag_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"            
            write_table(xs, ys, filename)
        end
        println()
    end

    # Plot SAG
    # for nsubpartitions in sort!(unique(df.nsubpartitions))
    nsubpartitions = 160
    # for nsubpartitions in [80, 120, 160, 240, 320]
    stepsize = 0.9
    dfi = df
    dfi = dfi[dfi.nwait .== nworkers, :]
    dfi = dfi[dfi.variancereduced .== true, :]
    dfi = dfi[dfi.stepsize .== stepsize, :]    
    dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
    # dfi = dfi[dfi.nostale .== true, :]
    println("SAG p: $nsubpartitions, $(length(unique(dfi.jobid))) jobs")
    dfj = combine(groupby(dfi, :iteration), :mse => mean => :mse, :time => mean => :time)
    sort!(dfj, :iteration)    
    if size(dfj, 1) > 0
        if latency == "empirical"
            println("Plotting SAG with empirical latency")
        else
            df_comm, df_comp = niidm
            nflops = mean(dfi.worker_flops)        
            dfs = simulate_iterations(nbytes, nflops/upscale; balanced=true, nruns=50, niterations=maximum(dfj.iteration), nworkers=nworkers*upscale, nwait=nworkers*upscale, df_comm, df_comp, update_latency=0.0022031946363636366)
            ys = opt .- dfj.mse
            dfj.time .= dfs.time[dfj.iteration]
    
            println("Plotting SAG with model latency for $latency")
        end        
        xs = dfj.time
        ys = opt.-dfj.mse
        plt.semilogy(xs, ys, "o-", label="SAG p=$nsubpartitions")
        filename = "sag_$(nworkers)_$(nsubpartitions)_$(stepsize).csv"
        write_table(xs, ys, filename)        
    end

    # Plot SGD
    nsubpartitions = 160
    stepsize = 0.9
    dfi = df
    dfi = dfi[dfi.nwait .== nworkers, :]
    dfi = dfi[dfi.nsubpartitions .== nsubpartitions, :]
    dfi = dfi[dfi.variancereduced .== false, :]
    dfi = dfi[dfi.stepsize .== stepsize, :]
    println("SGD p: $nsubpartitions, $(length(unique(dfi.jobid))) jobs")
    dfj = combine(groupby(dfi, :iteration), :mse => mean => :mse, :time => mean => :time)
    sort!(dfj, :iteration)
    if size(dfj, 1) > 0
        if latency == "empirical"
            println("Plotting SGD with empirical latency")
        else
            # dfj.time .= predict_latency(nworkers, mean(dfi.worker_flops), nworkers; type=latency) .* dfj.iteration
    
            df_comm, df_comp = niidm
            nflops = mean(dfi.worker_flops)        
            dfs = simulate_iterations(nbytes, nflops/upscale; niterations=maximum(dfj.iteration), nworkers=nworkers*upscale, nwait=nworkers*upscale, df_comm, df_comp, update_latency=0.0022031946363636366)
            ys = opt .- dfj.mse
            dfj.time .= dfs.time[dfj.iteration]
    
            println("Plotting SGD with model latency for $latency")
        end            
        xs = dfj.time
        ys = opt.-dfj.mse
        plt.semilogy(xs, ys, "c^-", label="SGD p=$nsubpartitions")
        filename = "sgd_$(nworkers)_$(nsubpartitions)_$(stepsize).csv"
        write_table(xs, ys, filename)        
    end

    # # Plot GD
    # stepsize = 1.0
    # dfi = df
    # dfi = dfi[dfi.nwait .== nworkers, :]
    # dfi = dfi[dfi.nsubpartitions .== 1, :]
    # dfi = dfi[dfi.variancereduced .== false, :]
    # dfi = dfi[dfi.stepsize .== stepsize, :]
    # println("GD $(length(unique(dfi.jobid))) jobs")
    # dfj = by(dfi, :iteration, :mse => mean => :mse, :time => mean => :time)
    # if latency == "empirical"
    #     println("Plotting GD with empirical latency")
    # else
    #     # dfj.time .= predict_latency(nworkers, mean(dfi.worker_flops), nworkers; type=latency) .* dfj.iteration

    #     df_comm, df_comp = niidm
    #     nflops = mean(dfi.worker_flops)        
    #     dfs = simulate_iterations(nbytes, nflops/upscale; niterations=maximum(dfj.iteration), nworkers=nworkers*upscale, nwait=nworkers*upscale, df_comm, df_comp, update_latency=0.0022031946363636366)
    #     ys = opt .- dfj.mse
    #     dfj.time .= dfs.time[dfj.iteration]        

    #     println("Plotting GD with model latency for $latency")
    # end    
    # if size(dfj, 1) > 0
    #     xs = dfj.time
    #     ys = opt.-dfj.mse
    #     plt.semilogy(xs, ys, "ms-", label="GD")
    #     filename = "gd_$(nworkers)_$(stepsize).csv"
    #     write_table(xs, ys, filename)
    # end

    # # plot coded computing bound
    # if !isnothing(niidm)
    #     r = 2 # replication factor
    #     Nw = 1 # number of workers to wait for
    #     samp = 1 # workload up-scaling

    #     # get the average error per iteration of GD
    #     dfi = df
    #     dfi = dfi[dfi.nsubpartitions .== 1, :]
    #     dfi = dfi[dfi.nwait .== nworkers, :]
    #     dfi = dfi[dfi.stepsize .== 1, :]
    #     dfi = dfi[dfi.variancereduced .== false, :]
    #     dfi = dfi[dfi.nostale .== false, :]
    #     dfj = combine(groupby(dfi, :iteration), :mse => mean => :mse)
    #     sort!(dfj, :iteration)
    #     ys = opt .- dfj.mse

    #     # compute the iteration time for a scheme with a factor r replication
    #     @assert length(unique(dfi.worker_flops)) == 1
    #     worker_flops = r*mean(dfi.worker_flops)
    #     nbytes, = unique(dfi.nbytes)

    #     # latency predicted by the new non-iid model
    #     t_iter = predict_latency_niid(nbytes, worker_flops, nworkers; niidm)[Nw]
    #     # t_iter = predict_latency(Nw, worker_flops, nworkers)
    #     xs = t_iter .* dfj.iteration

    #     # # latency predicted by the event-driven model
    #     # df_comm, df_comp = niidm
    #     # dfs = simulate_iterations(nbytes, worker_flops; niterations=maximum(dfj.iteration), nworkers, nwait=Nw, df_comm, df_comp, update_latency=0.0022031946363636366)        
    #     # xs = dfs.time[dfj.iteration]
    #     # return dfs

    #     # make the plot
    #     plt.semilogy(xs, ys, "--k", label="Bound r: $r, Nw: $Nw")
    #     filename = "bound_$(nworkers)_$(stepsize).csv"
    #     write_table(xs, ys, filename)    
    # end

    # plt.xlim(1e-2, 1e2)
    plt.xscale("log")
    plt.grid(true)
    plt.legend()    
    plt.xlabel("Time [s]")
    plt.ylabel("Explained Variance Sub-optimality Gap")
    return
end

end