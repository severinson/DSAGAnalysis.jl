module DSAGAnalysis

using LinearAlgebra, SparseArrays
using CSV, DataFrames, PyPlot
using Random, StatsBase, Statistics, Distributions
using DataStructures
using HDF5, H5Sparse
using Polynomials

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
        t = size(dfi, 1)
        if "loadbalanced" in names(df)
            v = findfirst(!iszero, dfi.loadbalanced)
            if !isnothing(v)
                t = v - 1
            end
        end
        for j in 1:nworkers
            dfi[1, latency_columns[j]] = mean(dfi[2:t, latency_columns[j]])
        end
        if compute_latency_columns[1] in names(df) && !ismissing(dfi[1, compute_latency_columns[1]])
            for j in 1:nworkers
                dfi[1, compute_latency_columns[j]] = mean(dfi[2:t, compute_latency_columns[j]])
            end
        end
        dfi[1, :latency] = mean(dfi[2:t, :latency])
        dfi[1:2, :update_latency] .= mean(dfi[3:t, :update_latency])
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

Set the comm. latency of each worker equal to the global average.
"""
function equalize_comm_latency!(df)
    total = 0.0
    nsamples = 0
    maxworkers = maximum(df.nworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    compute_latency_columns = ["compute_latency_worker_$i" for i in 1:maxworkers]
    for j in 1:maxworkers
        vs = filter(!isnan, df[:, latency_columns[j]] - df[:, compute_latency_columns[j]])
        total += sum(vs)
        nsamples += length(vs)
    end
    total /= nsamples
    for j in 1:maxworkers
        df[:, latency_columns[j]] .= df[:, compute_latency_columns[j]] .+ total
    end
end

"""

Set the overall iteration latency equal to the latency of the `nwait`-th fastest worker, and move 
the difference between the overall recorded latency to `update_latency`. The motivation for doing 
so is that the difference between the latency of the `nwait`-th fastest worker and the overall 
recorded latency is due to delays at the coordinator, which should be counted as part of the 
update latency.
"""
function fix_update_latency!(df; increase_update_latency=false)
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
        if increase_update_latency
            df[i, :update_latency] += df[i, :latency] - nwait_latency
        end
        df[i, :latency] = nwait_latency
    end
    df
end

function compute_cumulative_time!(df)
    sort!(df, [:jobid, :iteration])
    df.time .= combine(groupby(df, :jobid), :latency => cumsum => :time).time
    df.time .+= combine(groupby(df, :jobid), :update_latency => cumsum => :time).time
    df
end

"""

Scale the compute latency of all workers by a factor `c`.
"""
function scale_compute_latency!(df, c::Real)
    maxworkers = maximum(df.nworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    compute_latency_columns = ["compute_latency_worker_$i" for i in 1:maxworkers]
    for i in 1:size(df, 1)
        nworkers = df.nworkers[i]
        for j in 1:nworkers
            comm_latency = df[i, latency_columns[j]] - df[i, compute_latency_columns[j]]
            df[i, compute_latency_columns[j]] *= c
            df[i, latency_columns[j]] = df[i, compute_latency_columns[j]] + comm_latency
        end
    end
    df = DSAGAnalysis.fix_update_latency!(df, increase_update_latency=false)
    df = DSAGAnalysis.compute_cumulative_time!(df)
    df
end

"""

Add a column `nfresh` to `df` indicating the number of fresh results received in each iteration.
"""
function add_nfresh!(df)
    nfresh = zeros(size(df, 1))
    maxworkers = maximum(df.nworkers)
    repoch_columns = ["repoch_worker_$i" for i in 1:maxworkers]
    for i in 1:size(df, 1)
        nworkers = df[i, :nworkers]
        iteration = df[i, :iteration]
        for j in 1:nworkers
            nfresh[i] += df[i, repoch_columns[j]] == iteration
        end
    end
    df[!, :nfresh] = nfresh
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

Find the transition point between regular operation and latency bursts.
"""
function find_corner(ys)
    ys = sort(ys)
    min_latency, max_latency = ys[1], ys[end]    
    rv = 0.0
    dmin = Inf
    for i in 1:length(ys)
        dx = (length(ys) - i) / length(ys)
        dy = (ys[i] - min_latency) / (max_latency - min_latency)
        d = sqrt(dx^2 + dy^2)
        if d < dmin
            rv = ys[i]
            dmin = d
        end
    end
    rv
end

"""

Return a DataFrame with column `latency`, containing the latency of the `worker_index`-th worker 
in job `jobid`, and boolean column `burst` indicating if the worker is currently experiencing a 
latency burst.
"""
function burst_df(df, jobid, worker_index; bin_size=10, min_increase=1.05, joinbins=false)
    df = filter(:jobid => (x)->x==jobid, df)
    name = "compute_latency_worker_$worker_index"
    samples = replace(df[:, name], missing=>0.0)
    rv = DataFrame(:latency => samples, :time => df.time)
    min_latency, max_latency = minimum(samples), maximum(samples)
    mean_latency = mean(samples)
    thr = find_corner(samples)
    rv.bin = ceil.(Int, rv.time ./ bin_size)
    dfi = combine(groupby(rv, :bin), :latency => mean => :latency, :time => mean => :time)
    if max_latency < mean_latency * min_increase
        dfi.burst = zeros(Bool, size(dfi, 1))
    else
        dfi.burst = dfi.latency .> thr
    end
    if joinbins
        select!(dfi, Not(:latency))
        select!(dfi, Not(:time))
        return leftjoin(rv, dfi, on=:bin, matchmissing=:error)
    else
        return dfi
    end
end

function fuckinga(df, jobid)
    df = filter(:jobid => (x)->x==jobid, df)
    sort!(df, :iteration)
    nworkers = df.nworkers[1]
    rv = zeros(size(df, 1))
    for i in 1:size(df, 1)
        vmin = Inf
        j = 0
        for k in 1:nworkers
            v = df[i, "latency_worker_$k"]
            if v < vmin
                vmin = v
                j = k
            end
        end
        rv[i] = df[i, "latency_worker_$j"] - df[i, "compute_latency_worker_$j"]
    end
    rv
end

"""

Plot the iteration latency of workers with indices in `worker_indices` of job `jobid`.
"""
function plot_timeseries(df, jobid, worker_indices; separate=true, cumulative=false, mark_bursts=false, time=false)
    !cumulative || !mark_bursts || throw(ArgumentError("can't plot bursts with cumulative=true"))
    println("Plotting per-worker iteration latency for workers: $worker_indices of jobid: $jobid")
    df = filter(:jobid => (x)->x==jobid, df)    

    println("nworkers: $(mean(df.nworkers)), nwait: $(mean(df.nwait))")
    # nslow: $(mean(df.nslow)), slowprob: $(mean(df.slowprob))

    sort!(df, :iteration)
    plt.figure()
    miny, maxy = Inf, -Inf
    overall = zeros(size(df, 1))
    for worker in worker_indices
        xs = time ? df.time : df.iteration
        if separate
            # compute
            ys = df[:, "compute_latency_worker_$worker"]
            ys = cumulative ? cumsum(ys) : ys
            if count(ismissing, ys) > 0
                continue
            end
            # ys = filter!(!ismissing, ys)
            vs = filter(!isnan, ys)
            if length(vs) > 0
                miny = min(miny, minimum(vs))
                maxy = max(maxy, maximum(vs))
            end
            if mark_bursts
                dfi = burst_df(df, jobid, worker)
                select!(dfi, Not(:latency))
                dfi = leftjoin(df, dfi, on=:bin, matchmissing=:error)
                mask = Bool.(dfi.burst)
                plt.plot(xs[mask], ys[mask], "k.")
                plt.plot(xs[.!mask], ys[.!mask], label="Worker $worker (comp.)")
            else
                plt.plot(xs, ys, label="Worker $worker (comp.)")                
            end
            # write_table(xs[4001:5000], ys[4001:5000], "timeseries_compute_$(jobid)_$(worker).csv", nsamples=1000)

            # communication
            ys = df[:, "latency_worker_$worker"] .- df[:, "compute_latency_worker_$worker"]
            ys = cumulative ? cumsum(ys) : ys
            vs = filter(!isnan, ys)
            if length(vs) > 0
                miny = min(miny, minimum(vs))
                maxy = max(maxy, maximum(vs))
            end
            plt.plot(xs, ys, label="Worker $worker (comm.)")
            # write_table(xs[1:100], ys[1:100], "timeseries_communication_$(jobid)_$(worker).csv")
        else
            ys = df[:, "latency_worker_$worker"]
            ys = cumulative ? cumsum(ys) : ys
            vs = filter(!isnan, ys)
            if length(vs) > 0
                miny = min(miny, minimum(vs))
                maxy = max(maxy, maximum(vs))
            end        
            # overall .= max.(overall, vs)                            
            plt.plot(xs, ys, label="Worker $worker (total)")
            # write_table(xs[1:150], ys[11:160], "timeseries_$(jobid)_$(worker).csv", nsamples=150)
        end
    end    

    # plt.plot(1:size(df, 1), overall, "k-")
    # write_table((1:size(df, 1))[1:150], overall[11:160], "timeseries_overall_$(jobid).csv", nsamples=150)

    # if enabled, mark iterations with load-balancing
    if "loadbalanced" in names(df)
        Is = findall(isone, df.loadbalanced)
        println("load-balanced in iterations $Is")
        for i in Is
            x = time ? df.time[i] : i
            plt.plot([x, x], [miny, maxy], "k-")
        end
    end    

    plt.grid()
    # plt.legend()
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

    # if enabled, mark iterations with load-balancing
    if "loadbalanced" in names(df)
        Is = findall(isone, df.loadbalanced)
        println("load-balanced in iterations $Is")        
        miny, maxy = minimum(ys), maximum(ys)
        for i in Is
            plt.plot([i, i], [miny, maxy], "k-")
        end
    end       

    plt.grid()
    plt.legend()
    plt.title("Job $jobid")
    plt.xlabel("Iteration")
    plt.ylabel("Total iteration latency [s]")
    plt.tight_layout()
    return
end

### Two-state Markov model

"""

Return the state transition matrix for job `jobid` and worker `worker_index`.
"""
function state_transition_matrix(df, jobid, worker_index)
    df = burst_df(df, jobid, worker_index)
    states = Bool.(df.burst)
    rv = zeros(2, 2)
    state = states[1]
    for i in 2:length(states)
        rv[Int(state) + 1, Int(states[i]) + 1] += 1
        state = df.burst[i]
    end
    @views s1 = sum(rv[1, :])
    if !iszero(s1)
        @views rv[1, :] ./= sum(rv[1, :])
    end
    @views s2 = sum(rv[2, :])
    if !iszero(s2)
        @views rv[2, :] ./= sum(rv[2, :])
    end
    rv
end

"""

Return the state transition matirx averaged over all workers.
"""
function state_transition_matrix(df, jobid)
    df = filter(:jobid => (x)->x==jobid, df)
    rv = zeros(2, 2)
    nworkers = maximum(df.nworkers)
    n = 0
    for i in 1:nworkers
        P = state_transition_matrix(df, jobid, i)
        if (P[2, 1] + P[2, 2]) != 0
            rv .+= P
            n += 1
        end
    end
    rv ./= n
end

function base_burst_mean_latency(df, jobid, worker_index)
    df = burst_df(df, jobid, worker_index)
    base_latency = 0.0
    burst_latency = 0.0
    for i in 1:size(df, 1)
        if df.burst[i]
            burst_latency += df.latency[i]
        else
            base_latency += df.latency[i]
        end
    end
    nbursts = sum(df.burst)
    base_latency /= size(df, 1) - nbursts
    burst_latency /= nbursts
    base_latency, burst_latency
end

function base_burst_mean_latency_increase(df, jobid)
    df = filter(:jobid => (x)->x==jobid, df)
    nworkers = maximum(df.nworkers)
    n = 0
    rv = 0.0
    for i in 1:nworkers
        base_latency, burst_latency = base_burst_mean_latency(df, jobid, i)
        if !isnan(burst_latency)
            @info "worker $i: $(burst_latency / base_latency)"
            rv += burst_latency / base_latency
            n += 1
        end
    end
    rv / n
end

### per-worker latency distribution

# function worker_latency_ratio(df; miniterations=1000)
#     maxworkers = maximum(df.nworkers)
#     latency_columns = ["compute_latency_worker_$i" for i in 1:maxworkers]
#     for dfi in groupby(df, :jobid)
#         if size(dfi, 1) < miniterations
#             continue
#         end
#         nworkers = dfi.nworkers[1]
#         jobid = dfi.jobid[1]
#         for i in 1:nworkers
#             ratio = maximum(dfi[:, latency_columns[i]]) / minimum(dfi[:, latency_columns[i]])
#             @info "jobid: $jobid, worker_index: $i, ratio: $ratio"
#         end
#         return
#     end
#     return
# end

"""

For each worker in job with ID `jobid`, compute the mean and variance of the comp. and comm. latency.
"""
function compute_worker_stats(df, jobid)
    df = filter(:jobid => (x)->x==jobid, df)
    if size(df, 1) == 0
        error("no job with ID $jobid")
    end
    sort!(df, :iteration)
    nworkers = df.nworkers[1]
    nsubpartitions = df.nsubpartitions[1]    
    comp_ms = zeros(nworkers)
    comp_vs = zeros(nworkers)
    comm_ms = zeros(nworkers)
    comm_vs = zeros(nworkers)
    if "loadbalanced" in names(df)
        i = findfirst(!iszero, df.loadbalanced)
    else
        i = size(df, 1)
    end
    iu = isnothing(i) ? size(df, 1) : i
    il = 1

    # il, iu = 1, 42
    il, iu = 1, 100

    @info "il: $il, iu: $iu"

    latency_columns = ["latency_worker_$i" for i in 1:nworkers]
    compute_latency_columns = ["compute_latency_worker_$i" for i in 1:nworkers]
    for j in 1:nworkers
        comp_ms[j], comp_vs[j] = mean_and_var(skipmissing(df[il:iu, compute_latency_columns[j]]))
        comp_ms[j] /= (1/nworkers * 1/nsubpartitions)
        comp_vs[j] /= (1/nworkers * 1/nsubpartitions)^2
        comm_ms[j], comm_vs[j] = mean_and_var(skipmissing(df[il:iu, latency_columns[j]] .- df[il:iu, compute_latency_columns[j]]))
    end

    ms = comp_ms .* (1/nworkers * 1/nsubpartitions) .+ comm_ms
    ratio = maximum(ms) / minimum(ms)
    println("Mean latency ratio: $ratio")

    # println("comp_ms = $comp_ms")
    # println("comp_vs = $comp_vs")
    # println("comm_ms = $comm_ms")
    # println("comm_vs = $comm_vs")
    
    return
end

"""

Plot the latency distribution of individual workers.
"""
function plot_worker_latency_distribution(df, jobid, worker_indices=[1, 2]; dist=Gamma, prune=0.05)
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
        samples = float.(df[:, latency_columns[i]])
        xs = sort(samples)
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

        # only select samples outside of bursts
        @info "1"
        dfi = burst_df(df, jobid, i, joinbins=true)
        @info "2"
        samples = df[:, compute_latency_columns[i]]
        mask = Bool.(dfi.burst)
        samples = float.(samples[mask])

        xs = sort(samples)
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

"""

Compute the mean and variance of the latency for each worker.
"""
function per_worker_statistics(df, jobid)
    df = filter(:jobid => (x)->x==jobid, df)
    if size(df, 1) == 0
        println("no job with ID $jobid")
        return
    end
    nworkers = df.nworkers[1]
    nsubpartitions = df.nsubpartitions[1]
    println("nworkers: $nworkers, nsubpartitions: $nsubpartitions")

    latency_columns = ["latency_worker_$i" for i in 1:nworkers]
    compute_latency_columns = ["compute_latency_worker_$i" for i in 1:nworkers]    

    comp_ms = [mean(df[:, col]) for col in compute_latency_columns]
    comp_vs = [var(df[:, col]) for col in compute_latency_columns]

    comp_mcs = [mean(df[:, col] ./ ((1 / nworkers) ./ df.nsubpartitions)) for col in compute_latency_columns]
    comp_vcs = [var(df[:, col]  ./ sqrt.((1 / nworkers) ./ df.nsubpartitions)) for col in compute_latency_columns]    

    println("comp_mcs = $comp_mcs")
    println("comp_vcs = $comp_vcs")

    comm_ms = [mean(df[:, latency_columns[i]] .- df[:, compute_latency_columns[i]]) for i in 1:nworkers]
    comm_vs = [var(df[:, latency_columns[i]] .- df[:, compute_latency_columns[i]]) for i in 1:nworkers]

    println("comm_ms = $comm_ms")
    println("comm_vs = $comm_vs")

    return
end

### distribution of the mean and variance of the per-worker latency

"""

pca-1000genomes-dense-equiv-c5.xlarge-eu-north-1.csv
[30048]
[0.0005688430066438187]
[2.975438171605625e-8]

latency-c5.xlarge-eu-north-1.csv
[43508208, 60096]
[4.178164252137641, 0.00040377245389770224]
[0.3648310268706024, 2.6410875254830194e-8]

combined
[43508208, 60096, 30048]
[4.178164252137641, 0.00040377245389770224, 0.0005688430066438187]
[0.3648310268706024, 2.6410875254830194e-8, 2.975438171605625e-8]

Plot the mean and variance of the computation latency vs. worker_flops.
"""
function per_worker_mean_var_scatter(dfg, minsamples=100)
    dfi = combine(groupby(dfg, :nbytes), :comm_mean => mean => :mean, :comm_var => mean => :var, :comm_mean => length => :nsamples)
    dfi = filter(:nsamples => (x)->minsamples<=x, dfi)
    # row = Dict(:nbytes => 0.0, :comm_mean => 0.0, :comm_var => 0.0, :nsamples => 1)
    # push!(dfi, row)
    println(dfi.nbytes)
    println(dfi.mean)
    println(dfi.var)

    nbytes = [43508208, 60096, 30048]
    means = [4.178164252137641, 0.00040377245389770224, 0.0005688430066438187]
    vars = [0.3648310268706024, 2.6410875254830194e-8, 2.975438171605625e-8]    

    # mean
    plt.figure()
    plt.plot(dfg.nbytes, dfg.comm_mean, ".")
    plt.plot(dfi.nbytes, dfi.mean, "o")
    # p = Polynomials.fit(dfi.nbytes, dfi.mean, 2)
    # @info "p: $p"
    xs = 10.0.^range(log10(dfi.nbytes[1]), log10(maximum(dfi.nbytes)), length=100)
    # ys = p.(xs)    
    # plt.plot(xs, ys, "k-")

    # fitted line through the origin
    slope = mean(dfi.mean ./ dfi.nbytes)
    @info "slope: $slope"
    ys = xs .* slope
    plt.plot(xs, ys, "k--")    

    # temp.
    plt.plot(nbytes, means, "s")

    xs = 10.0.^range(log10(minimum(nbytes)), log10(maximum(nbytes)), length=100)
    slope = mean(means[1:2] ./ nbytes[1:2])
    ys = xs .* slope
    plt.plot(xs, ys, "m--")    


    plt.xscale("log")
    plt.yscale("log")
    plt.grid()

    write_table(dfg.nbytes, dfg.comm_mean, "comm_mean.csv")
    write_table(dfi.nbytes, dfi.mean, "comm_mean_mean.csv")
    write_table(xs, ys, "comm_mean_line.csv")

    # variance
    plt.figure()
    plt.plot(dfg.nbytes, dfg.comm_var, ".")
    plt.plot(dfi.nbytes, dfi.var, "o")

    # degree-2 polynomial
    p = Polynomials.fit(dfg.nbytes, dfg.var, 2)
    # @info "p: $p"
    ys = p.(xs)
    plt.plot(xs, ys, "k-")

    # # fitted line through the origin    
    # slope = mean(dfi.var ./ dfi.nbytes)
    # @info "slope: $slope"
    # ys = xs .* slope
    # plt.plot(xs, ys, "k--")

    plt.xscale("log")
    plt.yscale("log")    

    write_table(dfg.nbytes, dfg.comm_var, "comm_var.csv")
    write_table(dfi.nbytes, dfi.var, "comm_var_mean.csv")
    write_table(xs, ys, "comm_var_line.csv")    

    plt.grid()
end

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

# I want to bin rows by log of time
# Let's say power of 10 for now
# Let's start by creating a df with jobid, worker index and latency for each worker
# Let's bin latency and compute the mean in each bin
# Next, for each bin, compute the ratio for each subsequent bin
# Code works (I think)
# Next, let's make a plot!
# Let's make things simpler and choose a particular workload

"""

Takes as its argument a string of the form `"latency_worker_i"` or `"compute_latency_worker_i`, and returns `i` as an integer.
"""
function index_from_latency_name(s::AbstractString)
    i = findlast((x)->x=='_', s)
    parse(Int, view(s, i+1:length(s)))
end


"""

Return a df with columns worker_index, jobid, worker_latency, worker_compute_latency, worker_flops and nbytes, 
where each row corresponds to a particular worker and job.
"""
function tall_from_wide(df)
    df = filter([:nwait, :nworkers] => (x, y)->x==y, df)
    maxworkers = maximum(df.nworkers)
    latency_columns = ["latency_worker_$i" for i in 1:maxworkers]
    compute_latency_columns = ["compute_latency_worker_$i" for i in 1:maxworkers]    
    
    # stack by latency
    @time dftot = stack(df, latency_columns, [:jobid, :nbytes, :worker_flops, :iteration, :time], variable_name=:name, value_name=:latency)
    # @time dftot.worker_index = [parse(Int, split(name, "_")[end]) for name in dftot.name]
    @time dftot.worker_index = index_from_latency_name.(dftot.name)
    @time select!(dftot, Not(:name))

    # stack by compute latency
    @time dfcomp = stack(df, compute_latency_columns, [:jobid], variable_name=:name, value_name=:comp_latency)
    # @time dfcomm.worker_index = [parse(Int, split(name, "_")[end]) for name in dfcomm.name]
    # @time dfcomp.worker_index = index_from_latency_name.(dfcomp.name)
    # @time select!(dfcomp, Not(:name))

    # join
    # @time rv = innerjoin(dftot, dfcomm, on=[:jobid, :worker_index])
    @time dftot.comp_latency = dfcomp.comp_latency

    # add communication latency
    @time dftot.comm_latency = dftot.latency .- dftot.comp_latency
    dftot
end

"""

Plot the 
"""
function plot_change_rate(dft, nlevels=20, maxchange=2, col=:latency)
    dft = copy(dft)
    dft.bin = max.(1, ceil.(Int, log10.(dft.time)))
    rv = zeros(nlevels, maximum(dft.bin))
    for dfi in groupby(dft, [:jobid, :worker_index])
        dfj = combine(groupby(dfi, :bin), col => mean => :latency)
        i = findfirst(isone, dfj.bin)
        if ismissing(dfj.latency[i])
            continue
        end
        for j in 1:size(dfj, 1)
            if ismissing(dfj.latency[j])
                continue
            end
            v = dfj.latency[j] / dfj.latency[i]            
            x = dfj.bin[j]
            y = round(Int, v / maxchange * nlevels)
            y = min(nlevels, y)
            y = max(1, y)
            rv[y, x] += 1
        end
    end    
    rv
end

# 0.9 - 1.1 => 5
# 1.1 - 1.2 => 6
# 0.8-0.9 => 4

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

# Results aren't looking super fun
# Clearly, reducing nwait gives an advantage
# Let's add SAG back in
# It doesn't seem like the load-balancer has time to make much of an improvement in these short-running jobs
# The situation might be better for log. reg.
# It's clear though that reducing nwait and using async results bring clear benefits
# That's enough for us
# I got what we need
# Let's put in a few more experiments

# I see now why the one with LB is faster already in the first iteration
# It's because I set it equal to the mean over the remaining iterations
# (to remove initialization latency)
# I'd need to replace it with the mean over the iterations up until the first LB
# I've fixed it

# First, let's consider DSAG with w=9 under perfect conditions
# That's the baseline for all scenarios
# Next, let's consider the 4 different scenarios separately
# For each scenario, plot the baseline together with DSAG for different parameters

# nslow=0, slowprob=0
# (loss 4e-3)
# w=6, no LB
# 164.468

# w=9, LB
# 167.225 

# w=6, LB
# 171.996

# w=9, no LB
# 172.657 

# DSAG nwait=6, no LB is 5% faster than SAG
# DSAG nwait=6, LB is about the same as SAG
# DSAG nwait=9, LB is in between

# nslow=3, slowprob=0
# (loss 4e-3)
# DSAG nwait=6, no LB is best, and is more than 2x faster than SAG and about the same speed as SAG under perfect conditions
# DSAG nwait=6 and nwait=9, LB are similar, but nwait=6 is slightly faster

# nslow=0, slowprob=0.1
# (loss 4e-3)
# DSAG nwait=9 with and without LB are the same
# DSAG nwait=6, LB is almost twice as fast, but not quite as fast as SAG under ideal conditions
# I'm missing data for DSAG=6, no LB

# nslow=3, slowprob=0.1
# DSAG nwait=9 with LB is 20-25% faster than without LB
# I'm missing data for nwait=6

# The conclusion is that reducing nwait gets you pretty much ideal performance
# (the same as SAG under ideal conditions)
# Using the LB actually reduces performance, since it slows down the fastest workers
# That's an inherent feature of the LB
# Although I could, of course, change the LB so that it only speeds up slow workers and never slows down fast workers
# I need to think a bit about what do to now

# GD, stepsize 128
# Is = [0, 1, 5, 10, 20, 40, 50, 75, 100]
# fs = [0.6931471805599444, 0.5757509367339131, 0.39244359279752594, 0.30707040115301165, 0.23614727589228687, 0.18346042017331302, 0.17000816720049297, 0.1491794304064495, 0.1368085331380532]
# To reach 1e-4: > 1e6 iterations

# GD, stepsize 256
# Is = [0, 1, 5, 10, 15, 20]
# fs = [0.6931471805599444, 0.4901311532575216, 0.30067788002418033, 0.2330460895591535, 0.20131232433266838, 0.1821599500258606]
# To reach 1e-4: > 1e6 iterations

# GD, stepsize 1000
# Is = [0, 1, 5, 10, 15, 20]
# fs = [0.6931471805599444, 0.2700252259844102, 0.19942048473537807, 0.14104499661819198, 0.12726369493098577, 0.11864164451048687]
# To reach 1e-4: > 1e6 iterations

# SAG, stepsize 1000, nwait 48, nsubpartitions 480
Is = [0, 10, 50, 100, 150, 200]
fs = [0.6931471805599444, 0.14787831085616412, 0.09752716859470263, 0.1382957363421898, 0.17973698135153582, 0.10014377415557275]

Is = [0, 10, 50, 100, 200, 400, 500, 750, 1000]
fs = [0.6931471805599444, 0.14787831085616412, 0.09752716859470263, 0.1382957363421898, 0.10014377415557275, 0.06931350121788657, 0.06915655450649712, 0.08117820654429656, 0.06649721139774649]

# GD, stepsize 10000
# Is = [0, 1, 5, 10, 15, 20]
# fs = [0.6931471805599444, 0.527436814382326, 5.63786891630697, 0.5669757343704201, 0.3911818140869798, 0.28593461743066795]
# To reach 1e-4: ≈ 1e4-2e4

# SAG, stepsize 10000, nwait 48, nsubpartitions 480
Is = [0, 10, 50, 100, 150, 200]
fs = [0.6931471805599444, 0.6458971233161701, 0.13879046284147123, 0.07283336094723879, 0.08034888335618584, 0.1068566398225466]

Is = [0, 10, 50, 100, 200, 400, 500, 750, 1000]
fs = [0.6931471805599444, 0.6458971233161701, 0.13879046284147123, 0.07283336094723879, 0.1068566398225466, 0.2371697935004405, 0.44595219902308675, 0.12881074897148503, 0.14836394614832593]

# GD, stepsize 100000
# Is = [0, 1, 5, 10, 15, 20]
# fs = [0.6931471805599444, Inf, Inf, 3.25800791386318, 5.1815523657650004, Inf]

# ex3
# SAG, stepsize 1000
Is = [1, 2, 3, 6, 11, 21, 38, 70, 127, 234, 428, 785, 1438, 2637, 4833, 8859, 16238, 29764, 54556, 100000]
fs = [0.26834265364438425, 0.20494044286187493, 0.18453340895954867, 0.15988284517293044, 0.14349297868733563, 0.11821682341158148, 0.10448101312138802, 0.08740593815032184, 0.20757379542066404, 0.08214655195785492, 0.06639599388187367, 0.060916167226869176, 0.0569138145867912, 0.05318697826946615, 0.04971212562850331, 0.046522679820757494, 0.04365483222663333, 0.041140123472714875, 0.039003760637618436, 0.037264706092371855]

# SAG, stepsize 10000
Is = [1, 2, 3, 6, 11, 21, 38, 70, 127, 234, 428, 785, 1438, 2637, 4833, 8859, 16238, 29764, 54556, 100000]
fs = [0.5103203261292784, 0.6321112353472755, 0.6961155657885862, 0.7331210561255893, 0.6708938131228802, 0.44520704891691165, 0.3609102302713835, 0.9531964842847881, 1.8108350458978526, 0.8161233711215429, 0.41436585542672416, 0.7572459212147262, 0.4289757693465181, 0.2561249222133476, 0.15757529057239483, 0.3881388761550672, 0.13902790295164846, 0.32675605385829254, 0.7129304091954426, 0.135995569313817]

# GD, stepsize 1000
Is = [1, 2, 3, 4, 7, 11, 18, 30, 48, 78, 127, 207, 336, 546, 886, 1438, 2336, 3793, 6158, 10000]
fs = [0.2700252172197113, 0.23782497262987526, 0.2248956483906497, 0.21636728546543718, 0.16223770018082442, 0.13697029505242303, 0.12086405537958705, 0.10675533663753153, 0.09599213762757994, 0.0868644357618402, 0.07943239547397388, 0.07337431752420415, 0.06838392380453197, 0.06411012376600528, 0.06037688133657744, 0.05702530071077946, 0.05394218440137071, 0.051072875818447556, 0.048396595452253194, 0.04591793314540445]

# GD, stepsize 10000
Is = [1, 2, 3, 4, 7, 11, 18, 30, 48, 78, 127, 207, 336, 546, 886, 1438, 2336, 3793, 6158, 10000]
fs = [0.5274367019280838, 0.4408785035709458, 0.5596329650465613, 1.1475935182656603, 3.656783132128757, 0.5351843644819414, 0.37338246539680914, 0.2483774893875378, 0.1679318649604483, 0.11190084432865868, 0.07509226910333948, 0.135896635802303, 0.0675397604450637, 0.0727854747272732, 0.06000141017603729, 0.05712261035824127, 0.052868390352489004, 0.04922075927152398, 0.0497684082828835, 0.04278610479777781]


# Jobs for which the LB malfunctioned
# (df4)
# nwait=10: 65
# nwait=40: 23, 55, 87
# nwait=49: 45, 61

# (df5)
# nwait=10: 49, 81
# nwait=40: 103, 151
# nwait=49: 13, 29

function plot_convergence_by_job(df; problem="pca")
    
    if problem == "pca"        
        opt = 31.024077080126776
    elseif problem == "logreg"
        opt = 0.6375689567978461
    else
        throw(ArgumentError("unknown problem $problem"))
    end
    
    df = filter(:mse => (x)->!ismissing(x), df)

    plt.figure()
    for dfi in groupby(df, :jobid)
        xs = dfi.time
        if problem == "pca"
            ys = opt.-dfi.mse
        elseif problem == "logreg"
            ys = dfi.mse .- opt
        end        
        plt.plot(xs, ys, "o-", label="$(dfi.jobid[1])")
    end

    plt.xscale("log")
    plt.yscale("log")
    plt.grid(true)
    plt.legend()    
    plt.xlabel("Time [s]")
    plt.ylabel("Sub-optimality Gap")
    return    
end

"""

Plot the rate of convergence over time for DSAG, SAG, SGD, and coded computing. Let 
`latency=empirical` to plot against empirical latency, or let `latency=c5xlarge` to plot against 
latency computed by the model, fitted to traces recorded on `c5xlarge` instances.

rcv1full optimal value: 0.08294410910152755
1000genomes optimal value from orig. paper: 15.512054259119793
1000genomes optimal value, new value: 31.024077073103722
(new results have doubled this for some reason)
opt=maximum(skipmissing(df.mse))

covtype (incl. reg.) opt: 0.2707541904571277
higgs (incl. reg.) opt: 0.6375689567978464

"""
function plot_convergence(df, nworkers; latency="empirical", niidm=nothing, problem="pca")

    if problem == "pca"
        # opt = 31.024077073103722
        opt = 31.024077080126776 # discovered a better solution
    elseif problem == "logreg"
        # opt = 0.6375689567978464
        opt = 0.6375689567978461
    else
        throw(ArgumentError("unknown problem $problem"))
    end

    df = filter(:nworkers => (x)->x==nworkers, df)
    df = filter(:nreplicas => (x)->x==1, df)
    df_coded = df # used for the coded computing bound later
    df = filter(:mse => (x)->!ismissing(x), df)    
    # sort!(df, [:jobid, :iteration])
    sort!(df, :iteration)
    println("nworkers: $nworkers, opt: $opt")

    # parameters are recorded as a tuple (nwait, nsubpartitions, stepsize)
    fields = [:loadbalance, :nwait, :nsubpartitions]

    plt.figure()
    for dfi in groupby(df, fields)
        if size(dfi, 1) == 0
            continue
        end
        label = join(["$(field)_$(dfi[1, field])" for field in fields], "-")
        println("label: $label")

        # use dashed lines for load-balanced schemes
        if dfi.loadbalance[1]
            line = "--"
        else
            line = "-"
        end        

        # compute latency jointly for DSAG and SAG
        df_vr = filter(:variancereduced => (x)->x, dfi)
        xs = combine(groupby(df_vr, :iteration), :time => mean => :time).time

        ### DSAG
        dfj = dfi
        dfj = filter(:variancereduced => (x)->x, dfj)
        dfj = filter(:nostale => (x)->!x, dfj)
        println("DSAG: $(length(unique(dfj.jobid))) jobs")
        println(unique(dfj.jobid))
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
            # xs = dfj.time
            if problem == "pca"
                ys = opt.-dfj.mse
            elseif problem == "logreg"
                ys = dfj.mse .- opt
            end
            # ys = dfj.mse .- opt

            # println("xs: $xs")
            # println("ys: $ys")



            plt.semilogy(xs, ys, "s"*line, label="DSAG $label")
            # filename = "dsag_$(nworkers)_$(nwait)_$(nsubpartitions)_$(stepsize).csv"       
            filename = "dsag_$(label).csv"
            write_table(xs, ys, filename)
        end
        println()

        ### SAG
        dfj = dfi
        dfj = filter(:variancereduced => (x)->x, dfj)
        dfj = filter(:nostale => (x)->x, dfj)        
        println("SAG: $(length(unique(dfj.jobid))) jobs")
        println(unique(dfj.jobid))
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
            # xs = dfj.time
            if problem == "pca"
                ys = opt.-dfj.mse
            elseif problem == "logreg"
                ys = dfj.mse .- opt
            end            
            plt.semilogy(xs, ys, "o"*line, label="SAG $label")
            filename = "sag_$(label).csv"
            write_table(xs, ys, filename)
        end
        println()

        ### SGD
        dfj = dfi
        dfj = filter(:variancereduced => (x)->!x, dfj)
        println("SGD: $(length(unique(dfj.jobid))) jobs")
        println(unique(dfj.jobid))
        if size(dfj, 1) > 0
            dfj = combine(groupby(dfj, :iteration), :mse => mean => :mse, :time => mean => :time)
            if latency == "empirical"
                println("Plotting SGD with empirical latency")
            else
                # TODO: not checked
                # df_comm, df_comp = niidm
                # dfs = simulate_iterations(nbytes, nflops/upscale; niterations=maximum(dfj.iteration), nworkers=nworkers*upscale, nwait=nwait*upscale, df_comm, df_comp, update_latency=0.0022031946363636366)
                # ys = opt .- dfj.mse
                # dfj.time .= dfs.time[dfj.iteration]
                # println("Plotting DSAG with model latency for $latency")
            end
            xs = dfj.time
            if problem == "pca"
                ys = opt.-dfj.mse
            elseif problem == "logreg"
                ys = dfj.mse .- opt
            end                        
            plt.semilogy(xs, ys, "^"*line, label="SGD $label")
            filename = "sgd_$(label).csv"
            write_table(xs, ys, filename)
        end
        println()        
    end

    # Plot GD
    stepsize = 1.0
    dfi = df
    dfi = filter(:nwait => (x)->x==nworkers, dfi)
    dfi = filter(:nsubpartitions => (x)->x==1, dfi)
    dfi = filter(:variancereduced => (x)->x==false, dfi)
    dfi = filter(:stepsize => (x)->isapprox(x, stepsize), dfi)

    # dfi = dfi[dfi.nwait .== nworkers, :]
    # dfi = dfi[dfi.nsubpartitions .== 1, :]
    # dfi = dfi[dfi.variancereduced .== false, :]
    # dfi = dfi[dfi.stepsize .== stepsize, :]

    println("GD $(length(unique(dfi.jobid))) jobs")
    println(unique(dfi.jobid))
    dfj = combine(groupby(dfi, :iteration), :mse => mean => :mse, :time => mean => :time)
    # dfj = by(dfi, :iteration, :mse => mean => :mse, :time => mean => :time)
    if latency == "empirical"
        println("Plotting GD with empirical latency")
    else
        # dfj.time .= predict_latency(nworkers, mean(dfi.worker_flops), nworkers; type=latency) .* dfj.iteration

        df_comm, df_comp = niidm
        nflops = mean(dfi.worker_flops)        
        dfs = simulate_iterations(nbytes, nflops/upscale; niterations=maximum(dfj.iteration), nworkers=nworkers*upscale, nwait=nworkers*upscale, df_comm, df_comp, update_latency=0.0022031946363636366)
        ys = opt .- dfj.mse
        dfj.time .= dfs.time[dfj.iteration]        

        println("Plotting GD with model latency for $latency")
    end    
    if size(dfj, 1) > 0
        xs = dfj.time
        if problem == "pca"
            ys = opt .- dfj.mse
        elseif problem == "logreg"
            ys = dfj.mse .- opt
        end
        plt.semilogy(xs, ys, "ms-", label="GD")
        filename = "gd_$(nworkers)_$(stepsize).csv"
        write_table(xs, ys, filename)
    end

    # plot coded computing bound
    # (based on dfi, which contains data for GD)
    dfi = df_coded
    dfi = filter(:nwait => (x)->x==nworkers, dfi)
    dfi = filter(:nsubpartitions => (x)->x==1, dfi)
    dfi = filter(:variancereduced => (x)->x==false, dfi)
    dfi = filter(:stepsize => (x)->isapprox(x, stepsize), dfi)
    println("Coded computing $(length(unique(dfi.jobid))) jobs")
    code_nwait = nworkers - 10
    if size(dfi, 1) > 0
        dfi.nwait .= code_nwait
        scale_compute_latency!(dfi, nworkers / code_nwait) # increase comp. latency due to coding overhead
        dfi = filter(:mse => !ismissing, dfi)
        # return dfi
        dfj = combine(groupby(dfi, :iteration), :mse => mean => :mse, :time => mean => :time)
        xs = dfj.time
        if problem == "pca"
            ys = opt .- dfj.mse
        elseif problem == "logreg"
            ys = dfj.mse .- opt
        end
        if size(dfj, 1) > 0
            plt.semilogy(xs, ys, "k--", label="MDS (w: $code_nwait)")
            filename = "bound_$(nworkers)_$(code_nwait)_$(stepsize).csv"
            write_table(xs, ys, filename)
        end
    end

    # get the GD df
    # re-index the workers by latency
    # multiply the 


    # # plot coded computing bound (old)
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