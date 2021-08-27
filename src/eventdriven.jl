export simulate_iterations

"""

Sample the total latency of a worker.
"""
function sample_worker_latency(d_comm, d_comp)
    !isnothing(d_comm) || !isnothing(d_comp) || error("either d_comm or d_comp must be provided")
    rv = 0.0
    if !isnothing(d_comm)
        rv += rand(d_comm)
    end
    if !isnothing(d_comp)
        rv += rand(d_comp)
    end
    rv
end

"""

Simulate `niterations` iterations of the computation.
"""
function simulate_iterations(;nwait, niterations=100, ds_comm, ds_comp, update_latency=0.5e-3)
    length(ds_comm) == length(ds_comp) || throw(DimensionMismatch("ds_comm has dimension $(length(ds_comm)), but ds_comp has dimension $(length(ds_comp))"))
    nworkers = length(ds_comm)
    0 < nwait <= nworkers || throw(DomainError(nwait, "nwait must be in [1, nworkers]"))
    sepochs = zeros(Int, nworkers) # epoch at which an iterate was last sent to each worker
    repochs = zeros(Int, nworkers) # epoch that each received gradient corresponds to
    stimes = zeros(nworkers) # time at which each worker was most recently assigned a task
    rtimes = zeros(nworkers) # time at which each worker most recently completed a task
    pq = PriorityQueue{Int,Float64}() # queue for event-driven simulation
    time = 0.0 # simulation time
    times = zeros(niterations) # time at which each iteration finished
    nfreshs = zeros(Int, niterations)
    nstales = zeros(Int, niterations)
    latencies = zeros(niterations)
    idle_times = zeros(niterations)
    fresh_times = zeros(niterations)
    stale_times = zeros(niterations)
    participation_prob = zeros(nworkers) # fraction of iterations the worker partitipates in
    for k in 1:niterations
        nfresh, nstale = 0, 0
        idle_time = 0.0 # total time workers spend being idle
        fresh_time = 0.0 # total time workers spend working on fresh gradients
        stale_time = 0.0 # total time workers spend working on stale gradients

        # enqueue all idle workers
        # (workers we received a fresh result from in the previous iteration are idle)
        t0 = time + update_latency # start time of this iteration
        for i in 1:nworkers
            if repochs[i] == k-1
                enqueue!(pq, i, t0 + sample_worker_latency(ds_comm[i], ds_comp[i]))
                sepochs[i] = k
                stimes[i] = t0
            end
        end

        # wait for nwait fresh workers
        while nfresh < nwait
            i, time = dequeue_pair!(pq)
            repochs[i] = sepochs[i]
            rtimes[i] = time
            if k > 1 && time < t0
                idle_time += t0 - time
                time = t0
            end
            if repochs[i] == k
                nfresh += 1
                participation_prob[i] += 1                
                fresh_time += rtimes[i] - stimes[i]
            else
                # put stale workers back in the queue
                nstale += 1
                stale_time += rtimes[i] - stimes[i]
                enqueue!(pq, i, time + sample_worker_latency(ds_comm[i], ds_comp[i]))
                sepochs[i] = k
                stimes[i] = time
            end
        end

        # tally up for how long each worker has been idle
        # (only workers we received a fresh result from are idle)
        for i in 1:nworkers
            if repochs[i] == k
                idle_time += time - rtimes[i] + update_latency
            end
        end

        # record
        nfreshs[k] = nfresh
        nstales[k] = nstale
        times[k] = time + update_latency
        latencies[k] = time - t0
        idle_times[k] = idle_time
        fresh_times[k] = fresh_time
        stale_times[k] = stale_time
    end
    participation_prob ./= niterations
    
    rv = DataFrame()
    rv.time = times
    rv[!, :update_latency] .= update_latency
    rv.latency = latencies
    rv.iteration = 1:niterations
    rv.idle_time = idle_times
    rv.fresh_time = fresh_times
    rv.stale_time = stale_times
    rv[!, :nworkers] .= nworkers
    rv[!, :nwait] .= nwait
    rv, participation_prob
end