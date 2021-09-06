using StatsBase, Statistics, OnlineStats, Dates

function latency_profiler(chin::Channel; chout::Channel, nworkers::Integer, qlower::Real=0.1, qupper::Real=0.9, buffersize::Integer=1000, windowsize::Dates.AbstractTime=Second(10))
    @info "latency_profiler task started"    
    
    # maintain a window of latency samples for each worker
    ws = [MovingTimeWindow(windowsize, valtype=@NamedTuple{comp::Float64, comm::Float64}, timetype=Time) for _ in 1:nworkers]

    # utility function for updating the correct window with a new latency measurement
    function fitwindow(v)
        0 < v.worker <= nworkers || @error "Expected v.worker to be in [1, $nworkers], but it is $(v.worker)"
        isnan(v.comm) || isnan(v.comp) || fit!(ws[v.worker], (v.timestamp, @NamedTuple{comp::Float64,comm::Float64}((v.comp, v.comm))))
        return
    end

    # helper functions for computing the mean and variance over the samples in a window
    buffer = zeros(buffersize)
    function processwindow(w::MovingTimeWindow, key::Symbol)
        key == :comp || key == :comm || throw(ArgumentError("key must be either :comp or :comm, but is $key"))        

        # populate the buffer
        i = 0
        n = 0
        for (_, t) in value(w)
            v = t[key]
            if !isnan(v)
                buffer[i+1] = v
                i = mod(i + 1, buffersize)
                n += 1
            end
        end
        n = min(n, buffersize)

        # return NaNs if there are no values
        if n == 0
            return NaN, NaN
        end

        # compute quantile indices
        sort!(view(buffer, 1:n))
        il = max(1, round(Int, n*qlower))
        iu = min(buffersize, round(Int, n*qupper))

        # compute mean and variance over the values between qlower and qupper
        vs = view(buffer, il:iu)
        m = mean(vs)
        v = var(vs, mean=m, corrected=true)        
        m, v
    end

    # process incoming latency samples
    while isopen(chin)

        # consume all values currently in the channel
        try
            vin = take!(chin)
            fitwindow(vin)
        catch e
            if e isa InvalidStateException
                break
            else
                rethrow()
            end
        end            
        while isready(chin)
            try
                vin = take!(chin)
                fitwindow(vin)
            catch e
                if e isa InvalidStateException
                    break
                else
                    rethrow()
                end
            end
        end        

        for i in 1:nworkers

            # for each worker, compute the mean and variance of the values in the window            
            comp_mean, comp_var = processwindow(ws[i], :comp)
            comm_mean, comm_var = processwindow(ws[i], :comm)
            if isnan(comp_mean) || isnan(comp_var) || isnan(comm_mean) || isnan(comm_var)
                continue
            end

            # push the computed statistics into the output channel
            vout = @NamedTuple{worker::Int,comp_mean::Float64,comp_var::Float64,comm_mean::Float64,comm_var::Float64}((i,comp_mean,comp_var,comm_mean,comm_var))            
            try
                push!(chout, vout)
            catch e
                if e isa InvalidStateException
                    break
                else
                    rethrow()
                end
            end            
        end
    end
    @info "latency_profiler task finished"
end

function process_stats(chin::Channel)
    @info "process_stats task started"
    for v in chin        
        @info v
    end
    @info "process_stats task finished"
end

function get_test_values(df)
    df = filter(:jobid => (x)->x==1, df)
    df.time, df.compute_latency_worker_4
end

function main(df, jobid=17, worker=4, nworkers=10)
    
    df = filter(:jobid => (x)->x==jobid, df)
    sort!(df, :iteration)
    ts = df.time        
    comps = df[:, "compute_latency_worker_$worker"]
    comms = df[:, "latency_worker_$worker"] .- comps

    nwait = 1
    ps = fill(240, nworkers)
    θs = fill(1/nworkers, nworkers)
    min_processed_fraction = sum(θs ./ ps) * nwait / nworkers

    # channel to push recorded latency into
    latency_channel = Channel{@NamedTuple{worker::Int,timestamp::Time,comp::Float64,comm::Float64}}(Inf)
    stats_channel = Channel{@NamedTuple{worker::Int,comp_mean::Float64,comp_var::Float64,comm_mean::Float64,comm_var::Float64}}(Inf)
    partitions_channel = Channel{@NamedTuple{worker::Int,p::Int}}(Inf)
    
    t1 = Threads.@spawn latency_profiler(latency_channel; chout=stats_channel, nworkers)
    t2 = Threads.@spawn load_balancer(stats_channel; chout=partitions_channel, ps0=ps, θs, min_processed_fraction, nwait)
    # t2 = Threads.@spawn process_stats(stats_channel)

    # while !istaskstarted(t1) || !istaskstarted(t2)
    #    sleep(1e-3) 
    # end

    timestamps = Nanosecond.(round.(Int, ts .* 1e9)) .+ Time(0)
    i = 0
    for i in 1:size(df, 1)
        for j in 1:nworkers
            comp = df[i, "compute_latency_worker_$j"]
            comm = df[i, "latency_worker_$j"] .- comp
            if !isnan(comp) && !isnan(comm)
                v = @NamedTuple{worker::Int,timestamp::Time,comp::Float64,comm::Float64}((j, timestamps[i], comp, comm))
                push!(latency_channel, v)            
            end
        end

        while isready(partitions_channel)
            try
                vin = take!(partitions_channel)
                @info "ps[$(vin.worker)] = $(vin.p)"
                ps[vin.worker] = vin.p
            catch e
                if e isa InvalidStateException
                    break
                else
                    rethrow()
                end
            end
        end

        sleep(1e-6)

        # if i > 100
        #     break
        # end        
    end
    
    println("coord finished")

    # wait for all latency values to be processed
    while !istaskfailed(t1) && !istaskfailed(t2) && isready(latency_channel)
        sleep(0.1)
    end

    # close the channels and wait fo the tasks to finish
    close(latency_channel)
    close(stats_channel)
    wait(t1)
    wait(t2)

    return
end