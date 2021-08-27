
"""

Return the expected fraction of the dataset processed per iteration. Here, `θs` is the fraction of 
the dataset stored, `ps` is the number of partitions, and `Ls` is the probability of finishing 
among the `nwait` fastest workers, for each worker.
"""
function fraction_processed(ps::AbstractVector, θs::AbstractVector, Ls::AbstractVector)
    rv = 0.0
    for i in 1:length(Ls)
        rv += Ls[i] * θs[i] / ps[i]
    end
    rv
end

"""Helper function for running event-drive simulations"""
function simulate(;θs, ps, cms_comp, cvs_comp, ds_comm, nwait, niterations=100, nsamples=10, update_latency=0)
    ms_comp = cms_comp .* θs ./ ps
    vs_comp = cvs_comp .* θs ./ ps
    ds_comp = compute_latency_distribution.(ms_comp, vs_comp, θs, ps)
    nworkers = length(ps)
    latency = 0.0
    Ls = zeros(nworkers)
    for _ in 1:nsamples
        dfi, Lsi = simulate_iterations(;nwait, niterations, ds_comm, ds_comp, update_latency)
        latency += mean(dfi.latency)
        Ls .+= Lsi        
    end
    latency /= nsamples
    Ls ./= nsamples
    latency, Ls
end

"""

Minimize the variance of the vector `Ls .* θs ./ ps` using a genetic optimization algorithm.
"""
function balance_contribution(ps0::AbstractVector{<:Real}, min_processed_fraction::Real; θs::AbstractVector{<:Real}, comp_means, comp_vars, comm_means, comm_vars, nwait::Integer, populationSize::Integer=100, tournamentSize::Integer=10, mutationRate::Real=1.0, time_limit::Real=10.0)
    nworkers = length(ps0)
    length(θs) == nworkers || throw(DimensionMismatch("θs has dimension $(length(θs)), but nworkers is $nworkers"))
    length(comp_means) == nworkers || throw(DimensionMismatch("comp_means has dimension $(length(comp_means)), but nworkers is $nworkers"))
    length(comp_vars) == nworkers || throw(DimensionMismatch("comp_vars has dimension $(length(comp_vars)), but nworkers is $nworkers"))
    length(comm_means) == nworkers || throw(DimensionMismatch("comm_means has dimension $(length(comp_means)), but nworkers is $nworkers"))
    length(comm_vars) == nworkers || throw(DimensionMismatch("comm_vars has dimension $(length(comp_vars)), but nworkers is $nworkers"))    
    all((x)->0<x<=1, θs) || throw(ArgumentError("The entries of θs must be in (0, 1], but got $θs"))
    all((x)->0<x, ps0) || throw(ArgumentError("The entries of ps must be positive, but got $ps0"))

    # initialization
    ds_comm = distribution_from_mean_variance.(Gamma, comm_means, comm_vars)
    cms_comp = comp_means ./ (θs ./ ps)
    cvs_comp = comp_vars ./ (θs ./ ps)

    # constraint
    # (the total expected conbtribution must be above some threshold)
    function g(ps, Ls)
        fraction_processed(ps, θs, Ls) - min_processed_fraction
    end

    # objective function
    # (the variance of the contribution between workers)
    fworst = 0.0
    function f(ps)
        _, Ls = simulate(;θs, ps, cms_comp, cvs_comp, ds_comm, nwait)
        c = g(ps, Ls)
        if c < 0
            rv = fworst + abs(c)
        else
            rv = var(Ls .* θs ./ ps)
            fworst = max(fworst, rv)
        end
        rv
    end

    # evolutionary algorithm setup
    nworkers = length(ps0)
    selection = Evolutionary.tournament(tournamentSize)
    crossover = Evolutionary.LX()
    lower = max.(ones(nworkers), ceil.(Int, ps0 ./ 2))
    upper = 2 .* ps0
    mutation = Evolutionary.domainrange((lower .- upper) ./ 10) # as recommended in the BGA paper

    # integer-output mutation that wraps another mutation
    function integer_mutation(x)
        if isone(mutationRate) || rand() < mutationRate
            mutation(x)
        end
        for i in 1:length(x)
            if rand() < 0.5
                x[i] = floor(x[i])
            else
                x[i] = ceil(x[i])
            end
        end
        x
    end

    # optimization algorithm
    opt = Evolutionary.GA(;populationSize, mutationRate=1.0, selection, crossover, mutation=integer_mutation)
    options = Evolutionary.Options(;time_limit, Evolutionary.default_options(opt)...)
    Evolutionary.optimize(f, lower, upper, ps0, opt, options)
end

function load_balancer(chin::Channel; chout::Channel, ps0::AbstractVector, θs, min_processed_fraction::Real, nwait::Integer)
    @info "load_balancer task started"
    nworkers = length(ps0)
    0 < nworkers || throw(ArgumentError("nworkers must be positive, but is $nworkers"))
    length(θs) == nworkers || throw(DimensionMismatch("θs has dimension $(length(θs)), but nworkers is $nworkers"))
    ps = Vector{Int}(ps0)        
    comp_means = zeros(nworkers)
    comp_vars = zeros(nworkers)
    comm_means = zeros(nworkers)
    comm_vars = zeros(nworkers)

    function process_sample(v)
        0 < v.worker <= nworkers || throw(ArgumentError("v.worker is $(v.worker), but nworkers is $nworkers"))
        isnan(v.comp_mean) || comp_means[v.worker] = v.comp_mean
        isnan(v.comp_var) || comp_vars[v.worker] = v.comp_var
        isnan(v.comm_mean) || comm_means[v.worker] = v.comm_mean
        isnan(v.comm_var) || comm_vars[v.worker] = v.comm_var
        return
    end

    # helper to check if there is any missing latency data
    all_populated = false    
    function check_populated()
        if all_populated
            return all_populated
        end        
        all_populated = iszero(count(isnan, comp_means))
        if all_populated
            all_populated = all_populated && iszero(count(isnan, comp_vars))
        end
        if all_populated
            all_populated = all_populated && iszero(count(isnan, comm_means))
        end
        if all_populated
            all_populated = all_populated && iszero(count(isnan, comm_vars))
        end
        all_populated
    end

    while isopen(chin)

        # consume all values currently in the channel
        try
            vin = take!(chin)
            process_sample(vin)
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
                process_sample(vin)
            catch e
                if e isa InvalidStateException
                    break
                else
                    rethrow()
                end
            end
        end     
    
        # verify that we have complete latency information for all workers
        if !check_populated()
            continue
        end

        # run the load-balancer
        new = load_balance()

        # new = balance_contribution(ps, min_processed_fraction; θs, ds_comm, cms_comp, cvs_comp, nwait)
        @info "started load-balancing optimization"
        new = balance_contribution(ps, min_processed_fraction; θs, comp_means, comp_vars, comm_means, comm_vars, nwait)

        # push any changes into the output channel
        for i in 1:nworkers
            if new[i] != ps[i]
                vout = @NamedTuple{worker::Int,p::Int}(i, new[i])
                push!(chout, vout)
                ps[i] = new[i]
            end
        end
    end
    @info "load_balancer task finished"
    return
end
