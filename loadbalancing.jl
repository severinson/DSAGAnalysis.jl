using DSAGAnalysis
using Random, StatsBase, Distributions
using Evolutionary

function of(nwait, ps, ds, zmin, hmin)
    nworkers = length(ps)
    df, Ls = simulate_iterations(;nwait, niterations=100, ds_comm=fill(nothing, nworkers), ds_comp=ds, update_latency=0)
    θ = 1/nworkers
    z = sum(Ls .* θ ./ ps)
    if z < zmin
        return Inf
    end
    h = minimum(Ls ./ ps)
    if h < hmin
        return Inf
    end
    mean(df.latency)
end


"""

Select the number of partitions to minimize the mean squared expected latency across all workers,
while maintaining the size of the fraction of the dataset processed per iteration.

* h: target size of the fraction of the dataset processed in each iteration
* ps: current number of partitions per worker
* θs: fraction of the dataset stored at each worker
* ms_comp: expected compute latency of each worker
* ms_comm: expected communication latency of each worker

"""
function optimize_partitions(;h, ps, θs, ms_comp, ms_comm)
    nworkers = length(θs)
    
    # estimate the underlying coefficients for each worker determining the mean and variance
    cms = ms_comp ./ (θs ./ ps)

    # System of equations based on the Lagrangian
    A = zeros(nworkers+1, nworkers+1)
    b = zeros(nworkers+1)
    for i in 1:nworkers
        A[i, i] = 2 .* (θs[i] .* cms[i]).^2
    end
    A[1:nworkers, end] .= -θs
    b[1:nworkers] .= -2θs .* cms .* ms_comm 

    # constraint
    A[end, 1:nworkers] .= θs
    b[end] = h

    # solve the system of equations
    x = A \ b
    qs = x[1:nworkers]
    1 ./ qs
end

"""

Update all partitions, except the `j`-th, such that a fraction `h` of the dataset is processed in
each iteration.
"""
function balance_partitions!(ps, j; h, θs)
    num = h - θs[j]/ps[j]
    den = 0.0
    for i in 1:length(ps)
        if i != j
            den += θs[i] / ps[i]
        end
    end
    c = num / den
    for i in 1:length(ps)
        if i != j
            ps[i] /= c
        end
    end
    ps
end

"""

"""
function compute_latency_distribution(cm, cv, θ, p)
    m = cm * θ / p
    v = cv * θ / p
    DSAGAnalysis.distribution_from_mean_variance.(ShiftedExponential, m, v)        
end



function fraction_processed(ps, θs, Ls)
    rv = 0.0
    for i in 1:length(Ls)
        rv += Ls[i] * θs[i] / ps[i]
    end
    rv
end

function load_balance_workers!(ps, h; θs, ds_comm, cms_comp, cvs_comp, nwait, niterations=100)
    nworkers = length(ps)

    function loss(L::Real)
        (L - nwait/nworkers)^2
    end

    function loss(Ls::AbstractVector{<:Real})
        rv = zero(eltype(Ls))
        for L in Ls
            rv += loss(L)
        end
        rv
    end    

    function select_worst(ps, Ls)
        j = 0
        v = 0
        for (i, L) in enumerate(Ls)
            l = loss(L)
            if l > v
                if (L > nwait/nworkers && ps[i] > 1) || (L < nwait/nworkers)
                    j = i
                    v = l
                end
           end
        end
        j
    end

    function constraint(h, ps, Ls)
        fraction_processed(ps, θs, Ls) >= h
    end

    function satisfy_constraint!(ps, Ls)
        while !constraint(h, ps, Ls)

            # select the worker most likely to be among the nwait fastest for which the number of 
            # partitions is at least 1
            j = 0
            v = 0.0
            for (i, L) in enumerate(Ls)
                if ps[i] > 1 && L > v
                    j = i
                    v = L
                end
            end

            # decrease the number of partitions for this worker, thus increasing the fraction of
            # the dataset processed per iteration
            ps[j] -= 1
            _, Ls = simulate(;θs, ps, cms_comp, cvs_comp, ds_comm, nwait)
        end
        ps
    end

    # Question: does it matter for which value of w I do this (other than nworkers)?


    # load-balancing across workers
    for _ in 1:niterations        
        # _, Ls = simulate(ps)
        _, Ls = simulate(;θs, ps, cms_comp, cvs_comp, ds_comm, nwait)
        j = select_worst(ps, Ls)
        if Ls[j] < nwait/nworkers
            ps[j] += 1
        else
            ps[j] -= 1
        end
        # println("After update")
        # println(Ls)
        # println(ps)
        # println(loss(Ls))
        # println("$(fraction_processed(ps, Ls)) / $h")
        # println()

        satisfy_constraint!(ps, Ls)
        # println("After constraint check")        
        # println(Ls)
        # println(ps)
        # println(loss(Ls))
        # println("$(fraction_processed(ps, Ls)) / $h")        
        # println()
    end

    ps
end

function load_balance_samples!(ps, h; θs, ds_comm, cms_comp, cvs_comp, nwait, niterations=200)
    nworkers = length(ps)

    function losses(ps, Ls)
        vs = Ls .* θs ./ ps
        (vs .- mean(vs)).^2
    end

    function loss(ps, Ls)
        var(Ls .* θs ./ ps)
    end    

    function select_worst(ps, Ls)
        argmax(losses(ps, Ls))
    end

    function constraint(h, ps, Ls)
        fraction_processed(ps, θs, Ls) >= h
    end

    function satisfy_constraint!(ps, Ls)
        while !constraint(h, ps, Ls)

            # select the worker most likely to be among the nwait fastest for which the number of 
            # partitions is at least 1
            j = 0
            v = 0.0
            for (i, L) in enumerate(Ls)
                if ps[i] > 1 && L > v
                    j = i
                    v = L
                end
            end

            # decrease the number of partitions for this worker, thus increasing the fraction of
            # the dataset processed per iteration
            ps[j] -= 1
            _, Ls = simulate(;θs, ps, cms_comp, cvs_comp, ds_comm, nwait)
        end
        ps
    end

    function finite_differences(j, ps)
        ps[j] += 1
        _, Ls = simulate(;θs, ps, cms_comp, cvs_comp, ds_comm, nwait)
        forward = loss(ps, Ls)
        ps[j] -= 2
        _, Ls = simulate(;θs, ps, cms_comp, cvs_comp, ds_comm, nwait)
        backward = loss(ps, Ls)
        ps[j] += 1
        (forward - backward) / 2
    end

    # solve
    _, Ls = simulate(;θs, ps, cms_comp, cvs_comp, ds_comm, nwait)
    best = loss(ps, Ls)
    for _ in 1:niterations        
        j = rand(1:nworkers)
        p = ps[j]

        ps[j] -= 1
        _, Ls_new = simulate(;θs, ps, cms_comp, cvs_comp, ds_comm, nwait)
        new = loss(ps, Ls_new)
        if rand() < 0.1 || (constraint(h, ps, Ls_new) && new < best)
            p = ps[j]
            best = new
            Ls = Ls_new
        end

        ps[j] += 2
        _, Ls_new = simulate(;θs, ps, cms_comp, cvs_comp, ds_comm, nwait)
        new = loss(ps, Ls_new)
        if rand() < 0.1 || (constraint(h, ps, Ls_new) && new < best)
            p = ps[j]
            best = new
            Ls = Ls_new
        end

        ps[j] = p

        # println("After update")
        # println(Ls)
        # println(ps)
        # println(best)
        # println("$(fraction_processed(ps, θs, Ls)) / $h")
        # println()
    end    

    # # (old solver)
    # for _ in 1:niterations        
    #     _, Ls = simulate(;θs, ps, cms_comp, cvs_comp, ds_comm, nwait)
    #     j = select_worst(ps, Ls)
    #     δ = finite_differences(j, ps)
    #     ps[j] -= sign(δ)
    #     # println("After update")
    #     # println(Ls)
    #     # println(ps)
    #     # println(loss(ps, Ls))
    #     # println("$(fraction_processed(ps, θs, Ls)) / $h")
    #     # println()

    #     satisfy_constraint!(ps, Ls)
    #     # println("After constraint check")        
    #     # println(Ls)
    #     # println(ps)
    #     # println(loss(ps, Ls))
    #     # println("$(fraction_processed(ps, θs, Ls)) / $h")        
    #     # println()
    # end

    ps
end

function main(nworkers=10, nwait=5, nwait_max=8, hmin=0.01, c0=1e9, p0=10.0, b=100)

    Random.seed!(123)

    # iterative load-balancing
    θs = fill(1/nworkers, nworkers)
    ps = fill(p0, nworkers)
    h = sum(θs ./ ps) * nwait / nworkers / 2
    ps0 = ps    

    # compute latency distribution
    ms_comp = rand(LogNormal(1, 0.1), nworkers)
    vs_comp = rand(LogNormal(0.1, 0.01), nworkers)
    cms_comp = ms_comp ./ (θs ./ ps)#  ./ (c0 / 1e9)
    cvs_comp = vs_comp ./ (θs ./ ps)#  ./ (c0 / 1e9)
    ds_comp = DSAGAnalysis.distribution_from_mean_variance.(ShiftedExponential, ms_comp, vs_comp)

    # communication latency distribution
    ms_comm = rand(LogNormal(1, 0.1), nworkers) ./ 100
    vs_comm = rand(LogNormal(0.1, 0.01), nworkers) ./ 100
    ds_comm = DSAGAnalysis.distribution_from_mean_variance.(ShiftedExponential, ms_comm, vs_comm)    

    println()
    println("Uniform")
    latency, Ls = simulate(;θs, ps=ps, cms_comp, cvs_comp, ds_comm, nwait, nsamples=100)

    println("Probs.")    
    println(ps)
    println(Ls)    
    println((latency, fraction_processed(ps, θs, Ls)))

    println("Contribs.")
    contribs = Ls .* θs ./ ps
    println(contribs)    
    println((mean(contribs), var(contribs)))

    println("Latencies")
    ms = cms_comp .* θs ./ ps .+ ms_comm    
    println(ms)
    println((mean(ms), var(ms)))

    println()
    println("Evolutionary contribution")
    result = balance_contribution(copy(ps0), h; θs, ds_comm, cms_comp, cvs_comp, nwait)
    ps = result.minimizer
    latency, Ls = simulate(;θs, ps=ps, cms_comp, cvs_comp, ds_comm, nwait, nsamples=100)

    println("Probs.")
    println(ps)    
    println(Ls)
    println((latency, fraction_processed(ps, θs, Ls)))    

    println("Contribs.")
    contribs = Ls .* θs ./ ps
    println(contribs)    
    println((mean(contribs), var(contribs)))    

    println("Latencies")
    ms = cms_comp .* θs ./ ps .+ ms_comm
    println(ms)
    println((mean(ms), var(ms)))
    
    return    

    println()
    println("Lagrangian")    
    ps = optimize_partitions(;h=h / (nwait / nworkers), ps, θs, ms_comp, ms_comm)
    ps .= floor.(Int, ps)
    ps .= max.(ps, 1)    
    latency, Ls = simulate(;θs, ps=ps, cms_comp, cvs_comp, ds_comm, nwait)

    println("Probs.")
    println(ps)    
    println(Ls)
    println((latency, fraction_processed(ps, θs, Ls)))

    println("Contribs.")
    contribs = Ls .* θs ./ ps
    println(contribs)    
    println((mean(contribs), var(contribs)))    

    println("Latencies")
    ms = cms_comp .* θs ./ ps .+ ms_comm
    println(ms)
    println((mean(ms), var(ms)))

    println()
    println("Worker-balanced")
    ps = load_balance_workers!(copy(ps0), h; θs, ds_comm, cms_comp, cvs_comp, nwait)
    latency, Ls = simulate(;θs, ps=ps, cms_comp, cvs_comp, ds_comm, nwait)

    println("Probs.")
    println(ps)    
    println(Ls)
    println((latency, fraction_processed(ps, θs, Ls)))    

    println("Contribs.")
    contribs = Ls .* θs ./ ps
    println(contribs)    
    println((mean(contribs), var(contribs)))    

    println("Latencies")
    ms = cms_comp .* θs ./ ps .+ ms_comm
    println(ms)
    println((mean(ms), var(ms)))    

    println()
    println("Sample-balanced")    
    ps = load_balance_samples!(copy(ps0), h; θs, ds_comm, cms_comp, cvs_comp, nwait)
    latency, Ls = simulate(;θs, ps=ps, cms_comp, cvs_comp, ds_comm, nwait)

    println("Probs.")
    println(ps)    
    println(Ls)
    println((latency, fraction_processed(ps, θs, Ls)))    

    println("Contribs.")
    contribs = Ls .* θs ./ ps
    println(contribs)    
    println((mean(contribs), var(contribs)))    

    println("Latencies")
    ms = cms_comp .* θs ./ ps .+ ms_comm
    println(ms)
    println((mean(ms), var(ms)))

    # latency, Ls = simulate(;θs, ps=ps, cms_comp, cvs_comp, ds_comm, nwait)
    # println("Event-driven")
    # println(ps)        
    # println((latency, Ls, fraction_processed(ps, θs, Ls)))    

    return

    # balancing real-valued partitions
    θs = fill(1/nworkers, nworkers)        
    ps = fill(p0, nworkers)
    h = sum(θs ./ ps)
    j = 1
    ps[j] *= 10
    println("h: $h")
    balance_partitions!(ps, j; h, θs)
    h = sum(θs ./ ps)    
    println("h: $h")
    
    return ps

    # Random.seed!(123)

    # compute latency distribution
    cms_comp = rand(nworkers) ./ 1e9
    cvs_comp = rand(nworkers) ./ 1e9

    # communication latency distribution
    cms_comm = rand(nworkers) ./ 1e9
    cvs_comm = rand(nworkers) ./ 1e9    
    
    # fraction of dataset stored on each worker and current number of partitions
    θs = fill(1/nworkers, nworkers)        
    ps0 = fill(p0, nworkers) # uniform partitioning

    # target size of the fraction of the dataset to process per iteration
    ztarget = sum(θs ./ p0)

    # compute latency mean and variance
    ms_comp = cms_comp .* c0 .* θs ./ ps0
    vs_comp = cvs_comp .* c0 .* θs ./ ps0

    # communication latency mean and variance
    ms_comm = cms_comm .* b
    vs_comm = cvs_comm .* b

    ps = optimize_partitions(;ztarget, ps0, θs, c0, ms_comp, ms_comm)

    ms_comp_new = cms_comp .* c0 .* θs ./ ps
    vs_comp_new = cvs_comp .* c0 .* θs ./ ps

    # println(θs ./ ps0)
    # println(θs .* qs)
    println("ztarget: $ztarget, z: $(sum(θs ./ ps))")
    old_loss = sum((ms_comp .+ ms_comm).^2)
    new_loss = sum((ms_comp_new .+ ms_comm).^2)
    println("Old loss: $old_loss, new loss: $new_loss")
    println("Old:")
    println(ms_comp)
    println("New:")
    println(ms_comp_new)


    ds_comp = DSAGAnalysis.distribution_from_mean_variance.(ShiftedExponential, ms_comp, vs_comp)    
    ds_comm = DSAGAnalysis.distribution_from_mean_variance.(ShiftedExponential, ms_comm, vs_comm)
    df, Ls = simulate_iterations(;nwait, niterations=100, ds_comm, ds_comp, update_latency=0)
    println("Ls (old):\t$Ls")

    ds_comp = DSAGAnalysis.distribution_from_mean_variance.(ShiftedExponential, ms_comp_new, vs_comp_new)
    df, Ls = simulate_iterations(;nwait, niterations=100, ds_comm, ds_comp, update_latency=0)    
    println("Ls:\t\t$Ls")

    return ps    

    # Fascinating that the optimization approach I considered seems to make things worse
    # Let's re-think. I want to load-balance, i.e., I want all Ls to have the correct value
    # Let's address that using an iterative strategy, where I iteratively improve the worker furthest from its target
end

function evo_main()

    # constraint
    # (the sum of all elements must be >= 3)
    g = x -> sum(x) - 4
    fworst = 0.0
    function f(x)
        c = g(x)
        if c < 0
            rv = fworst + abs(c)
            println("not feasible: $rv")            
        else
            rv = sum(x.^2)
            println("feasible: $rv")
            fworst = max(fworst, rv)
        end
        rv
    end

    x0 = fill(2.0, 3)

    populationSize = 100
    selection = Evolutionary.tournament(10)
    crossover = Evolutionary.LX()
    lower = ones(3)
    upper = fill(10, 3)
    mutation = Evolutionary.PM(lower, upper)

    function int_mutation(x)
        mutation(x)
        for i in 1:length(x)
            if rand() < 0.5
                x[i] = floor(x[i])
            else
                x[i] = ceil(x[i])
            end
        end
        x
    end

    rv = Evolutionary.optimize(f, x0, Evolutionary.GA(;populationSize, selection, crossover, mutation=int_mutation))

    println("fworst: $fworst")    
    rv
end