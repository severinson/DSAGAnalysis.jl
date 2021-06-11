# Statistics code

export ShiftedExponential, ExponentialOrder, OrderStatistic, NonIDOrderStatistic

"""
    ExponentialOrder(scale::Real, total::Int, order::Int)

Random variable representing the order-th largest value out of total
realizations of an exponential random variable with given scale.

"""
function ExponentialOrder(scale::Real, total::Int, order::Int)
    scale > 0 || throw(DomainError(scale, "scale must be positive"))
    total > 0 || throw(DomainError((total, order), "total must be positive"))
    order > 0 || throw(DomainError((total, order), "order must be positive"))
    order <= total || throw(DomainError((total, order), "order must be <= total"))
    var = sum(1/(i^2)  for i=(total-order+1):total) * scale^2
    mean = sum(1/i for i=(total-order+1):total) * scale
    alpha = mean^2 / var # shape parameter
    theta = var / mean # scale
    return Gamma(alpha, theta)
end

### Shifted exponential distribution

struct ShiftedExponential{T<:Real} <: ContinuousUnivariateDistribution
    shift::T
    exponential::Exponential{T}
    ShiftedExponential{T}(s::T, θ::T) where T = new{T}(s, Exponential(θ))
end

function ShiftedExponential(s::T, θ::T) where {T<:Real}
    θ > zero(θ) || throw(DomainError(θ), "θ must be positive")
    ShiftedExponential{T}(s, θ)
end

function ShiftedExponential(s::T, θ::T) where {T<:Integer}
    ShiftedExponential(float(s), float(θ))
end

struct ShiftedExponentialStats <: SufficientStats
    vx::Float64     # variance of x
    sx::Float64     # (weighted) sum of x
    sw::Float64     # sum of sample weights
end

Distributions.suffstats(::Type{<:ShiftedExponential}, x::AbstractArray{T}) where {T<:Real} = ShiftedExponentialStats(var(x), sum(x), length(x))

function Distributions.fit_mle(::Type{<:ShiftedExponential}, ss::ShiftedExponentialStats)
    θ = sqrt(ss.vx)
    s = ss.sx / ss.sw - θ
    ShiftedExponential(s, θ)
end

Distributions.quantile(d::ShiftedExponential, p::Real) = Distributions.quantile(d.exponential, p) + d.shift

Distributions.pdf(d::ShiftedExponential, p::Real) = pdf(d.exponential, p-d.shift)
Distributions.cdf(d::ShiftedExponential, p::Real) = cdf(d.exponential, p-d.shift)
Base.rand(rng::AbstractRNG, d::ShiftedExponential) = rand(rng, d.exponential) + d.shift
Base.minimum(d::ShiftedExponential) = d.shift
Base.maximum(::ShiftedExponential) = Inf
Statistics.mean(d::ShiftedExponential) = mean(d.exponential) + d.shift
Statistics.var(d::ShiftedExponential) = var(d.exponential)
Distributions.params(d::ShiftedExponential) = (d.shift, params(d.exponential)...)

### Fitting a distribution from its mean and variance

function distribution_from_mean_variance(::Type{ShiftedExponential}, m, v)
    θ = sqrt(v)
    s = m .- θ
    ShiftedExponential(s, θ)
end

function distribution_from_mean_variance(::Type{Gamma}, m, v)
    θ = v / m
    α = m / θ
    Gamma(α, θ)
end

### Order statistics sampling

struct OrderStatistic{S<:Union{Discrete,Continuous},Spl<:Sampleable{Univariate,S},T} <: Sampleable{Univariate,S}
    spl::Spl
    k::Int
    buffer::Vector{T}
end

Base.show(io::IO, s::OrderStatistic) = print(io, "OrderStatistic($(s.spl), k=$(s.k), n=$(length(s.buffer)))")

OrderStatistic(s::Sampleable, k::Integer, n::Integer) = OrderStatistic(s, k, Vector{eltype(s)}(undef, n))

function Random.rand(rng::AbstractRNG, s::OrderStatistic)
    Distributions.rand!(rng, s.spl, s.buffer)
    partialsort!(s.buffer, s.k)
    s.buffer[s.k]
end

struct NonIDOrderStatistic{S<:Union{Discrete,Continuous},Spl<:Sampleable{Univariate,S},T} <: Sampleable{Univariate,S}
    spls::Vector{Spl}
    k::Int
    buffer::Vector{T}
end

Base.show(io::IO, s::NonIDOrderStatistic) = print(io, "NonIDOrderStatistic($(eltype(s.spls)), k=$(s.k), n=$(length(s.buffer)))")

NonIDOrderStatistic(spls::AbstractVector{<:Sampleable}, k::Integer) = NonIDOrderStatistic(spls, k, Vector{promote_type((eltype(spl) for spl in spls)...)}(undef, length(spls)))

function Random.rand(rng::AbstractRNG, s::NonIDOrderStatistic)
    for (i, spl) in enumerate(s.spls)
        s.buffer[i] = Distributions.rand(rng, spl)
    end
    partialsort!(s.buffer, s.k)
    s.buffer[s.k]
end