# DSAGAnalysis

```julia
# AWS, 1000 genomes
using Revise, CodedComputing; includet("src\\Analysis.jl"); using CSV, DataFrames; df = DataFrame(CSV.File("C:\\Users\\albin\\Dropbox\\PhD\\Eigenvector project\\AWS traces\\traces\\pca-1000genomes-c5.xlarge-eu-north-1.csv")); strip_columns!(df); fix_update_latency!(df); remove_initialization_latency!(df); notes="AWS, 1000genomes";

# AWS, 1000 genomes v2 (w. sep. computational latency)
using Revise, CodedComputing; includet("src\\Analysis.jl"); using CSV, DataFrames; df = DataFrame(CSV.File("C:\\Users\\albin\\Dropbox\\PhD\\Eigenvector project\\AWS traces\\traces\\pca-1000genomes-c5.xlarge-eu-north-1_v2.csv")); strip_columns!(df); fix_update_latency!(df); remove_initialization_latency!(df); notes="AWS, 1000genomes v2";

# AWS, 1000 genomes dense equiv. (w. sep. computational latency)
using Revise, CodedComputing; includet("src\\Analysis.jl"); using CSV, DataFrames; df = DataFrame(CSV.File("C:\\Users\\albin\\Dropbox\\PhD\\Eigenvector project\\AWS traces\\traces\\pca-1000genomes-dense-equiv-c5.xlarge-eu-north-1.csv")); strip_columns!(df); fix_update_latency!(df); remove_initialization_latency!(df); df.ncolumns .= 4356480; df.worker_flops .= CodedComputing.worker_flops_from_df(df, density=1); notes="AWS, dense equiv.";

# AWS, latency
using Revise, CodedComputing; includet("src\\Analysis.jl"); using CSV, DataFrames; df = DataFrame(CSV.File("C:\\Users\\albin\\Dropbox\\PhD\\Eigenvector project\\AWS traces\\traces\\latency-c5.xlarge-eu-north-1.csv")); strip_columns!(df); fix_update_latency!(df); remove_initialization_latency!(df); notes = "AWS, latency"; notes="AWS, latency";

# ex3, 1000 genomes
using Revise, CodedComputing; includet("src\\Analysis.jl"); using CSV, DataFrames; df = DataFrame(CSV.File("C:\\Users\\albin\\Dropbox\\PhD\\Eigenvector project\\ex3 traces\\pca-1000genomes-ex3.rome16q.csv")); strip_columns!(df); fix_update_latency!(df); remove_initialization_latency!(df); notes="eX3, 1000genomes";

# ex3, dense equivalent
using Revise, CodedComputing; includet("src\\Analysis.jl"); using CSV, DataFrames; df = DataFrame(CSV.File("C:\\Users\\albin\\Dropbox\\PhD\\Eigenvector project\\ex3 traces\\pca-1000genomes-dense-equiv-ex3.rome16q.csv")); strip_columns!(df); fix_update_latency!(df);remove_initialization_latency!(df); notes="eX3, dense equiv.";

# Azure, dense equiv.
using Revise, CodedComputing; includet("src\\Analysis.jl"); using CSV, DataFrames; df = DataFrame(CSV.File("C:\\Users\\albin\\Dropbox\\PhD\\Eigenvector project\\Azure traces\\pca-1000genomes-dense-equiv-azure.hpc.F2s_v2.csv")); strip_columns!(df); fix_update_latency!(df); remove_initialization_latency!(df); df.ncolumns .= 4356480; df.worker_flops .= CodedComputing.worker_flops_from_df(df, density=1); notes="Azure, dense equiv.";

# Azure, 1000 genomes
using Revise, CodedComputing; includet("src\\Analysis.jl"); using CSV, DataFrames; df = DataFrame(CSV.File("C:\\Users\\albin\\Dropbox\\PhD\\Eigenvector project\\Azure traces\\pca-1000genomes-azure.hpc.F2s_v2.csv")); strip_columns!(df); remove_initialization_latency!(df); fix_update_latency!(df); notes="Azure, 1000genomes";
df = filter(:niterations => (x)->x==100, df);

# initialization
using Revise # optional, needed for changes made to the source code be reflected in the REPL
using DSAGAnalysis
const m = DSAGAnalysis

## load pca data from disk
using CSV, DataFrames
df = DataFrame(CSV.File("<path-to-traces-directory>/pca-1000genomes-c5.xlarge-eu-north-1.csv"))
m.strip_columns!(df) # optional, to reduce DataFrame size
m.remove_initialization_latency!(df) # remove latency in the first iteration that is due to, e.g., compilation

# plot the per-iteration latency of workers 1 and 2 of job 10
jobid = 10
worker_indices = [1, 2]
m.plot_timeseries(df, jobid, worker_indices)
m.plot_timeseries(df, jobid, worker_indices, cumulative=true)

# plot the total iteration latency for job 10
m.plot_timeseries(df, jobid)
m.plot_timeseries(df, jobid, cumulative=true)

# plot the per-iteration latency distribution for a particular set of workers
m.plot_worker_latency_distribution(df, jobid, worker_indices)

# plot order statistics latency for a particular job
m.plot_orderstats(df, jobid)

# compute the mean variance of the latency of each worker (needed for predictions later)
dfg = m.per_worker_statistics_df(df)

# plot the distribution of the mean and variance associated with each worker for a particular workload
nbytes = 30048
nflops = 2.84e6
m.plot_worker_mean_var_distribution(dfg; nbytes, nflops)

# plot order statistics latency for a particular job
m.plot_orderstats(df, jobid)

# plot prior order statistics
nworkers = 72
m.plot_prior_orderstats(df; nworkers, nflops, model=(df_comm, df_comp))

# plot autocorrelation
m.plot_autocorrelation(df; nbytes, nflops)

# plot cross-correlation
m.plot_worker_latency_cov_cdf(df; nbytes, nflops)

# compute prior distribution statistics
df_comm, df_comp, df_total = m.prior_statistics_df(dfg)

# compute predicted and empirical latency
m.plot_latency_vs_nworkers(;df_comm, df_comp, df)
```