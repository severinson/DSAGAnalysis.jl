# DSAGAnalysis

We have collected latency traces on Amazon Web Services (AWS) using these two kernels, which are available as `.csv` files [here](https://www.dropbox.com/sh/wa3s4yeasqeko5h/AABLPknDQO6TU2s-NDhzpI1Ia?dl=0). Traces collected using the latency and PCA kernels are prefixed with `latency` and `pca`, respectively. For the PCA files, the next section of the filename indicates the dataset used (see below). Finally, the last two sections of the filename indicates the AWS [instance type](https://aws.amazon.com/ec2/instance-types/) and [region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#concepts-available-regions) used.

The `.h5` files and corresponding `.csv` files that each trace file is the concatenation of are available in the `.zip` file with the same filename.

## Datasets

For the PCA kernel, we have collected traces using two different datasets, denoted by `1000enomes` and `sprand`, respectively. The `1000genomes` matrix is derived from the [1000 Genomes dataset](https://www.internationalgenome.org/). More precisely, we consider a binary representation of the data, where a non-zero entry in the `(i, j)`-th position indicates that the genome of the `i`-th subject differs from that of the reference genome in the `j`-th position, i.e., rows correspond to persons and columns to positions in the genome. The matrix is created by concatenating the sub-matrices corresponding to each chromosome and then randomly permuting the columns, which we do to break up dense blocks. It is a sparse matrix with dimensions `(2504, 81 271 767)` and density about 5.36%. When stored in compressed column format, with 64-bit indices, the matrix is about 100 gigabytes. The `sprand` matrix is created in Julia with the command `sprand(2504, 3600000, 0.05360388070027386)`.

Both matrices are available [here](https://www.dropbox.com/sh/ak5d9elhra2h4in/AAB2qqleIxAYTpVlxHba_q_0a?dl=0) and can be read with the [H5Sparse.jl](https://github.com/severinson/H5Sparse.jl) Julia package.



## Configuration

All experiments are carried out using the configuration listed below.

```julia
# setup
pkg> add MPI, MKL, MKLSparse
> ENV["JULIA_MPI_BINARY"] = "system"
pkg> build MPI
pkg> build MKL

# package versions
pkg> status MPI
[da04e1cc] MPI v0.16.1
pkg> status MKL
[33e6dc65] MKL v0.4.0
pkg> status MKLSparse
[0c723cd3] MKLSparse v1.1.0

# versioninfo
> versioninfo()
Julia Version 1.5.4
Commit 69fcb5745b (2021-03-11 19:13 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Platinum 8124M CPU @ 3.00GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-9.0.1 (ORCJIT, skylake-avx512)  

> MPI.Get_library_version()
"Open MPI v4.0.3, package: Open MPI root@b0fe8a010177 Distribution, ident: 4.0.3, repo rev: v4.0.3, Mar 03, 2020"

> MPI.Get_version()
v"3.1.0"  

> LinearAlgebra.versioninfo()
BLAS: libmkl_rt.so.1
LAPACK: libmkl_rt.so.1

> BLAS.vendor()
:mkl
```

## DataFrame columns

Each row of the concatenated DataFrame corresponds to one iteration of a particular job. The DataFrame resulting from the PCA kernel have the following columns.

* `iteration`: Iteration index
* `jobid`: Unique ID for each run of the kernel, i.e., each unique `jobid` corresponds to one `.h5` file
* `latency`: Overall latency of the iteration
* `latency_worker_<i>`: Latency of the `i`-th worker in this iteration
* `mse`: Explained variance of the iterate computed in this iteration
* `nbytes`: Total number of bytes transferred in each direction per iteration
* `ncolumns`: Number of columns of the data matrix
* `ncomponents`: Number of PCs computed
* `niterations`: Total number of iterations of this job
* `nostale`: If `true`, stale results received by the coordinator are discarded (only relevant when `variancereduced` is `true`)
* `npartitions`: Total number of data partitions
* `nreplicas`: Number of replicas of each partition of the data matrix
* `nrows`: Number of rows of the data matrix
* `nsubpartitions`: Number of sub-partitions per worker
* `nwait`: Number of workers waited for in each iterations
* `nworkers`: Total number of workers for this run
* `repoch_worker_<i>`: Iteration that the result received from the `i`-th worker was computed for, i.e., the result is stale if it is less than `iteration`
* `saveiterates`: If `true`, iterates were saved for this job
* `stepsize`: Step size used for the gradient update
* `time`: Cumulative iteration latency up this iteration
* `update_latency`: Latency associated with computing the updated iterate at the coordinator
* `variancereduced`: If `true`, the variance-reduced DSAG method was used for this job, whereas, if `false`, SGD was used
* `worker_flops`: Estimated number of FLOPS per worker and iteration

DataFrames resulting from the latency kernel have the following columns. Columns with no explanation have the same meaning as above.

* `nwait`:
* `timeout`: Amount of time that the coordinator waits between iterations
* `nworkers`:
* `ncomponents`
* `nbytes`
* `niterations`
* `nrows`
* `latency`
* `iteration`
* `ncols`
* `density`: Matrix density
* `timestamp`: Iteration timestamp in nanoseconds
* `jobid`
* `worker_flops`
* `time`
* `repoch_worker_<i>`
* `latency_worker_<i>`
* `compute_latency_worker_<i>`: Compute latency of the `i`-th worker

Converting the DataFrame to tall format deletes the following columns:

* `latency_worker_<i>`
* `repoch_worker_<i>`
* `compute_latency_worker_<i>`

And introduces:

* `worker_index`: Index of the worker that this row corresponds to
* `isstraggler`: Is `true` if the worker was a straggler in this iteration
* `worker_latency`: Overall latency of this worker
* `worker_compute_latency`: Compute latency of this worker
* `repoch`: Iteration that the result received from the `i`-th worker was computed for, i.e., the result is stale if it is less than `iteration`
* `order`: Completion order of this worker for this iteration, e.g., if `order=3`, then this worker was the third fastest in this iteration
* `compute_order`: Same as `order`, but for `worker_compute_latency`



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