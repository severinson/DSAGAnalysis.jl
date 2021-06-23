# Code for parsing libsvm files

export sparse_from_libsvm_file, write_libsvm_dataset

function parse_libsvm_token(s::AbstractString; Ti, Tv)
    tokens = split(s, ':')
    length(tokens) == 2 || error("Malformed token: $s")
    parse(Ti, tokens[1]), parse(Tv, tokens[2])
end

"""
    sparse_from_libsvm_file(filename::AbstractString; Tl=Int, Ti=Int, Tv=Float32)

Read a libsvm file and return the result as a tuple `(data, labels)`, where `data` is a matrix of 
type `SparseMatrixCSC{Ti,Tv}`, for which each column corresponds to a sample, and `labels` is a 
vector composed of the label for each sample of type `Vector{Tl}`.

Each line of a libsvm file corresponds to a sample, and is of the format 
"<label> <index1>:<value1> <index2>:<value2> ...", where label is an integer denoting the label of
that sample, and each index-value pair corresponds to a particular feature. Note that, e.g., 
<label> is a palceholder, i.e., the < and > characters are not part of the file. For example, 
10:0.5 means that the 10-th feature has value 0.5 for this sample.
"""
function sparse_from_libsvm_file(filename::AbstractString; Tl=Int16, Ti=Int32, Tv=Float32)
    labels = zeros(Tl, 0)
    Is = zeros(Ti, 0) # row indices
    Js = zeros(Ti, 0) # column indices
    Vs = zeros(Tv, 0) # values
    for (j, line) in enumerate(eachline(filename))
        tokens = split(line, " ")
        if length(tokens) <= 1
            continue
        end
        label = parse(Tl, tokens[1])
        push!(labels, label)
        for k in 2:length(tokens)
            i, v = parse_libsvm_token(tokens[k]; Ti, Tv)
            push!(Is, i)
            push!(Js, j)
            push!(Vs, v)
        end
    end
    sparse(Is, Js, Vs), labels
end

function write_libsvm_dataset(filename, data, labels; dataname="X", labelname="b")
    fid = h5open(filename, "cw")
    H5SparseMatrixCSC(fid, dataname, data)
    fid[labelname] = labels
    fid
end