using Knet
function createDict(file, tok2int, int2tok, sequences)
    global strings = map(chomp,readlines(file))
    global words = map(split, strings)
    push!(int2tok,"\n"); tok2int["\n"]=1 # We use '\n'=>1 as the EOS token                                                 
    
    for w in words
    s = Vector{Int}()
        for c in w
            #assign each word a unique int
            if !haskey(tok2int,c)
                push!(int2tok,c)
                tok2int[c] = length(int2tok)
            end
            push!(s, tok2int[c])
        
    end
    push!(sequences, s)
    end
    return tok2int, int2tok, sequences
end


function createDictDir(path)    
    files = readdir(path)
    global tok2int = Dict{String,Int}()
    global int2tok = Vector{String}()
    sequences = Vector{Vector{Int}}()

    for f in files
        #update the vocabulary by adding new words from each file
        tok2int,int2tok,sequences = createDict(string(path,f),tok2int,int2tok, sequences)
    end
    return tok2int, int2tok, sequences
end

function lstm(param, state, input)
    weight,bias = param
    hidden,cell = state
    h       = size(hidden,2)
    gates   = hcat(input,hidden) * weight .+ bias
    forget  = sigm(gates[:,1:h])
    ingate  = sigm(gates[:,1+h:2h])
    outgate = sigm(gates[:,1+2h:3h])
    change  = tanh(gates[:,1+3h:4h])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

function initmodel(H, V; atype=(gpu()>=0 ? KnetArray{Float32} : Array{Float32}))
    init(d...)=atype(xavier(d...))
    model = Dict{Symbol,Any}()
    model[:state0] = [ init(1,H), init(1,H) ]
    model[:embed1] = init(V,H)
    model[:encode] = [ init(2H,4H), init(1,4H) ]
    model[:embed2] = init(V,H)
    model[:decode] = [ init(2H,4H), init(1,4H) ]
    model[:output] = [ init(H,V), init(1,V) ]
    return model
end


function s2s(model, inputs, outputs)
    state = initstate(inputs[1], model[:state0])
    for input in reverse(inputs)
        input = onehotrows(input, model[:embed1])
        input = input * model[:embed1]
        state = lstm(model[:encode], state, input)
    end
    EOS = eosmatrix(outputs[1], model[:embed2])
    input = EOS * model[:embed2]
    sumlogp = 0
    for output in outputs
        state = lstm(model[:decode], state, input)
        ypred = predict(model[:output], state[1])
        ygold = onehotrows(output, model[:embed2])
        sumlogp += sum(ygold .* logp(ypred,2))
        input = ygold * model[:embed2]
    end
    state = lstm(model[:decode], state, input)
    ypred = predict(model[:output], state[1])
    sumlogp += sum(EOS .* logp(ypred,2))
    return -sumlogp
end

function predict(param, input)
    input * param[1] .+ param[2]
end

function initstate(idx, state0)
    h,c = state0
    h = h .+ fill!(similar(AutoGrad.getval(h), length(idx), length(h)), 0)
    c = c .+ fill!(similar(AutoGrad.getval(c), length(idx), length(c)), 0)
    return (h,c)
end

function onehotrows(idx, embeddings)
    nrows,ncols = length(idx), size(embeddings,1)
    z = zeros(Float32,nrows,ncols)
    @inbounds for i=1:nrows
        z[i,idx[i]] = 1
    end
    oftype(AutoGrad.getval(embeddings),z)
end

let EOS=nothing; global eosmatrix
function eosmatrix(idx, embeddings)
    nrows,ncols = length(idx), size(embeddings,1)
    if EOS==nothing || size(EOS) != (nrows,ncols)
        EOS = zeros(Float32,nrows,ncols)
        EOS[:,1] = 1
        EOS = oftype(AutoGrad.getval(embeddings), EOS)
    end
    return EOS
end
end

tok2int, int2tok, sequences = createDictDir("data/test/");

function minibatch(sequences, batchsize)
    table = Dict{Int,Vector{Vector{Int}}}()
    data = Any[]
    for s = 1:2:length(sequences)-1
        n = length(sequences[s])
        m = length(sequences[s+1])
        longer = n
        if(n>m)
            for k = m+1:n
                push!(sequences[s+1],0)
            end
        end
         if(m>n)
            for k = n+1:m
                push!(sequences[s],0)
            end
            longer = m
        end
        nsequences = get!(table,longer, Any[])
        push!(nsequences, sequences[s])
        push!(nsequences, sequences[s+1])
        if length(nsequences) == batchsize
            push!(data, [[ nsequences[i][j] for i in 1:batchsize] for j in 1:n ])
            empty!(nsequences)
        end
    end
    return data
end

#function minibatch(sequences, batchsize)
    #table = Dict{Int,Vector{Vector{Int}}}()
    #data = Any[]
    #for s in sequences
        #n = length(s)
        #nsequences = get!(table, n, Any[])
        #push!(nsequences, s)
        #if length(nsequences) == batchsize
            #push!(data, [[ nsequences[i][j] for i in 1:batchsize] for j in 1:n ])
            #empty!(nsequences)
        #end
    #end
    #return data
#end
batchsize, statesize, vocabsize = 2, 100, length(int2tok)
data = minibatch(sequences,batchsize);
model = initmodel(statesize,vocabsize);

function avgloss(model, data)
    sumloss = cntloss = 0
    for sequence in data
        tokens = (1 + length(sequence)) * length(sequence[1])
        sumloss += s2s(model, sequence, sequence)
        cntloss += tokens
    end
    return sumloss/cntloss
end

#avgloss(model,data)
#exp(ans)

s2sgrad = grad(s2s)

function train(model, data, opts)
    for i = 1:2:length(data)-1
        grads = s2sgrad(model, data[i], data[i+1])
        update!(model, grads, opts)
    end
end

oparams{T<:Number}(::KnetArray{T}; o...)=Adam()
oparams{T<:Number}(::Array{T}; o...)=Adam()
oparams(a::Associative; o...)=Dict(k=>oparams(v) for (k,v) in a)
oparams(a; o...)=map(x->oparams(x;o...), a)

opts = oparams(model);
#train(model,data,opts)
#avgloss(model,data)


function translate(model, str)
    state = model[:state0]
    for c in reverse(split(str))
        input = onehotrows(tok2int[c], model[:embed1])
        input = input * model[:embed1]
        state = lstm(model[:encode], state, input)
    end
    input = eosmatrix(1, model[:embed2]) * model[:embed2]
    output = String[]
    for i=1:100 #while true                                                                                                
        state = lstm(model[:decode], state, input)
        pred = predict(model[:output], state[1])
        i = indmax(Array(pred))
        i == 1 && break
        push!(output, int2tok[i])
        input = onehotrows(i, model[:embed2]) * model[:embed2]
    end
    o = String("")
    for w in output
        o = o * w
        o = o * " "
    end
    return o
end


for epoch=1:2
    train(model,data[epoch],opts)
    #println((epoch,avgloss(model,data)))
end