using Knet, JLD
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

s2sgrad = grad(s2s)

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


function minibatch(sequences, batchsize, longest)
    tableX = Dict{Int,Vector{Vector{Int}}}()
    tableY = Dict{Int,Vector{Vector{Int}}}()
    
    dataX = Any[]
    dataY = Any[]
    questionlength = Dict{Int,Int}()
    
    data=Any[]
    for s = 1:2:length(sequences)-1
        #clamping long sequences
        n = length(sequences[s]) > longest ? longest : length(sequences[s])
        m = length(sequences[s+1])>longest ? longest : length(sequences[s+1]) 
        longer = n>m ? n : m
        nquestlength = get(questionlength, m, 0)
        if(nquestlength==0)
            questionlength[m]=m
            nquestlength=m
        end
        nsequencesX = get!(tableX,m, Any[])
        nsequencesY = get!(tableY,m, Any[])

        if(n>m && n>nquestlength)

            questionlength[m]=n
            for i = 1:length(nsequencesX)
                for k = nquestlength+1:n
                   #pad the othe questions with zeros in front since currently came the longest quest.
                    unshift!(nsequencesX[i],0)
              
                end
            end
            

        end

        if(m>n)
            
            for k = n+1:m
                #pad question with zeros in front
                unshift!(sequences[s],0)
            end
        end
        
        
        push!(nsequencesX, sequences[s])
        push!(nsequencesY, sequences[s+1])
        
        if length(nsequencesX) == batchsize
            push!(dataX, [[ nsequencesX[i][j] for i in 1:batchsize] for j in 1:m])
            push!(dataY, [[ nsequencesY[i][j] for i in 1:batchsize] for j in 1:m])
            empty!(nsequencesX)
            empty!(nsequencesY)
            questionlength[m]=0
           
        end

    end
    return dataX, dataY
end


function avgloss(model, x, ygold)
    sumloss = cntloss = 0
    for (question, answer) in zip(x,ygold)
        tokens = (1 + length(answer)) * length(answer[1])
        sumloss += s2s(model, question, answer)
        cntloss += tokens
    end
    return sumloss/cntloss
end

function respond(model, str)
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
    ans = String("")
    for w in output
        ans = ans * w
        ans = ans * " "
    end
    return ans
end

function train(model, x, ygold, opts, epochs)
    epochsloss = Dict{Int,Float32}()
    for i=1:epochs
        for (question, answer) in zip(x,ygold)
        #a batch of questions and answers
            convert(KnetArray{KnetArray{Int,1}}, question);
            convert(KnetArray{KnetArray{Int,1}}, answer)

            grads = s2sgrad(model, question, answer)
            update!(model, grads, opts)

            
        end
            currloss=avgloss(model, x, ygold);
            epochsloss[i]=currloss;

            floatmodel=Dict{Symbol,Any}()
            floatmodel[:state0]= map(a->convert(Array{Float32},a),model[:state0]);
            floatmodel[:embed1] = convert(Array{Float32},model[:embed1]);
            floatmodel[:encode] = map(a->convert(Array{Float32},a),model[:encode]);
            floatmodel[:embed2] = convert(Array{Float32},model[:embed2]);
            floatmodel[:decode] = map(a->convert(Array{Float32},a),model[:decode]);
            floatmodel[:output] = map(a->convert(Array{Float32},a),model[:output]);
            save("modelepoch$i.jld", "model", floatmodel);
    end
    save("epochloss.jld", "epochsloss", epochsloss);  
end



function main()
    global model, text, data, tok2int, o, opts

    
    tok2int, int2tok, sequences = createDictDir("data/train/");
    longest = 20;
    batchsize, statesize, vocabsize = 100, 1000, length(int2tok)
    x, ygold = minibatch(sequences,batchsize,longest);
    model = initmodel(statesize,vocabsize);

    oparams{T<:Number}(::KnetArray{T}; o...)=Adam()
    oparams{T<:Number}(::Array{T}; o...)=Adam()
    oparams(a::Associative; o...)=Dict(k=>oparams(v) for (k,v) in a)
    oparams(a; o...)=map(x->oparams(x;o...), a)

    opts = oparams(model);

    train(model,x,ygold,opts,10);
end

main()