using Knet, JLD

function createTestSequences(testPath, tok2int)
    testSequences = Vector{Vector{Int}}()
    testFiles = readdir(testPath)
    for f in testFiles
        strings = map(chomp,readlines(string(testPath,f)))
        words = map(split, strings)
        for w in words
            s = Vector{Int}()
            for c in w
                if !haskey(tok2int,c)
                    push!(s,tok2int["unk"])
                else
                    push!(s,tok2int[c])
                end

            end
            push!(testSequences,s)
        end


    end
    return testSequences

end

function cleanSequences(path, sequences)
    sequences = Vector{Vector{Int}}()
    files = readdir(path)
    for f in files
        strings = map(chomp,readlines(string(path,f)))
        words = map(split, strings)
        for w in words
            s = Vector{Int}()
            for c in w
                if !haskey(tok2intFreq,c)
                    push!(s,tok2intFreq["unk"])
                else
                    push!(s,tok2intFreq[c])
                end
            end
            push!(sequences,s)
        end


    end
    return sequences

end
function createDict(file, tok2int, int2tok)
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
                freqCount[tok2int[c]]=0
            else

                freqCount[tok2int[c]]=freqCount[tok2int[c]]+1
            end
            
                    
    end
    end
    return tok2int, int2tok, freqCount
end


function createDictDir(path,clamp=30)    
    files = readdir(path)
    global tok2int = Dict{String,Int}()
    global int2tok = Vector{String}()
    global tok2intFreq = Dict{String,Int}()
    global int2tokFreq = Vector{String}()
    push!(int2tokFreq,"\n"); tok2intFreq["\n"]=1
    global freqCount = Dict{Int,Int}()
    sequences = Vector{Vector{Int}}()
    count =0;
    for f in files
        #update the vocabulary by adding new words from each file

        tok2int,int2tok,freqCount = createDict(string(path,f),tok2int,int2tok)
    end
    for word in freqCount
        if(word[2]>clamp)
          push!(int2tokFreq,int2tok[word[1]])
          tok2intFreq[int2tok[word[1]]]=length(int2tokFreq)
        end

    end
    push!(int2tokFreq,"unk")
    tok2intFreq["unk"]=length(int2tokFreq)

    sequences = cleanSequences(path, sequences)

    return tok2intFreq, int2tokFreq, sequences, freqCount
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
    model[:state0] = [ init(1,H), init(1,H), init(1,H), init(1,H) ]
    model[:embed1] = init(V,H)
    model[:encode] = [ init(2H,4H), init(1,4H) ]
    model[:encodeH] = [ init(2H,4H), init(1,4H) ]
    
    model[:embed2] = init(V,H)
    model[:decodeH] = [ init(2H,4H), init(1,4H) ]
    model[:decode] = [ init(2H,4H), init(1,4H) ]
    #P = div(H,2)
    #model[:project] = init(H, P)
    model[:output] = [ init(H,V), init(1,V) ]
    return model
end

function s2s(model, inputs, outputs)
    state = initstate(inputs[1], model[:state0])

    for input in reverse(inputs)
        input = onehotrows(input, model[:embed1])
        input = input * model[:embed1]
        state[1] = lstm(model[:encodeH], state[1], input)
        input = state[1][1]
        state[2] = lstm(model[:encode], state[2], input)
    end
    EOS = eosmatrix(outputs[1], model[:embed2])
    input = EOS * model[:embed2]
    sumlogp = 0
    for output in outputs
        state[1] = lstm(model[:decodeH], state[1], input)
        input = state[1][1]
        state[2] = lstm(model[:decode], state[2], input)
        #ypred = predict(model[:project], model[:output], state[2][1])
        ypred = predict(model[:output], state[2][1])
        ygold = onehotrows(output, model[:embed2])
        sumlogp += sum(ygold .* logp(ypred,2))
        input = ygold * model[:embed2]
    end
    state[1] = lstm(model[:decodeH], state[1], input)
    input = state[1][1]
    state[2] = lstm(model[:decode], state[2], input)
    ypred = predict(model[:output], state[2][1])
    sumlogp += sum(EOS .* logp(ypred,2))
    return -sumlogp
end

function predict(out, input)
    #input = input * project
    return input * out[1] .+ out[2]
end


function initstate(idx, state0)
    hidden = 2;
    state = []
    h,c, h2, c2 = state0
    h = h .+ fill!(similar(AutoGrad.getval(h), length(idx), length(h)), 0)
    c = c .+ fill!(similar(AutoGrad.getval(c), length(idx), length(c)), 0)
    h2 = h2 .+ fill!(similar(AutoGrad.getval(h2), length(idx), length(h2)), 0)
    c2 = c2 .+ fill!(similar(AutoGrad.getval(c2), length(idx), length(c2)), 0)
    push!(state,(h,c))
    push!(state,(h2,c2))
    return state
end

function onehotrows(idx, embeddings)
    nrows,ncols = length(idx), size(embeddings,1)
    z = zeros(Float32,nrows,ncols)
    for i=1:nrows
        if(idx[i]!=0 && idx[i]<=ncols)
        z[i,idx[i]] = 1
        end
    end
    oftype(AutoGrad.getval(embeddings),z)
end

function getIndex(idx, embeddings)
    batch, vocab = length(idx), size(embeddings,1)
    max = size(embeddings,2)+1
    index = zeros(Float32,batch)
    for i=1:batch

        if(idx[i]!=0)
            index[i]=idx[i]
        else
            index[i] = max
        end
    end
    oftype(AutoGrad.getval(embeddings),index)
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
        question = map(d->KnetArray(d),question);
        answer = map(d->KnetArray(d),answer);
        tokens = (1 + length(answer)) * length(answer[1])
        sumloss += s2s(model, question, answer)
        cntloss += tokens
    end
    return sumloss/cntloss
end

function respond(model, str)
    h,c,h2,c2=model[:state0]
    state=[]
    push!(state,(h,c))
    push!(state,(h2,c2))
    for c in split(str)
        input = onehotrows(tok2int[c], model[:embed1])
        input = input * model[:embed1]
        state[1] = lstm(model[:encodeH], state[1], input)
        input = state[1][1]
        state[2] = lstm(model[:encode], state[2], input)
    end
    input = eosmatrix(1, model[:embed2]) * model[:embed2]
    output = String[]
    for i=1:100 #while true 
        state[1] = lstm(model[:decodeH], state[1], input)
        input = state[1][1]
        state[2] = lstm(model[:decode], state[2], input)
        pred = predict(model[:output], state[2][1])
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
    testepochsloss = Dict{Int,Float32}()
    for i=1:epochs
        for (question, answer) in zip(x,ygold)
        #a batch of questions and answers
            

         question = map(d->KnetArray(d),question)
         answer = map(d->KnetArray(d),answer)
         grads = s2sgrad(model, question, answer)
 
         update!(model, grads, opts)
            end
        
            currloss=avgloss(model, x, ygold);
            epochsloss[i]=currloss;
            testloss = avgloss(model,xtest,ytest);
            testepochsloss[i]=testloss;
            save("epochloss_$i.jld","epochloss",currloss);
            save("testloss_$i.jld","testloss",testloss);
            floatmodel=Dict{Symbol,Any}()
            floatmodel[:state0]= map(a->convert(Array{Float32},a),model[:state0]);
            floatmodel[:embed1] = convert(Array{Float32},model[:embed1]);
            floatmodel[:encode] = map(a->convert(Array{Float32},a),model[:encode]);
            floatmodel[:encodeH] = map(a->convert(Array{Float32},a),model[:encodeH]);
            floatmodel[:embed2] = convert(Array{Float32},model[:embed2]);
            floatmodel[:decode] = map(a->convert(Array{Float32},a),model[:decode]);
            floatmodel[:decodeH] = map(a->convert(Array{Float32},a),model[:decodeH]);
             #floatmodel[:project] = convert(Array{Float32},model[:project]);
            floatmodel[:output] = map(a->convert(Array{Float32},a),model[:output]);
            save("model_$i.jld", "model", floatmodel);
    end
    save("epochloss_total.jld", "epochsloss", epochsloss);
    save("testepochloss_total.jld", "testepochsloss", testepochsloss);    
end

s2sgrad = grad(s2s)


function main()
    
    global model, x, ygold, xtest, ytest
    tok2int, int2tok, sequences = createDictDir("data/train/",30);
    testSequences = createTestSequences("data/test/", tok2int);
    longest = 20;
    save("vocab.jld", "int2tok", int2tok);
    save("vocab2int.jld", "tok2int", tok2int);    
    batchsize, statesize, vocabsize = 100, 1024, length(int2tok)
    x, ygold = minibatch(sequences,batchsize,longest);
    xtest, ytest = minibatch(testSequences,batchsize,longest);
    model = initmodel(statesize,vocabsize);

    
    oparams{T<:Number}(::KnetArray{T}; o...)=Adam()
    oparams{T<:Number}(::Array{T}; o...)=Adam()
    oparams(a::Associative; o...)=Dict(k=>oparams(v;o...) for (k,v) in a)
    oparams(a; o...)=map(x->oparams(x;o...), a)

    opts = oparams(model);

    train(model,x,ygold,opts,10);
    
end

main()