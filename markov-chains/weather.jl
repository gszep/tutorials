using CSV,DataFrames
data = CSV.File("markov-chains/meteo0.csv",header=false) |> DataFrame
data = convert(Matrix,data)

begin
    transitions = zeros(3,3)
    function estimate_transition!(n,m)

        for i in 1:500
            for j in 1:100-1
                if (data[i,j] == n-1) & (data[i,j+1] == m-1)
                    transitions[n,m] += 1
                end
            end
        end
        return transitions
    end

    for (n,m) ∈ Iterators.product(1:3,1:3)
        estimate_transition!(n,m)
    end

    transitions ./= sum(transitions,dims=1)
end

cloudy = 0.0
for i in 1:500
    for j in 1:100-1
        if (data[i,j] == 2)
            cloudy += 1
        end
    end
end

println(cloudy/sum(prod(size(data))))

p₀ = [0,0,1]
p(n) = transitions^n * p₀

plot()
p₀ = [1,0,0]
for i ∈ 1:3
    plot!(1:25,n->p(n)[i])
end
plot!() |> display


data = CSV.File("markov-chains/meteo1.csv",header=false) |> DataFrame
data = convert(Matrix,data)

heatmap(data)