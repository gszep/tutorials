### A Pluto.jl notebook ###
# v0.12.17

using Markdown
using InteractiveUtils

# ╔═╡ 102fce2e-13b9-11eb-0da5-ab502c3ea430
begin 
	import Pkg
	Pkg.activate("..")
	Pkg.instantiate()
	
	using LaTeXStrings,Parameters
	using LinearAlgebra
	using Zygote
	
	using WGLMakie,AbstractPlotting
	AbstractPlotting.inline!(true)
	using AbstractPlotting.MakieLayout
end

# ╔═╡ 9114bea2-4373-11eb-3780-53292d278b2d
	md"""## Expectation Maximisation
	We shall demonstrate the expectation-maximisation algorithm on the gaussian
	mixture model ``p(\mathbf{x}|\mathbf{\theta})`` where parameters we want to 
	estimate ``\mathbf{\theta}=
	(\Sigma_1\dots,\Sigma_K,\mu_1\dots\mu_K,\pi_1\dots\pi_K)`` are the 
	means ``\mu_k`` covariance matrices ``\Sigma_k`` and weight ``\pi_k`` for each mixture. The model 
	is written as
	"""

# ╔═╡ 9461a482-4060-11eb-2fd4-07e25338c86c
L"""
p(\mathbf{x}|\mathbf{\theta}):=
	\sum_{k=1}^K \pi_{k} \mathcal{N}(\mathbf{x}|\mu_k,\Sigma_k)
\quad\mathrm{where}\quad \sum_{k=1}^K\pi_k =1
"""

# ╔═╡ 97e8d860-410e-11eb-2fd5-8de2aee692b0
function 𝒩(x,μ,Σ) # multivariate gaussian
	return exp( -(x-μ)'inv(Σ)*(x-μ)/2 ) / √( (2π)^length(x)*abs(det(Σ)) )
end

# ╔═╡ a4935164-4062-11eb-3f47-0b92eaa966e9
md"""
We could just look for the maximum likelihood estimate. However this allows any optimiser to place mixture ``k`` right on top of a single data point and descrease the covariance ``\Sigma_k\rightarrow 0``. This leads to singularities in the likeliood and makes it very difficult to find the optimal parameters ``\theta^*``

Since optimising ``p(\mathbf{x}|\mathbf{\theta})`` directly is difficult we re-write the model in terms of ``K`` dimensional probability vectors ``\mathbf{z}``, its prior distirbution ``p(\mathbf{z}|\theta)`` and model ``p(\mathbf{x}|\mathbf{z},\theta)`` as
"""

# ╔═╡ 0b784f32-408c-11eb-06c4-f50a21ab46cb
L"
p(\mathbf{x}|\theta)=\sum_{\mathbf{z}}p(\mathbf{x}|\mathbf{z},\theta)p(\mathbf{z}|\theta)
"

# ╔═╡ 76e76e1e-4074-11eb-1a11-cfb9f827d4cd
L"""
p(\mathbf{x}|\mathbf{z},\mathbf{\theta}):=
	\prod_{k=1}^K \mathcal{N}(\mathbf{x}|\mu_k,\Sigma_k)^{z_k}
\qquad
p(\mathbf{z}|\theta):=\prod_{k=1}^K\pi_k^{z_k}
"""

# ╔═╡ 074ede5a-4076-11eb-3a0a-955ff06976d9
function model(x::Vector{<:Number},z::Vector{<:Bool},θ::NamedTuple)
	@assert(sum(z)==1,"z must be a one-hot vector")
	@unpack Σ,μ = θ
	
	k = findfirst(z) # index of one-hot element
	return 𝒩(x,μ[k],Σ[k])
end

# ╔═╡ a39c0bdc-4073-11eb-086e-f592bb72706b
function prior(z::Vector{<:Bool},θ::NamedTuple)
	@assert(sum(z)==1,"z must be a one-hot vector")
	@unpack π = θ

	k = findfirst(z) # index of one-hot element
	return π[k]
end

# ╔═╡ a091f3c8-408e-11eb-1d47-6b9b5044d660
md"""From the model and prior we can determine the joint distribution ``p(\mathbf{x},\mathbf{z}|\theta)`` and posterior ``p(\mathbf{z}|\mathbf{x},\theta)``
"""

# ╔═╡ f1ec823e-4114-11eb-165b-55e6d09960e4
L"p(\mathbf{z}|\mathbf{x},\theta)=\frac{
p(\mathbf{x}|\mathbf{z},\theta)p(\mathbf{z}|\theta)}{
p(\mathbf{x}|\mathbf{\theta})}"

# ╔═╡ 32c711cc-4078-11eb-2b47-8f9f2d6d256a
function joint(x::Vector{<:Number},z::Vector{<:Bool},θ::NamedTuple)
	return model(x,z,θ)*prior(z,θ)
end

# ╔═╡ 80e0b886-4078-11eb-2c90-ebfb374de47b
md"""Given a datasets ``\mathbf{X} = ( \mathbf{x}_1 \dots \mathbf{x}_D )`` and ``\mathbf{Z} = ( \mathbf{z}_1 \dots \mathbf{z}_D )`` we can calculate likelihoods for any distribution. This is because  the data are assumed independent and identically distributed, allowing the likelihoods factorise with respect to data. For example for the posterior likelihood
"""

# ╔═╡ 997132d0-408a-11eb-22e0-976c2df8ca0c
L"p(\mathbf{Z}|\mathbf{X},\theta):=\prod_{\mathbf{x}\in\mathbf{X},\mathbf{z}\in\mathbf{Z}}p(\mathbf{z}|\mathbf{x},\theta)"

# ╔═╡ d497c0f0-4115-11eb-1f01-df9a4d33dd08
L"=\mathrm{exp}\left( \sum_{\mathbf{x}\in\mathbf{X},\mathbf{z}\in\mathbf{Z}}\ln p(\mathbf{z}|\mathbf{x},\theta)\right)"

# ╔═╡ 0f49f918-4090-11eb-3865-1771ff1ecb70
function joint(X::Vector{<:Vector},Z::Vector{<:Vector},θ::NamedTuple)
	return exp( sum( log.( joint.(X,Z,Ref(θ)) ) ) )
end

# ╔═╡ c3b2b190-410f-11eb-1c38-ebbea30083d8
function marginal(x::Vector{<:Number},z::Vector{<:Bool},θ::NamedTuple)
	return sum( k->joint(x, circshift(z,k) ,θ), 1:length(z) ) # over all possible z
end

# ╔═╡ 786906c2-4078-11eb-1a98-77134f10075d
function posterior(x::Vector{<:Number},z::Vector{<:Bool},θ::NamedTuple)
	return joint(x,z,θ) / marginal(x,z,θ)
end

# ╔═╡ b159cec8-408a-11eb-3728-0d266078dfa3
function posterior(X::Vector{<:Vector},Z::Vector{<:Vector},θ::NamedTuple)
	return exp( sum( log.( posterior.(X,Z,Ref(θ)) ) ) )
end

# ╔═╡ ee8ca6b6-4121-11eb-2685-1bc42b156926
begin
	using CSV ##################################### data import
	data = CSV.File("data/old-faithful.tsv",type=Float64)
	data.waiting .= data.waiting/10.0
	
	xs = range(extrema(data.eruptions)..., length=100)
	ys = range(extrema(data.waiting)..., length=100)
	
	###################################################### scene construction
	scene, layout = layoutscene(resolution = (500,500))
	ax = layout[1,1] = LAxis(scene,
		
		xpanlock = true, xzoomlock = true,
		ypanlock = true, yzoomlock = true,
		
		xlabel="Eruptions", ylabel="Waiting")

	empty!(scene.events.mousedrag.listeners)
	mouseevents = addmouseevents!(ax.scene)
	
	######################################################### model parameters
	μ₁,μ₂= Node([4.0,5.0]), Node([2.5,8.0])
	θ = @lift((
		μ=[ $μ₁, $μ₂ ], π=[ 1/2, 1/2 ],
		Σ=[ [1/3 0; 0 1/3], [1/3 0; 0 1/3] ],
	))
	
	############################################################ marginal
	density = @lift([ marginal([x,y],[false,true],$θ) for x in xs, y in ys])
	heatmap!( ax, xs, ys, density,

		colormap=cgrad(:starrynight, 10, categorical=true),
		interpolate=true )
	
	scatter!( ax, @lift(transpose($μ₁)), marker="1", color=:white )
	scatter!( ax, @lift(transpose($μ₂)), marker="2", color=:white )
	
	############################################################# posterior
	scatter!( ax, map( (x,y) -> Point(x,y), data.eruptions, data.waiting),
		markersize=@lift( map( (x,y) -> 5posterior([x,y],[false,true],$θ), 
				data.eruptions, data.waiting )), color=:gold )
	
	scatter!( ax, map( (x,y) -> Point(x,y), data.eruptions, data.waiting),
		markersize=@lift( map( (x,y) -> 5posterior([x,y],[true,false],$θ), 
				data.eruptions, data.waiting )), color=:white )
	
	###################################################### bind input events	
	on(scene.events.keyboardbuttons) do button
		
		if ispressed(button, Keyboard._1)
			μ₁[] = mouseevents.obs.val.data
			
		elseif ispressed(button, Keyboard._2)
			μ₂[] = mouseevents.obs.val.data
		end
	end
	
    RecordEvents( scene, "output" )
    scene
end

# ╔═╡ 8a4a4e48-431c-11eb-07ff-09c8b60bd020
md"""
However in practice we are not given dataset ``\mathbf{Z}`` so the best thing we 
can do estimate what the probability of observing ``\mathbf{z}`` is given that we 
know ``\mathbf{x}``. This is precisely what we do in the expectation step, when 
we evaluate the posterior likelihood ``p(\mathbf{z}|\mathbf{x},\theta)``
"""

# ╔═╡ 01ac872a-4113-11eb-1ffa-f3087641bcf6
md"""
Let's look at this posterior for the old faithful dataset which has dimensions 
``N=2`` and seems to have ``K=2`` mixtures. We shall initialise a random 
``\theta`` and plot the contours of gaussian mixtures ``p(\mathbf{x}|\theta)``. 
**You can change the locations of the two mixtures my clicking on the plot, then 
press 1 or 2 on your keyboard when you would like to place the first or second 
mixture.**
"""

# ╔═╡ 2d939648-431c-11eb-16c4-136bd9e12d8e
md"""
We shall visualise the likelihood of each data point ``p(\mathbf{z}|\mathbf{x},\theta)`` as a colour: white if the probability of the first mixture ``\mathbf{z}=(1,0)`` is greater than the probability of the second ``\mathbf{z}=(0,1)`` and gold otherwise.
"""

# ╔═╡ cbc6114a-4314-11eb-1593-f588436f287c
md"""
We can see that the closer a data point ``\mathbf{x}`` is to mixture ``k``, the higher the probability ``p(\mathbf{z}|\mathbf{x},\theta)`` such that ``\mathbf{z}`` is a one-hot vector where the ``k``-th component is non-zero. How do we design an iterative proceedure to bring the mixtures to overlap with the high data density regions?

The evaluation of posterior likelihood ``p(\mathbf{z}|\mathbf{x},\theta)`` is known as the **expectation** step. What mixture do we expect each data point to belong to? In general, what do we expect the values of our latent variables ``\mathbf{z}`` to be given that we have ``\mathbf{x}``?

The **maximisation** step involves maximising the joint log likelihood ``\ln p(\mathbf{x},\mathbf{z}|\theta')`` as we would in the usual maximum likelihood approach, but here we take into account the expected values ``\mathbf{z}``. This can be done by averaging over ``\mathbf{Z}`` and since we have ``p(\mathbf{z}|\mathbf{x},\theta)`` we can write this as
"""

# ╔═╡ 95544938-4319-11eb-339b-e9349790aea3
L"
Q(\theta,\theta'):=\sum_{\mathbf{z}} p(\mathbf{z}|\mathbf{x},\theta) \ln p(\mathbf{x},\mathbf{z}|\theta')
"

# ╔═╡ 9009b516-431a-11eb-17ce-998e61d11549
function Q(θ::NamedTuple,θ′::NamedTuple,x::Vector{<:Number})
	Z = [[true,false],[false,true]] # specific for two mixtures
	return sum( z->posterior(x,z,θ)*log(joint(x,z,θ′)), Z )
end

# ╔═╡ aa8e98e2-4320-11eb-267d-ddd980a07d3f
md"
Since likelihoods factorise with respect whole datasets ``\mathbf{X}``, leading to sums in log likelihoods, the function ``Q(\theta,\theta')`` can be evaluated as a sum over the whole dataset
"

# ╔═╡ efb77868-431f-11eb-2173-8bb0a2cf97c9
function Q(θ::NamedTuple,θ′::NamedTuple,X::Vector{<:Vector})
	return sum( x->Q(θ,θ′,x), X )
end

# ╔═╡ e33547b0-4319-11eb-1f54-ff0d5101b85c
md"
We would maximise this function with respect to an unknown ``\theta'`` keeping the initial ``\theta`` we used for the expectation step fixed. Once the maximum has been found for some optimal ``\theta'`` we can update the parameter ``\theta\leftarrow\theta'`` and repeat the proceedure.
"

# ╔═╡ db2cd3b0-431b-11eb-0635-0d71b8e3ab29
begin
	
	############################# initialise dataset
	X = map( (x,y)->[x,y], data.eruptions, data.waiting )
	
	############################# initial parameters
	θₜ = ( μ=[ [4.0,5.0], [2.5,8.0] ], π=[ 1/2, 1/2 ],
		   Σ=[ [1/3 0; 0 1/3], [1/3 0; 0 1/3] ] )
	
	η = 10^-3 ###### learning rate
	for _ ∈ 1:40 ### algorithm iterations

		######################## expectaction-maximisation
		∂θ, = gradient( θ -> Q(θₜ,θ,X), θₜ )

		###################### update parameters
		θₜ.μ .= θₜ.μ + η*∂θ.μ
		θₜ.Σ .= θₜ.Σ + η*∂θ.Σ
		θₜ.π .= θₜ.π + η*∂θ.π

		###################### enforce constraints
		θₜ.π ./= sum(θₜ.π)
	end
end

# ╔═╡ 49de5dc2-4328-11eb-3352-57e88bddb934
begin
	###################################################### scene construction
	scene₁, layout₁ = layoutscene(resolution = (500,500))
	ax₁ = layout₁[1,1] = LAxis(scene₁,
		
		xpanlock = true, xzoomlock = true,
		ypanlock = true, yzoomlock = true,
		
		xlabel="Eruptions", ylabel="Waiting")
	empty!(scene₁.events.mousedrag.listeners)
	
	############################################################ marginal
	heatmap!( ax₁, xs, ys, [ marginal([x,y],[false,true],θₜ) for x in xs, y in ys],
		colormap=cgrad(:starrynight, 10, categorical=true),
		interpolate=true )
	
	############################################################# posterior
	scatter!( ax₁, map( (x,y) -> Point(x,y), data.eruptions, data.waiting),
		markersize= map( (x,y) -> 5posterior([x,y],[false,true],θₜ), 
				data.eruptions, data.waiting ), color=:gold )
	
	scatter!( ax₁, map( (x,y) -> Point(x,y), data.eruptions, data.waiting),
		markersize= map( (x,y) -> 5posterior([x,y],[true,false],θₜ), 
				data.eruptions, data.waiting ), color=:white )
	
    RecordEvents( scene₁, "output" )
    scene₁
end

# ╔═╡ e88e67b6-4317-11eb-1a9f-3b7864692294
md"""
In fact ``K``-means would suffice for this task so why are we bothing with this more complicated approach? Rather than assigning each data point to a cluster centroid like in  ``K``-means, each data point has a probability of belonging to a mixture ``p(\mathbf{z}|\mathbf{x},\theta)`` . 

Furthermore we are not limited the gaussian mixtures and can consider more complicated distributions ``p(\mathbf{x}|\mathbf{z},\theta)``. Finally we can consider the problem of latent variables in general, where ``\mathbf{z}`` is not just a one-hot encoding of a mixture.
"""

# ╔═╡ Cell order:
# ╟─102fce2e-13b9-11eb-0da5-ab502c3ea430
# ╟─9114bea2-4373-11eb-3780-53292d278b2d
# ╟─9461a482-4060-11eb-2fd4-07e25338c86c
# ╠═97e8d860-410e-11eb-2fd5-8de2aee692b0
# ╟─a4935164-4062-11eb-3f47-0b92eaa966e9
# ╟─0b784f32-408c-11eb-06c4-f50a21ab46cb
# ╟─76e76e1e-4074-11eb-1a11-cfb9f827d4cd
# ╠═074ede5a-4076-11eb-3a0a-955ff06976d9
# ╠═a39c0bdc-4073-11eb-086e-f592bb72706b
# ╟─a091f3c8-408e-11eb-1d47-6b9b5044d660
# ╟─f1ec823e-4114-11eb-165b-55e6d09960e4
# ╠═32c711cc-4078-11eb-2b47-8f9f2d6d256a
# ╠═c3b2b190-410f-11eb-1c38-ebbea30083d8
# ╠═786906c2-4078-11eb-1a98-77134f10075d
# ╟─80e0b886-4078-11eb-2c90-ebfb374de47b
# ╟─997132d0-408a-11eb-22e0-976c2df8ca0c
# ╟─d497c0f0-4115-11eb-1f01-df9a4d33dd08
# ╠═0f49f918-4090-11eb-3865-1771ff1ecb70
# ╠═b159cec8-408a-11eb-3728-0d266078dfa3
# ╟─8a4a4e48-431c-11eb-07ff-09c8b60bd020
# ╟─01ac872a-4113-11eb-1ffa-f3087641bcf6
# ╟─2d939648-431c-11eb-16c4-136bd9e12d8e
# ╟─ee8ca6b6-4121-11eb-2685-1bc42b156926
# ╟─cbc6114a-4314-11eb-1593-f588436f287c
# ╟─95544938-4319-11eb-339b-e9349790aea3
# ╠═9009b516-431a-11eb-17ce-998e61d11549
# ╟─aa8e98e2-4320-11eb-267d-ddd980a07d3f
# ╠═efb77868-431f-11eb-2173-8bb0a2cf97c9
# ╟─e33547b0-4319-11eb-1f54-ff0d5101b85c
# ╠═db2cd3b0-431b-11eb-0635-0d71b8e3ab29
# ╟─49de5dc2-4328-11eb-3352-57e88bddb934
# ╟─e88e67b6-4317-11eb-1a9f-3b7864692294
