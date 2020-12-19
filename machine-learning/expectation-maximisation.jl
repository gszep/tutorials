### A Pluto.jl notebook ###
# v0.12.17

using Markdown
using InteractiveUtils

# ╔═╡ 102fce2e-13b9-11eb-0da5-ab502c3ea430
begin 
	import Pkg
	Pkg.activate("..")
	
	using LaTeXStrings,Parameters
	using LinearAlgebra
	
	using AbstractPlotting.MakieLayout
	using WGLMakie,AbstractPlotting
	AbstractPlotting.inline!(true)

	md"""## Expectation Maximisation
	We shall demonstrate the expectation-maximisation algorithm on the gaussian
	mixture model ``p(\mathbf{x}|\mathbf{\theta})`` where parameters we want to 
	estimate ``\mathbf{\theta}=
	(\Sigma_1\dots,\Sigma_K,\mu_1\dots\mu_K,\pi_1\dots\pi_K)`` are the 
	means ``\mu_k`` covariance matrices ``\Sigma_k`` and weight ``\pi_k`` for each mixture. The model 
	is written as
	"""
end

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

# ╔═╡ ee8ca6b6-4121-11eb-2685-1bc42b156926
begin
	using CSV
	data = CSV.File("data/old-faithful.tsv")
	X = map( (x,y) -> [x,y], data.eruptions, data.waiting)
	html"""<body oncontextmenu="return false;">"""
	
	θ = (
		μ=[ [4.0,50.0], [2.5,80.0] ],
		Σ=[ 10*Diagonal([1/30,1]), 10*Diagonal([1/30,10]) ],
		π=[ 1/2, 1/2 ]
	)
	
	scene, layout = layoutscene(resolution = (500,500))
	ax = layout[1,1] = LScene(scene, camera=cam2d!,
		panlock = true, zoomlock = true,
		xlabel="Eruptions", ylabel="Waiting")
	
	xs = range(extrema(data.eruptions)..., length=100)
	ys = range(extrema(data.waiting)..., length=100)
	
# 	contour!( ax, xs, ys,
# 		[ log(posterior([x,y],[true,false],θ)) for x in xs, y in ys],
# 		label="" )
	
# 	contour!( ax, xs, ys,
# 		[ log(posterior([x,y],[false,true],θ)) for x in xs, y in ys],
# 		label="" )
	
	heatmap!( ax, xs, ys,
		[ marginal([x,y],[false,true],θ) for x in xs, y in ys],
		
		colormap=cgrad(:starrynight, 10, categorical=true),
		interpolate=true )
	
	scatter!( ax, map( (x,y) -> Point(x,y), data.eruptions, data.waiting),
		markersize=3, color=:gold)
	
	x = ax.scene.events.mouseposition
	scatter!( ax, @lift( Point($x)/500 ),
		markersize=20, color=:gold)
	
	controls = cameracontrols(ax.scene)
    RecordEvents( scene, "output" )
    scene
end

# ╔═╡ 786906c2-4078-11eb-1a98-77134f10075d
function posterior(x::Vector{<:Number},z::Vector{<:Bool},θ::NamedTuple)
	return joint(x,z,θ) / marginal(x,z,θ)
end

# ╔═╡ b159cec8-408a-11eb-3728-0d266078dfa3
function posterior(X::Vector{<:Vector},Z::Vector{<:Vector},θ::NamedTuple)
	return exp( sum( log.( posterior.(X,Z,Ref(θ)) ) ) )
end

# ╔═╡ 01ac872a-4113-11eb-1ffa-f3087641bcf6
md"""
However in practice we are not given dataset ``\mathbf{Z}`` so the best thing we can do estimate what the probability of observing ``\mathbf{z}`` is given that we know ``\mathbf{x}``. This is precisely what we do in the expectation step, when we evaluate the posterior likelihood ``p(\mathbf{Z}|\mathbf{X},\theta)``

Let's look at this posterior for the old faithful dataset which has dimensions ``N=2`` and seems to have ``K=2`` mixtures. We shall initialise a random ``\theta`` and plot the contours of marginal ``p(\mathbf{x}|\theta)``. We shall visualise the likelihood of each datum ``p(\mathbf{z}|\mathbf{x},\theta)`` as a colour
"""

# ╔═╡ bb4729f8-4222-11eb-3944-f932ae581b97
controls.zoombutton

# ╔═╡ Cell order:
# ╟─102fce2e-13b9-11eb-0da5-ab502c3ea430
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
# ╟─01ac872a-4113-11eb-1ffa-f3087641bcf6
# ╠═ee8ca6b6-4121-11eb-2685-1bc42b156926
# ╠═bb4729f8-4222-11eb-3944-f932ae581b97
