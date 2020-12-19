### A Pluto.jl notebook ###
# v0.12.17

using Markdown
using InteractiveUtils

# ╔═╡ 102fce2e-13b9-11eb-0da5-ab502c3ea430
begin 
	import Pkg
	Pkg.activate("..")

	using AbstractPlotting.MakieLayout
	using WGLMakie,AbstractPlotting
	AbstractPlotting.inline!(true)
	
	using JSServe,JSServe.DOM
	using LaTeXStrings

	md"""## Optimisation of Likelihood Functions
	We begin by defining the probability density that will model our data
	``p(\mathbf{x}|\mathbf{\theta})``. Our data ``\mathcal{D} = \{ X_1 \dots X_k \}`` 
	is composed of ``N``-dimensional binary vectors
	``X \in \{0,1\}^N`` whos components are either zero or one.
	In our code we will use Boolean components to restrict the input space. 
	Parameters ``\theta\in[0,1]^N`` are restricted to the unit 
	interval to ensure the normalisation of ``p(\mathbf{x}|\mathbf{\theta})`` with 
	respect to  ``\mathbf{x}``
	"""
end

# ╔═╡ 13578900-13d9-11eb-183a-793180b48614
θrange = range(0,1,length=50)

# ╔═╡ d44a80be-13d6-11eb-2a8e-2da04ecfbc01
L"""
p(\mathbf{x}|\mathbf{\theta}):=
	\prod_{n=1}^N \theta_{n}^{\,x_n} (1-\theta_n) ^ {1-x_n}
"""

# ╔═╡ 78229810-13b7-11eb-292a-e78eec461658
function model(x::Vector{T},θ::Vector) where T<:Bool
	N = length(x)
	return prod([ θ[n]^(x[n]) * (1-θ[n])^(1-x[n]) for n ∈ 1:N ])
end

# ╔═╡ 21725e0e-13c8-11eb-0b33-51f1af23bf7e
md"""
Suppose we look at a single observation ``X``. If we substitute it into our model
``p(\mathbf{x}=X|\theta)`` we could get a high probability for that data point or a 
low one. We would like to choose a ``\theta`` that overlaps the shape of
``p(\mathbf{x}=X|\theta)`` with the observed ``X``. This means maximising the probability ``p(\mathbf{x}=X|\theta)``. This is known as the likelihood
"""

# ╔═╡ 03c3f582-13ca-11eb-12a8-a979cfa5395b
L"L(X,\theta):=p(\mathbf{x}=X|\theta)"

# ╔═╡ f26688d8-13cc-11eb-391c-efc74ed78b8c
md"""
Below are examples of likelihoods ``L(X,\theta)``
for different single observations ``X\in\{0,1\}^2`` We can
see that the likelihoods for single observations are maximised
when ``\theta=X``
"""

# ╔═╡ 24eb8b64-13ca-11eb-0a3b-35fd73614563
begin
	
	scene_one, layout_one = layoutscene(resolution = (700,750))
	
	ax1 = layout_one[1,1] = LAxis(scene_one, ylabel="θ[2]", xlabel="θ[1]",
		title="X=(0,1)", panlock = true, zoomlock = true)
	
	heatmap!( ax1, θrange, θrange,
		[ model([false,true], [θ1,θ2]) for θ1 in θrange, θ2 in θrange],
		
		colormap=cgrad(:starrynight, 10, categorical=true),
		interpolate=true )
	
	ax2 = layout_one[1,2] = LAxis(scene_one, ylabel="θ[2]", xlabel="θ[1]",
		title="X=(1,1)", panlock = true, zoomlock = true)
	
	heatmap!( ax2, θrange, θrange,
		[ model([true,true], [θ1,θ2]) for θ1 in θrange, θ2 in θrange],
		
		colormap=cgrad(:starrynight, 10, categorical=true),
		interpolate=true )
	
	ax3 = layout_one[2,1] = LAxis(scene_one, ylabel="θ[2]", xlabel="θ[1]",
		title="X=(0,0)", panlock = true, zoomlock = true)
	
	heatmap!( ax3, θrange, θrange,
		[ model([false,false], [θ1,θ2]) for θ1 in θrange, θ2 in θrange],
		
		colormap=cgrad(:starrynight, 10, categorical=true),
		interpolate=true )
	
	ax4 = layout_one[2,2] = LAxis(scene_one, ylabel="θ[2]", xlabel="θ[1]",
		title="X=(1,0)", panlock = true, zoomlock = true)
	
	heatmap!( ax4, θrange, θrange,
		[ model([true,false], [θ1,θ2]) for θ1 in θrange, θ2 in θrange],
		
		colormap=cgrad(:starrynight, 10, categorical=true),
		interpolate=true )
		
    RecordEvents( scene_one, "output" )
    scene_one
end

# ╔═╡ cad146a0-13c7-11eb-03c0-1b75b17f3cb4
md"""
Assuming our dataset ``\mathcal{D}`` is independent and identically distributed we can write down the total likelihood as the product of individual datum likelihoods
"""

# ╔═╡ f63a953a-13d6-11eb-3dd0-2d04a3c8e022
L"L(\mathbf{\theta}):=\prod_{X\in\mathcal{D}}L(X,\mathbf{\theta})"

# ╔═╡ fc9ad2ac-13b6-11eb-3bad-ab10c89d252e
function likelihood(p::Vector,D::Vector{<:Vector})
	return prod([ model(X,p) for X ∈ D ])
end

# ╔═╡ 167e02c8-13cd-11eb-22d5-8faa2abf7a53
md"""
For a randomly generated dataset ``\mathcal{D}`` this product leads to a unique maximum in the likelihood. In this case -- but not in most cases -- we can derive the optimum analytically by differentiating the likelihood with respect to ``\theta`` and setting to zero
"""

# ╔═╡ ee80ff52-13cf-11eb-01d6-3baf3ea94c6f
L"""
\frac{\partial L(\theta)}{\partial\theta} = 0
"""

# ╔═╡ 148242fe-13d0-11eb-3d0c-919ab17f3cee
md"""
But since its easier to differentiate sums than products and taking a log does not change the position of the optimum, we would rather differentiate the log likelihood
``\log L(\theta)``. Taking one component ``\theta_k`` at a time
"""

# ╔═╡ 5737a2f0-13d0-11eb-3ffc-c504202b96c4
L"
\frac{\partial}{\partial\theta_k}\log L(\theta) = 
\frac{\partial}{\partial\theta_k}
\sum_{X\in\mathcal{D}}\sum_{n=1}^N 
X_n\log\theta_n+(1-X_n)\log(1-\theta_n)
"

# ╔═╡ e3e22f7c-13d0-11eb-26f6-4d9dc401d7ad
L"= 
\sum_{X\in\mathcal{D}}\sum_{n=1}^N 
\frac{X_n}{\theta_n}\delta_{nk}-\frac{1-X_n}{1-\theta_n}\delta_{nk}
"

# ╔═╡ 7e4bd566-13d1-11eb-0e03-391e1d95e197
L"= 
\sum_{X\in\mathcal{D}}
\frac{X_k}{\theta_k}-\frac{1-X_k}{1-\theta_k}
"

# ╔═╡ ff7045ae-13d2-11eb-3a81-2b9f085524f8
L"= 
\frac{\sum_{X\in\mathcal{D}}X_k}{\theta_k}-\frac{|\mathcal{D}|-\sum_{X\in\mathcal{D}}X_k}{1-\theta_k}
"

# ╔═╡ 4b0a5d60-13d3-11eb-01be-35077fe378c2
md"Setting `` \frac{\partial}{\partial\theta_k}\log L(\theta)|_{\theta=\theta^*} = 0``  for the optimal ``\theta^*`` and dividing through by dataset size ``|\mathcal{D}|`` yields
"

# ╔═╡ 5b04663c-13d3-11eb-289a-05a85e5206d4
L"
\frac{\frac{1}{|\mathcal{D}|}\sum_{X\in\mathcal{D}}X_k}{\theta_k^*}=\frac{1-\frac{1}{|\mathcal{D}|}\sum_{X\in\mathcal{D}}X_k}{1-\theta_k^*}
"

# ╔═╡ 33d36578-13d4-11eb-28ae-fd8cb25224cb
md"
Which suggests that `` \theta_k^*=\frac{1}{|\mathcal{D}|}\sum_{X\in\mathcal{D}}X_k `` We verify this below -- again for the two dimensional case -- by evaluating the likelihood ``L(\theta)`` for a sample dataset ``\mathcal{D}`` generated by randomly sampling the space of vectors ``X\in\{0,1\}^2``
"

# ╔═╡ 38b34190-1450-11eb-1c4d-c738d3feed5e
md"###### Number of Data Points
Drag the slider to increase the number of data points. Note how the uncertainty
around the maximum likelihood parameters decreases as more data is used
"

# ╔═╡ 0fe7a234-144f-11eb-281f-c948380d945c
begin
	nPoints = JSServe.Slider(5:50)
	JSServe.with_session() do s, r
		return DOM.div(nPoints, nPoints.value)
	end
end

# ╔═╡ ac667340-13bb-11eb-3bbb-2d7875c3598f
begin
	scene, layout = layoutscene(resolution = (500,550))
	ax = layout[1,1] = LAxis(scene, panlock = true, zoomlock = true,
		ylabel="θ[2]", xlabel="θ[1]", title="L(θ)")
	
	data = @lift([ rand([true,false],2) for _ ∈ 1:$nPoints ])
	θstar = @lift( transpose( sum($data) / length($data) ) )
	
	heatmap!( ax, θrange, θrange,
		@lift([ likelihood([x,y],$data) for x in θrange, y in θrange]),

		colormap=cgrad(:starrynight, 10, categorical=true),
		interpolate=true )

	scatter!( ax, θstar, color=:white, marker=:diamond)
    RecordEvents( scene, "output" )
    scene
end

# ╔═╡ Cell order:
# ╟─102fce2e-13b9-11eb-0da5-ab502c3ea430
# ╠═13578900-13d9-11eb-183a-793180b48614
# ╟─d44a80be-13d6-11eb-2a8e-2da04ecfbc01
# ╠═78229810-13b7-11eb-292a-e78eec461658
# ╟─21725e0e-13c8-11eb-0b33-51f1af23bf7e
# ╟─03c3f582-13ca-11eb-12a8-a979cfa5395b
# ╟─f26688d8-13cc-11eb-391c-efc74ed78b8c
# ╟─24eb8b64-13ca-11eb-0a3b-35fd73614563
# ╟─cad146a0-13c7-11eb-03c0-1b75b17f3cb4
# ╟─f63a953a-13d6-11eb-3dd0-2d04a3c8e022
# ╠═fc9ad2ac-13b6-11eb-3bad-ab10c89d252e
# ╟─167e02c8-13cd-11eb-22d5-8faa2abf7a53
# ╟─ee80ff52-13cf-11eb-01d6-3baf3ea94c6f
# ╟─148242fe-13d0-11eb-3d0c-919ab17f3cee
# ╟─5737a2f0-13d0-11eb-3ffc-c504202b96c4
# ╟─e3e22f7c-13d0-11eb-26f6-4d9dc401d7ad
# ╟─7e4bd566-13d1-11eb-0e03-391e1d95e197
# ╟─ff7045ae-13d2-11eb-3a81-2b9f085524f8
# ╟─4b0a5d60-13d3-11eb-01be-35077fe378c2
# ╟─5b04663c-13d3-11eb-289a-05a85e5206d4
# ╟─33d36578-13d4-11eb-28ae-fd8cb25224cb
# ╟─38b34190-1450-11eb-1c4d-c738d3feed5e
# ╟─0fe7a234-144f-11eb-281f-c948380d945c
# ╟─ac667340-13bb-11eb-3bbb-2d7875c3598f
