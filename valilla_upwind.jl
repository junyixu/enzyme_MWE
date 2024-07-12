#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#=
    a simple 1D FV upwind!,  with a uniform mesh and periodic boundaries.
    Artificially create a simple cache object holding intermediate values for the FV fluxes.
=#

using Enzyme

const C = 0.2 # C = Δt/Δx

struct Cache{T}
    v::T # velocity
    numerical_flux::Vector{T}
end


function upwind!(du::Vector, u::Vector, cache)
    cache.numerical_flux .= u .* cache.v
	for i = 2:length(u)
        du[i] = - C * (cache.numerical_flux[i] - cache.numerical_flux[i-1])  # Q_j^{n+1} = Q_j^n - Δt/Δx * ( F_{j+1/2}^n - F_{j-1/2}^n )
	end
    du[1] = - C * (cache.numerical_flux[1] - cache.numerical_flux[end])
end

function upwind!(du::Vector, u::Vector, v, numerical_flux)
    numerical_flux .= u * v
	for i = 2:length(u)
        du[i] = - C * (numerical_flux[i] - numerical_flux[i-1])  # Q_j^{n+1} = Q_j^n - Δt/Δx * ( F_{j+1/2}^n - F_{j-1/2}^n )
	end
    du[1] = - C * (numerical_flux[1] - numerical_flux[end])
    return nothing # important for reverse mode AD
end

function upwind!(du::Vector, u::Vector)
    v = 1.0
    numerical_flux = u * v
	for i = 2:length(u)
        du[i] = - C * (numerical_flux[i] - numerical_flux[i-1])  # Q_j^{n+1} = Q_j^n - Δt/Δx * ( F_{j+1/2}^n - F_{j-1/2}^n )
	end
    du[1] = - C * (numerical_flux[1] - numerical_flux[end])
    return nothing
end

function jacobian_ad_forward_enzyme_cache_upwind_right(x::AbstractVector)
    u_ode = zeros(length(x))
    du_ode = zeros(length(x))
    cache = (;v=1.0, numerical_flux=zeros(length(x)))

    dy = Tuple(zeros(size(du_ode)) for _ in 1:length(u_ode))
    dx = Enzyme.onehot(u_ode)
    cache_shadows =Tuple((;v=1.0, numerical_flux=zeros(length(x))) for i=1:length(x))

    # cache is passed to upwind!
    Enzyme.autodiff(Enzyme.Forward, upwind!, Enzyme.BatchDuplicated(du_ode, dy), Enzyme.BatchDuplicated(u_ode, dx), Enzyme.BatchDuplicated(cache, cache_shadows))
    return stack(dy)
end

x = -1.0:0.01:1.0
@time jacobian_ad_forward_enzyme_cache_upwind_right(x);


# %%

function gradients_ad_forward_enzyme_cache_upwind(x::AbstractVector)
    u_ode = zeros(length(x))
    du_ode = zeros(length(x))
    numerical_flux= zeros(length(x))

    dy = zeros(size(du_ode))
    dx = zeros(size(u_ode))
    numerical_flux_shadow= zeros(length(x))
    dys = zeros(length(du_ode), length(du_ode))

    for i in 1:length(x)
        dx[i] = 1.0
        # cache is passed to upwind!
        Enzyme.autodiff(Enzyme.Forward, upwind!, Enzyme.Duplicated(du_ode, dy), Enzyme.Duplicated(u_ode, dx), Const(1.0),Enzyme.Duplicated(numerical_flux, numerical_flux_shadow))
        dys[:, i] .= dy
        dx[i] = 0.0
    end
    return dys
end
# %%
# https://github.com/EnzymeAD/Enzyme.jl/pull/1545/files
function pick_batchsize(x)
    totalsize = length(x)
    return min(totalsize, 16)
end
function jacobian_ad_forward_enzyme_cache_upwind(x::AbstractVector, ::Val{chunk};
    dy = chunkedonehot(x, Val(chunk)),
    dx = chunkedonehot(x, Val(chunk))
    ) where {chunk}
   if chunk == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    u_ode = zeros(length(x))
    du_ode = zeros(length(x))

    tmp = ntuple(length(dx)) do i
        Enzyme.autodiff(Enzyme.Forward, upwind!, Enzyme.BatchDuplicated(du_ode, dy[i]), Enzyme.BatchDuplicated(u_ode, dx[i]))
        dy[i]
    end

    cols = Enzyme.tupleconcat(tmp...)
    return reduce(hcat, cols)
end
@time J2 =jacobian_ad_forward_enzyme_cache_upwind(x, Val(pick_batchsize(x)));
# %%

function gradients_ad_reverse_enzyme_cache_upwind(x::AbstractVector)
    u_ode = zeros(length(x))
    du_ode = zeros(length(x))
    numerical_flux=zeros(length(x))

    dy = zeros(size(du_ode))
    dx = zeros(size(u_ode))
    dxs = zeros(length(du_ode), length(du_ode))
    numerical_flux_shadow=zeros(length(x))

    for i in 1:length(x)
        dy[i] = 1.0
        # cache is passed to upwind!
        Enzyme.autodiff(Enzyme.Reverse, upwind!, Enzyme.Duplicated(du_ode, dy), Enzyme.Duplicated(u_ode, dx), Const(1.0),Enzyme.Duplicated(numerical_flux, numerical_flux_shadow))
        dxs[i, :] .= dx
        dx .= 0
    end
    return dxs
end
# %%
# Const is default?

x = -1.0:0.01:1.0
@time J1=gradients_ad_forward_enzyme_cache_upwind(x)
@time J2=gradients_ad_reverse_enzyme_cache_upwind(x)
J1 == J2
