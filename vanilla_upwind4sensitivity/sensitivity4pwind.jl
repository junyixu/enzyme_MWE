using SciMLSensitivity, OrdinaryDiffEq, Enzyme

# struct Cache{T}
#     v::T # velocity
#     numerical_flux::Vector{T}
# end

function init1!(u::AbstractVector, x::AbstractVector)
	@. u[ x < -0.4] = 0.0
	@. u[-0.4 <= x < -0.2] = 1.0 - abs(x[-0.4 <= x < -0.2]+0.3) / 0.1
	@. u[-0.2 <= x < -0.1] = 0.0
	@. u[-0.1 <= x < -0.0] = 1.0
	@. u[ x >= 0.0 ] = 0.0
    return nothing
end

const C = 0.2 # C = Δt/Δx

x = -1.0:0.1:2
u = similar(x)
init1!(u, x)

function upwind!(du::Vector, u::Vector, p, t)
    flux = u .* p[1]
	for i = 2:length(u)
        du[i] = - C * (flux[i] - flux[i-1])  # Q_j^{n+1} = Q_j^n - Δt/Δx * ( F_{j+1/2}^n - F_{j-1/2}^n )
	end
    du[1] = - C * (flux[1] - flux[end])
    return nothing
end

du = zero(u)
p = [2.0]
dp = [1.0]
prob = ODEProblem(upwind!, u, (0.0, 1.0), p)
prob = remake(prob, p = p)

sol=solve(prob, Tsit5())

loss(u, p, prob) = sum(solve(prob, Tsit5(), u0 = u, p = p, saveat = 0.1))

Enzyme.autodiff(Reverse, loss, Active, Duplicated(u, du), Duplicated(p, dp), Const(prob))
