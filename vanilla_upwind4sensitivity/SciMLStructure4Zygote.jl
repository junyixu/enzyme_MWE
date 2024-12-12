using SciMLSensitivity, OrdinaryDiffEq, Zygote

import SciMLStructures as SS
const C = 0.2 # C = Δt/Δx

# Modified Cache structure
struct Cache{T}
    v::T  # velocity - tunable
    numerical_flux::Vector{T}  # auxiliary - not tunable
end

# Mark Cache as a SciMLStructure
SS.isscimlstructure(::Cache) = true

# Cache is immutable
SS.ismutablescimlstructure(::Cache) = false

# Implement Tunable portion
SS.hasportion(::SS.Tunable, ::Cache) = true

function SS.canonicalize(::SS.Tunable, p::Cache)
    # Only velocity is tunable, return it as a scalar
    buffer = p.v
    
    # Repack function creates new Cache with new tunable value
    repack = let p = p
        function repack(newbuffer)
            SS.replace(SS.Tunable(), p, newbuffer)
        end
    end
    
    return buffer, repack, false
end

function SS.replace(::SS.Tunable, p::Cache, newbuffer)
    # Create new Cache with new velocity value but same numerical_flux
    return Cache(newbuffer, p.numerical_flux)
end

# Original initialization function
function init1!(u::AbstractVector, x::AbstractVector)
    @. u[ x < -0.4] = 0.0
    @. u[-0.4 <= x < -0.2] = 1.0 - abs(x[-0.4 <= x < -0.2]+0.3) / 0.1
    @. u[-0.2 <= x < -0.1] = 0.0
    @. u[-0.1 <= x < -0.0] = 1.0
    @. u[ x >= 0.0 ] = 0.0
    return nothing
end

# Modified upwind function to use Cache
function upwind!(du::Vector, u::Vector, p::Cache, t)
    numerical_flux = p.numerical_flux
    v = p.v
    @. numerical_flux = u * v
    
    for i = 2:length(u)
        du[i] = -C * (numerical_flux[i] - numerical_flux[i-1])
    end
    du[1] = -C * (numerical_flux[1] - numerical_flux[end])
    return nothing
end

# Setup and solve
x = -1.0:0.1:2
u = similar(x)
du = similar(x)
init1!(u, x)
cache = Cache(1.0, zero(u))
dcache = Cache(1.0, zero(u))

prob = ODEProblem(upwind!, u, (0.0, 1.0), cache)
sol = solve(prob, Tsit5())

# Loss function
function loss(p::Cache)
    _prob = remake(prob, p=p)
    sol = solve(_prob, Tsit5(), saveat=0.1)
    return sum(sol.u[end])
end

autodiff(Reverse, loss, Active, Duplicated(u, du), Duplicated(cache, dcache), Const(prob))

# Get gradient
gradient(p -> loss(u0, p, prob), p) # with the SciMLStructure interface
