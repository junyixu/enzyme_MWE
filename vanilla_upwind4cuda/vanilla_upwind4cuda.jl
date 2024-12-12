using CUDA
using Enzyme
using Test

# Constants for the simulation
const C = 0.5  # CFL number

# Main kernel for upwind scheme
function upwind_kernel!(du, u, p, t)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    
    n = length(u)
    for i = idx:stride:n
        if i == 1
            # Periodic boundary condition for first element
            du[1] = -p[1] * C * (u[1] * u[1] - u[end] * u[end])
        else
            # Regular upwind difference for other elements
            du[i] = -p[1] * C * (u[i] * u[i] - u[i-1] * u[i-1])
        end
    end
    return nothing
end

# Gradient kernel using Enzyme
function grad_upwind_kernel!(du, ddu, u, du_u, p, dp)
    autodiff_deferred(Reverse, upwind_kernel!, 
        Duplicated(du, ddu),   # output and its gradient
        Duplicated(u, du_u),   # input and its gradient 
        Duplicated(p, dp),     # parameters and their gradients
        Const(0.0)             # time is treated as constant
    )
    return nothing
end

# High-level CUDA kernel launcher
function upwind_cuda!(du::CuVector, u::CuVector, p, t)
    threads = 256  # Number of threads per block
    blocks = cld(length(u), threads)  # Number of blocks needed
    
    @cuda blocks=blocks threads=threads upwind_kernel!(du, u, p, t)
    return nothing
end

# High-level gradient kernel launcher
function grad_upwind_cuda!(du, ddu, u, du_u, p, dp)
    threads = 256
    blocks = cld(length(u), threads)
    
    @cuda blocks=blocks threads=threads grad_upwind_kernel!(du, ddu, u, du_u, p, dp)
    return nothing
end

# Helper function to setup simulation with gradients
function setup_cuda_simulation_with_grad(n::Int)
    # Primal arrays
    u = CUDA.ones(n)
    du = similar(u)
    p = cu([1.0])
    
    # Gradient arrays
    du_u = similar(u)  # gradient w.r.t. u
    ddu = similar(u)   # gradient w.r.t. du
    dp = similar(p)    # gradient w.r.t. parameters
    
    return du, ddu, u, du_u, p, dp
end

# Helper function for CPU reference implementation
function upwind_cpu!(du::Vector, u::Vector, p, t)
    flux = u .* p[1]
    for i = 2:length(u)
        du[i] = -C * (flux[i] - flux[i-1])
    end
    du[1] = -C * (flux[1] - flux[end])
    return nothing
end

# Test functions
function test_forward()
    n = 64
    # CPU version
    u_cpu = ones(n)
    du_cpu = similar(u_cpu)
    p_cpu = [1.0]
    upwind_cpu!(du_cpu, u_cpu, p_cpu, 0.0)
    
    # GPU version
    u_gpu = CUDA.ones(n)
    du_gpu = similar(u_gpu)
    p_gpu = cu([1.0])
    upwind_cuda!(du_gpu, u_gpu, p_gpu, 0.0)
    
    # Compare results
    return isapprox(Array(du_gpu), du_cpu, rtol=1e-5)
end

function test_gradient()
    n = 64  # smaller size for testing
    du, ddu, u, du_u, p, dp = setup_cuda_simulation_with_grad(n)
    
    # Initialize test values
    fill!(ddu, 1.0)  # seed the output gradient
    fill!(du_u, 0.0) # initialize input gradient
    fill!(dp, 0.0)   # initialize parameter gradient
    
    # Compute gradient
    grad_upwind_cuda!(du, ddu, u, du_u, p, dp)
    
    return Array(du_u), Array(dp)  # convert to CPU arrays for easier inspection
end

# Main function to run all tests
function run_all_tests()
    @testset "Upwind CUDA Tests" begin
        # Test forward pass
        @test test_forward()
        
        # Test gradient computation
        du_u, dp = test_gradient()
        @test all(isfinite, du_u)  # Check for valid gradients
        @test all(isfinite, dp)    # Check parameter gradients
        
        # Additional basic checks
        @test length(du_u) == 64   # Check expected size
        @test length(dp) == 1      # Check parameter gradient size
    end
end

# Example usage
function example_usage()
    # Setup
    n = 1000
    du, ddu, u, du_u, p, dp = setup_cuda_simulation_with_grad(n)
    
    # Forward pass
    upwind_cuda!(du, u, p, 0.0)
    
    # Gradient computation
    fill!(ddu, 1.0)  # Initialize gradient seed
    grad_upwind_cuda!(du, ddu, u, du_u, p, dp)
    
    # Get results back to CPU
    result_forward = Array(du)
    result_gradient_u = Array(du_u)
    result_gradient_p = Array(dp)
    
    return result_forward, result_gradient_u, result_gradient_p
end

# Run if this is the main file
if abspath(PROGRAM_FILE) == @__FILE__
    # Run tests
    run_all_tests()
    
    # Run example
    println("Running example simulation...")
    result_forward, result_gradient_u, result_gradient_p = example_usage()
    println("Simulation completed successfully!")
    println("First few values of forward pass: ", result_forward[1:5])
    println("First few values of u gradient: ", result_gradient_u[1:5])
    println("Parameter gradient: ", result_gradient_p)
end
