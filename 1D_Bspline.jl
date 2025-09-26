
# ---- headless / HPC safe (set BEFORE loading Plots) ----
if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"   # GR off-screen PNG device; prevents gksqt/Qt
end
ENV["DISPLAY"] = ""            # ensure no X/Wayland session is used
# --------------------------------------------------------

using LinearAlgebra
using Printf
using Plots
using Colors
using Measures: mm   # <-- added

# Force GR backend and avoid opening windows
gr()
default(show = false, size=(1100, 600))
default(bottom_margin = 8mm)    # <-- added (simple fix for x-axis label cutoff)

# ---------------------------
# Knot construction (open uniform)
# ---------------------------
"""
    open_uniform_knots(a, b, nel, p) -> Vector{Float64}

Open/clamped uniform knot vector on [a,b] with `nel` equal spans (elements) and degree `p`.
Structure: [a,...,a] (p+1 times), internal knots a+h, ..., b-h, [b,...,b] (p+1 times)
"""
function open_uniform_knots(a::Real, b::Real, nel::Integer, p::Integer)
    @assert nel ≥ 1 "Need at least one element."
    @assert p ≥ 0 "Degree must be nonnegative."
    h = (b - a) / nel
    internal = (nel > 1) ? collect(a .+ (1:nel-1) .* h) : Float64[]
    return vcat(fill(float(a), p+1), internal, fill(float(b), p+1))
end

"""
    n_basis(U, p) -> Int

Number of B-spline basis functions for knot vector `U` and degree `p`.
"""
n_basis(U::AbstractVector, p::Integer) = length(U) - p - 1

# ---------------------------
# Cox–de Boor (all basis at a single x)
# ---------------------------
@inline safe_div(num::Float64, den::Float64) = (den == 0.0 ? 0.0 : num/den)

"""
    bspline_all_at_x(U, p, x) -> Vector{Float64}

Return all basis values N_{i,p}(x) for i = 1..nb, using Cox–de Boor recursion.
Defines N_{i,0}(x)=1 on [ξ_i, ξ_{i+1}) except closes the last basis at x=b (so sum=1 at x=b).
Implements 0/0 → 0.
"""
function bspline_all_at_x(U::AbstractVector{<:Real}, p::Integer, x::Real)
    U = collect(float.(U))
    x = float(x)
    nb = n_basis(U, p)
    N = zeros(nb)
    # degree 0
    for i in 1:nb
        if (x >= U[i] && x < U[i+1]) || (x == U[end] && i == nb)  # close last interval at x=b
            N[i] = 1.0
        end
    end
    # elevate degree
    for k in 1:p
        Nprev = N
        N = zeros(nb)
        @inbounds for i in 1:nb
            denom1 = U[i+k] - U[i]
            t1 = safe_div(x - U[i], denom1) * Nprev[i]
            denom2 = U[i+k+1] - U[i+1]
            Ni1 = (i < nb) ? Nprev[i+1] : 0.0
            t2 = safe_div(U[i+k+1] - x, denom2) * Ni1
            N[i] = t1 + t2
        end
    end
    return N
end

"""
    basis_matrix(U, p, xs) -> Matrix{Float64}

Return matrix B of size (nbasis, length(xs)) with B[i,j] = N_{i,p}(xs[j]).
"""
function basis_matrix(U::AbstractVector{<:Real}, p::Integer, xs::AbstractVector{<:Real})
    nb = n_basis(U, p)
    B = Matrix{Float64}(undef, nb, length(xs))
    for (j, x) in pairs(xs)
        B[:, j] = bspline_all_at_x(U, p, x)
    end
    return B
end

# ---------------------------
# Derivatives (for continuity checks & plots)
# ---------------------------
"""
    bspline_all_derivative_at_x(U, p, x, order) -> Vector{Float64}

Compute the `order`-th derivative values of all degree-`p` B-splines at `x`.
d/dx N_{i,q}(x) = q/(ξ_{i+q}-ξ_i) N_{i,q-1}(x) - q/(ξ_{i+q+1}-ξ_{i+1}) N_{i+1,q-1}(x).
Valid for 0 ≤ order ≤ p.
"""
function bspline_all_derivative_at_x(U::AbstractVector{<:Real}, p::Integer, x::Real, order::Integer)
    @assert 0 ≤ order ≤ p
    order == 0 && return bspline_all_at_x(U, p, x)
    U = collect(float.(U)); x = float(x)
    q0 = p - order
    cur = bspline_all_at_x(U, q0, x)  # start at degree p-order
    for s in 1:order
        q = q0 + s
        nb_q = n_basis(U, q)
        nxt = zeros(nb_q)
        @inbounds for i in 1:nb_q
            t1 = safe_div(q * cur[i],                       U[i+q]   - U[i])
            t2 = safe_div(q * (i < length(cur) ? cur[i+1] : 0.0), U[i+q+1] - U[i+1])
            nxt[i] = t1 - t2
        end
        cur = nxt
    end
    return cur
end

# ---------------------------
# Checks
# ---------------------------
"""
    check_partition_of_unity(U, p; nx=2001) -> max_err
"""
function check_partition_of_unity(U::AbstractVector{<:Real}, p::Integer; nx::Int=2001)
    a, b = float(U[1]), float(U[end])
    xs = range(a, b; length=nx)
    B = basis_matrix(U, p, xs)
    return maximum(abs.(vec(sum(B; dims=1)) .- 1.0))
end

"""
    check_nonnegativity(U, p; nx=2001) -> min_value
"""
function check_nonnegativity(U::AbstractVector{<:Real}, p::Integer; nx::Int=2001)
    a, b = float(U[1]), float(U[end])
    xs = range(a, b; length=nx)
    B = basis_matrix(U, p, xs)
    return minimum(B)
end

"""
    check_support_overlap(U, p; tol=1e-12) -> (expected, min_count, max_count)
"""
function check_support_overlap(U::AbstractVector{<:Real}, p::Integer; tol::Float64=1e-12)
    K = unique(float.(U)); sort!(K)
    spans = [(K[j], K[j+1]) for j in 1:length(K)-1 if K[j+1] > K[j]]
    counts = Int[]
    for (s1, s2) in spans
        xmid = (s1 + s2) / 2
        N = bspline_all_at_x(U, p, xmid)
        push!(counts, count(>(tol), N))
    end
    expected = p + 1
    return (expected, minimum(counts), maximum(counts))
end

"""
    check_continuity(U, p; rmax=p-1, eps=1e-8) -> Dict{Int, Float64}
"""
function check_continuity(U::AbstractVector{<:Real}, p::Integer; rmax::Int=p-1, eps::Float64=1e-8)
    rmax = max(0, min(rmax, p-1))
    a, b = float(U[1]), float(U[end])
    K = unique(float.(U)); sort!(K)
    spans = [K[j+1]-K[j] for j in 1:length(K)-1 if K[j+1] > K[j]]
    hmin = isempty(spans) ? (b - a) : minimum(spans)
    δ = eps * hmin
    internal = [ξ for ξ in K if ξ > a && ξ < b]
    result = Dict{Int, Float64}(r => 0.0 for r in 0:rmax)
    for ξ in internal
        xL, xR = ξ - δ, ξ + δ
        for r in 0:rmax
            dL = (r == 0) ? bspline_all_at_x(U, p, xL) : bspline_all_derivative_at_x(U, p, xL, r)
            dR = (r == 0) ? bspline_all_at_x(U, p, xR) : bspline_all_derivative_at_x(U, p, xR, r)
            result[r] = max(result[r], maximum(abs.(dR .- dL)))
        end
    end
    return result
end

# ---------------------------
# Plotting
# ---------------------------
"""
    plot_bspline_basis(U, p; nx=4001) -> plt

Plot all N_{i,p}(x). Legend inside to avoid clipping; finer grid.
"""
function plot_bspline_basis(U::AbstractVector{<:Real}, p::Integer; nx::Int=4001)
    a, b = float(U[1]), float(U[end])
    xs = range(a, b; length=nx)
    B = basis_matrix(U, p, xs)
    nb = size(B, 1)
    pu_err = maximum(abs.(vec(sum(B; dims=1)) .- 1.0))
    @printf "degree p = %d | #basis = %d | max |Σ_i N_{i,p}(x) - 1| over grid = %.3e\n" p nb pu_err

    cols = distinguishable_colors(nb)
    plt = plot(xlabel="x", ylabel="N_{i,$p}(x)", legend=:right,
               title="Open uniform B-splines on [$(a), $(b)],\n p=$p, nbasis=$nb")
    for i in 1:nb
        plot!(plt, xs, B[i, :], label="N[$i,$p]", color=cols[i])
    end
    return plt
end

"""
    plot_partition_of_unity(U, p; nx=4001) -> plt

Standalone plot of Σ_i N_{i,p}(x) with y=1 reference line.
"""
function plot_partition_of_unity(U::AbstractVector{<:Real}, p::Integer; nx::Int=4001)
    a, b = float(U[1]), float(U[end])
    xs = range(a, b; length=nx)
    B  = basis_matrix(U, p, xs)
    s  = vec(sum(B; dims=1))
    pu_err = maximum(abs.(s .- 1.0))
    ttl = "Partition of Unity (p=$p) — max dev = $(@sprintf("%.2e", pu_err))"
    plt = plot(xs, s, label="Σ N", xlabel="x", ylabel="Σ N_{i,$p}(x)",
               title=ttl, legend=:right)
    hline!(plt, [1.0], label="1 (ref)", lw=2, linestyle=:dot, color=:gray)
    return plt
end

"""
    plot_bspline_derivative(U, p; order=1, nx=4001) -> plt

Plot order-th derivative curves for degree-p B-splines.
"""
function plot_bspline_derivative(U::AbstractVector{<:Real}, p::Integer; order::Int=1, nx::Int=4001)
    @assert 1 ≤ order ≤ p
    a, b = float(U[1]), float(U[end])
    xs = range(a, b; length=nx)
    nb = n_basis(U, p)
    cols = distinguishable_colors(nb)
    ordsym = order == 1 ? "′" : order == 2 ? "″" : "^($order)"

    plt = plot(xlabel="x", ylabel="d^$order/dx^$order N_{i,$p}(x)", legend=:right,
               title="Derivatives of open uniform B-splines on [$(a), $(b)],\n p=$p, order=$order")
    for i in 1:nb
        di = [bspline_all_derivative_at_x(U, p, x, order)[i] for x in xs]
        plot!(plt, xs, di, label="N[$i,$p]$ordsym", color=cols[i])
    end
    return plt
end

# ---------------------------
# Demo
# ---------------------------
function run_all()
    a, b, nel = 0.0, 10.0, 10
    outdir = "plots_bspline"
    isdir(outdir) || mkdir(outdir)

    for p in (1, 2, 3)
        U  = open_uniform_knots(a, b, nel, p)
        nb = n_basis(U, p)
        @printf "p=%d: |Ξ|=%d  #basis=%d  (expected: elements + p = %d)\n" p length(U) nb nel + p

        # Basis plots
        plt = plot_bspline_basis(U, p; nx=4001)
        savefig(plt, joinpath(outdir, "bspline_p$(p).png"))

        # NEW: separate Partition-of-Unity plots
        plt_pu = plot_partition_of_unity(U, p; nx=4001)
        savefig(plt_pu, joinpath(outdir, "bspline_pu_p$(p).png"))

        # Checks
        pu_err = check_partition_of_unity(U, p)
        minval = check_nonnegativity(U, p)
        expected, minc, maxc = check_support_overlap(U, p)
        cont = check_continuity(U, p)
        @printf "  Checks → PU max err = %.3e | min basis = %.3e | overlap min/max = %d/%d (expected %d)\n" pu_err minval minc maxc expected
        for r in sort(collect(keys(cont)))
            @printf "           continuity: max jump of order-%d derivative = %.3e\n" r cont[r]
        end

        # Derivative plots for p=2,3
        if p ≥ 2
            plt_d1 = plot_bspline_derivative(U, p; order=1, nx=4001)
            savefig(plt_d1, joinpath(outdir, "bspline_deriv1_p$(p).png"))
        end
        if p ≥ 3
            plt_d2 = plot_bspline_derivative(U, p; order=2, nx=4001)
            savefig(plt_d2, joinpath(outdir, "bspline_deriv2_p$(p).png"))
        end
    end
    println("\nNote: For nel=10 elements, #basis = nel + p → 11,12,13 for p=1,2,3.")
end

# Run the demo if executed as a script
if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
