# bsplines_2d_tensor_with_derivs.jl
# Tensor-product B-spline bases on [0,10]^2 with nel=10 per axis, p ∈ {1,2,3}.
# Saves surfaces for corner/edge/interior bases, and first-derivative surfaces for p=2,3.
# Headless-safe GR: writes PNGs without opening windows.

# --- headless-safe GR (set before using Plots) ---
if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"    # off-screen PNG backend
end
ENV["DISPLAY"] = ""             # avoid X/Wayland
# -------------------------------------------------

using LinearAlgebra
using Printf
using Plots
using Measures: mm

gr()
default(show=false, size=(1100, 800), bottom_margin=8mm)

# ---------------------------
# 1D knots and basis (open uniform)
# ---------------------------
"""
    open_uniform_knots(a, b, nel, p)
Open/clamped uniform knot vector on [a,b] with `nel` spans and degree `p`:
[a (p+1×), a+h, …, b−h, b (p+1×)] with h = (b-a)/nel.
"""
function open_uniform_knots(a::Real, b::Real, nel::Integer, p::Integer)
    @assert nel ≥ 1
    @assert p ≥ 0
    h = (b - a) / nel
    internal = (nel > 1) ? collect(a .+ (1:nel-1) .* h) : Float64[]
    return vcat(fill(float(a), p+1), internal, fill(float(b), p+1))
end

n_basis(U::AbstractVector, p::Integer) = length(U) - p - 1
@inline safe_div(num::Float64, den::Float64) = (den == 0.0 ? 0.0 : num/den)

"""
    bspline_all_at_x(U, p, x)
All degree-p values N_{i,p}(x), i = 1..n_basis(U,p). Last span closed at b.
"""
function bspline_all_at_x(U::AbstractVector{<:Real}, p::Integer, x::Real)
    U = collect(float.(U)); x = float(x)
    nb = n_basis(U, p)
    N = zeros(nb)
    # degree 0
    for i in 1:nb
        if (x >= U[i] && x < U[i+1]) || (x == U[end] && i == nb)
            N[i] = 1.0
        end
    end
    # elevate degree
    for k in 1:p
        Nprev = N
        N = zeros(nb)
        @inbounds for i in 1:nb
            t1 = safe_div(x - U[i],     U[i+k]   - U[i])   * Nprev[i]
            t2 = safe_div(U[i+k+1] - x, U[i+k+1] - U[i+1]) * (i < nb ? Nprev[i+1] : 0.0)
            N[i] = t1 + t2
        end
    end
    return N
end

"""
    basis_matrix(U, p, xs) -> B  (nbasis × length(xs))
B[i,j] = N_{i,p}(xs[j]).
"""
function basis_matrix(U::AbstractVector{<:Real}, p::Integer, xs::AbstractVector{<:Real})
    nb = n_basis(U, p)
    B = Matrix{Float64}(undef, nb, length(xs))
    for (j, x) in pairs(xs)
        B[:, j] = bspline_all_at_x(U, p, x)
    end
    return B
end

"""
    bspline_all_derivative_at_x(U, p, x, order)
order-th derivative of degree-p B-splines at x.
d/dx N_{i,q} = q/(ξ_{i+q}-ξ_i) N_{i,q-1} - q/(ξ_{i+q+1}-ξ_{i+1}) N_{i+1,q-1}
"""
function bspline_all_derivative_at_x(U::AbstractVector{<:Real}, p::Integer, x::Real, order::Integer)
    @assert 0 ≤ order ≤ p
    order == 0 && return bspline_all_at_x(U, p, x)
    U = collect(float.(U)); x = float(x)
    q0 = p - order
    cur = bspline_all_at_x(U, q0, x)
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

"""
    derivative_matrix(U, p, xs, order) -> D  (nbasis × length(xs))
D[i,j] = d^order/dx^order N_{i,p}(xs[j]).
"""
function derivative_matrix(U::AbstractVector{<:Real}, p::Integer, xs::AbstractVector{<:Real}, order::Integer)
    nb = n_basis(U, p)
    D = Matrix{Float64}(undef, nb, length(xs))
    for (j, x) in pairs(xs)
        D[:, j] = bspline_all_derivative_at_x(U, p, x, order)
    end
    return D
end

"""
    pick_index_at(U, p, x0) -> i
Return index i of the 1D basis that peaks near x0 (argmax at x0).
"""
function pick_index_at(U::AbstractVector{<:Real}, p::Integer, x0::Real)
    vals = bspline_all_at_x(U, p, x0)
    return findmax(vals)[2]
end

# ---------------------------
# 2D tensor-product surfaces
# ---------------------------
"""
    basis_surface(U, p, xs, ys, i, j) -> (X, Y, Z)
Z[m,n] = N_{i,p}(xs[m]) * N_{j,p}(ys[n]).
"""
function basis_surface(U::AbstractVector{<:Real}, p::Integer,
                       xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real},
                       i::Integer, j::Integer)
    Bx = basis_matrix(U, p, xs)      # (nb × Nx)
    By = basis_matrix(U, p, ys)      # (nb × Ny)
    Nx = Bx[i, :]                    # length Nx
    Ny = By[j, :]                    # length Ny
    Z  = Nx * transpose(Ny)          # (Nx × Ny)
    return (collect(xs), collect(ys), Z)
end

"""
    basis_surface_deriv(U, p, xs, ys, i, j; dir=:x) -> (X, Y, Z)
First-derivative surfaces:
- dir=:x → Z = N'_{i,p}(xs) * N_{j,p}(ys)
- dir=:y → Z = N_{i,p}(xs)  * N'_{j,p}(ys)
"""
function basis_surface_deriv(U::AbstractVector{<:Real}, p::Integer,
                             xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real},
                             i::Integer, j::Integer; dir::Symbol=:x)
    @assert dir == :x || dir == :y
    Bx = basis_matrix(U, p, xs)
    By = basis_matrix(U, p, ys)
    Dx = derivative_matrix(U, p, xs, 1)
    Dy = derivative_matrix(U, p, ys, 1)
    Nx, Ny = Bx[i, :], By[j, :]
    if dir == :x
        Z = Dx[i, :] * transpose(Ny)
    else # :y
        Z = Nx * transpose(Dy[j, :])
    end
    return (collect(xs), collect(ys), Z)
end

# ---------------------------
# Plot helpers
# ---------------------------
"""
    save_surface(xs, ys, Z, path; ttl, cmap=:viridis, symmetric=false)
Saves a surface plot. If `symmetric=true`, uses symmetric color limits about 0 (for derivatives).
"""
function save_surface(xs, ys, Z, path; ttl="B-spline surface", cmap=:viridis, symmetric::Bool=false)
    plt = plot(camera=(30,30), xlabel="x", ylabel="y", zlabel="B(x,y)",
               title=ttl, bottom_margin=8mm)
    if symmetric
        m = maximum(abs.(Z))
        surface!(plt, xs, ys, Z; color=cgrad(:balance), clim=(-m, m))
    else
        surface!(plt, xs, ys, Z; color=cmap)
    end
    savefig(plt, path)
end

# ---------------------------
# Demo
# ---------------------------
function run_all()
    a, b, nel = 0.0, 10.0, 10
    outdir = "plots_bspline_2d"
    isdir(outdir) || mkdir(outdir)

    for p in (1, 2, 3)
        U  = open_uniform_knots(a, b, nel, p)
        nb = n_basis(U, p)
        h  = (b - a) / nel
        @printf "Degree p=%d | #1D basis per axis = %d | element size h=%.3f\n" p nb h

        xs = range(a, b; length=201)
        ys = range(a, b; length=201)

        # Select indices
        i_corner, j_corner = 1, 1
        i_edge  = pick_index_at(U, p, 5.0); j_edge  = 1
        i_int   = pick_index_at(U, p, 5.0); j_int   = pick_index_at(U, p, 5.0)

        # --- Base surfaces (corner, edge, interior)
        for (lbl, i, j) in [("corner", i_corner, j_corner),
                            ("edge",   i_edge,   j_edge),
                            ("interior", i_int,  j_int)]
            X, Y, Z = basis_surface(U, p, xs, ys, i, j)
            ttl = "$(lbl) basis, p=$p (i=$i, j=$j)"
            fn  = joinpath(outdir, "bspline2d_p$(p)_$(lbl).png")
            save_surface(X, Y, Z, fn; ttl=ttl, cmap=:viridis)
        end

        # --- First derivatives for p ≥ 2
        if p ≥ 2
            for (lbl, i, j) in [("corner", i_corner, j_corner),
                                ("edge",   i_edge,   j_edge),
                                ("interior", i_int,  j_int)]
                # d/dx
                X, Y, Zx = basis_surface_deriv(U, p, xs, ys, i, j; dir=:x)
                ttlx = "∂/∂x of $(lbl) basis, p=$p (i=$i, j=$j)"
                fnx  = joinpath(outdir, "bspline2d_p$(p)_$(lbl)_dx.png")
                save_surface(X, Y, Zx, fnx; ttl=ttlx, symmetric=true)

                # d/dy
                X, Y, Zy = basis_surface_deriv(U, p, xs, ys, i, j; dir=:y)
                ttly = "∂/∂y of $(lbl) basis, p=$p (i=$i, j=$j)"
                fny  = joinpath(outdir, "bspline2d_p$(p)_$(lbl)_dy.png")
                save_surface(X, Y, Zy, fny; ttl=ttly, symmetric=true)
            end
        end

        # Console note on support & continuity (one example)
        x_supp = (U[i_int], U[i_int + p + 1])
        y_supp = (U[j_int], U[j_int + p + 1])
        @printf "  Support: (p+1)×(p+1) = %d×%d elements. Example interior x:[%.3f,%.3f], y:[%.3f,%.3f]\n" p+1 p+1 x_supp[1] x_supp[2] y_supp[1] y_supp[2]
        println("  Continuity: C^$(p-1) across interior knot lines in each direction.\n")
    end

    println("Saved surfaces to: $(abspath(outdir))/")
end

# Auto-run if executed as a script
if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
