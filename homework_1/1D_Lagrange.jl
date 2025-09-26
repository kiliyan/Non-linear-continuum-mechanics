# --- Headless-safe plotting: force file backend for GR (must be set before using Plots)
ENV["GKSwstype"] = "100"      # no GUI; write directly to files
ENV["QT_QPA_PLATFORM"] = "offscreen"  # belt-and-suspenders; avoid Qt display

# 1D C0 Lagrange Finite-Element Bases on [0,10]
# p = 1, 2, 3 with 10 uniform elements (h = 1)
# -----------------------------------------------------------------------------
# What this script does
#  - Defines reference-element Lagrange bases on [0,1] for p=1,2,3
#  - Maps to physical elements via x = x_e_left + h * ξ
#  - Builds global nodal bases φ_i(x) on [0,10]
#  - Plots all global basis functions per p
#  - Verifies partition of unity Σ_i φ_i(x) ≡ 1 on a fine grid
#  - Checks (for Lagrange): local support and C0 continuity across interfaces
#  - Reports non-negativity (NOTE: Lagrange bases with p>=2 are **not** globally nonnegative)
#  - Visualizes first derivatives dφ_i/dx for p=2 and p=3
#
# Usage:
#   ] add Plots
#   include("1D_Lagrange_FE_Basis.jl")
#
# Plots are shown on screen and also saved under ./figs/
# -----------------------------------------------------------------------------

using Printf
using Plots
gr()  # ensure GR backend




using Base: gcd
# ---------------------------- Mesh & Reference Nodes --------------------------
struct Mesh1D
    a::Float64
    b::Float64
    nel::Int
    h::Float64
    xL::Vector{Float64}  # left coordinate of each element (length nel)
end

function make_mesh(a::Real, b::Real, nel::Int)
    @assert b > a "Mesh domain requires b > a"
    h = (b - a) / nel
    xL = [a + (e-1)*h for e in 1:nel]
    Mesh1D(float(a), float(b), nel, h, xL)
end

"""
ref_nodes(p) -> Vector{Float64}

Equispaced Lagrange nodes on [0,1]: ξ_j = j/p for j=0..p.
"""
ref_nodes(p::Int) = [j/p for j in 0:p]

# -------------------- Reference Lagrange Basis & Derivatives ------------------
"""
lagrange_eval(t, i, ξ) -> ℓ_i(ξ)

Evaluate the i-th Lagrange basis polynomial (0-based index i in 0..p) for the
node set `t` on [0,1] at parameter ξ.
"""
function lagrange_eval(t::AbstractVector{<:Real}, i::Int, ξ::Real)
    p = length(t)-1
    num = 1.0
    den = 1.0
    ti = t[i+1]
    @inbounds for j in 0:p
        j == i && continue
        tj = t[j+1]
        num *= (ξ - tj)
        den *= (ti - tj)
    end
    return num/den
end

"""
lagrange_eval_deriv(t, i, ξ) -> dℓ_i/dξ(ξ)

Derivative of i-th Lagrange basis using product-rule expansion:
If ℓ_i(ξ) = Π_{j≠i} (ξ - t_j) / Π_{j≠i} (t_i - t_j), then
ℓ_i'(ξ) = [ Σ_{m≠i} Π_{j≠i,m} (ξ - t_j) ] / Π_{j≠i} (t_i - t_j).
"""
function lagrange_eval_deriv(t::AbstractVector{<:Real}, i::Int, ξ::Real)
    p = length(t)-1
    den = 1.0
    ti = t[i+1]
    @inbounds for j in 0:p
        j == i && continue
        den *= (ti - t[j+1])
    end
    # sum over m != i of product over j != i,m of (ξ - t_j)
    sumprod = 0.0
    @inbounds for m in 0:p
        m == i && continue
        prod = 1.0
        for j in 0:p
            (j == i || j == m) && continue
            prod *= (ξ - t[j+1])
        end
        sumprod += prod
    end
    return sumprod/den
end

# ------------------------ Global Topology (nodes & conn) ----------------------
"""
build_topology(mesh, p)

Returns (xnodes, conn) where
  - xnodes :: Vector{Float64} of global node coordinates (ascending)
  - conn   :: Vector{NTuple{p+1,Int}} local->global mapping per element e

Convention: reference nodes are equispaced; element-local nodes at ξ_j = j/p.
Vertices at element interfaces are shared globally; interior (element) nodes
are unique per element.
"""
function build_topology(mesh::Mesh1D, p::Int)
    t = ref_nodes(p)
    # Use a dictionary keyed by Rational to de-duplicate exactly
    # helper to make exact rationals at positions a + (e-1) + j/p
    function ratpos(e::Int, j::Int)
        # position = a + (e-1) + j/p
        num = j   # numerator over p
        den = p   # denominator
        # reduce j/p
        g = gcd(num, den)
        num ÷= g; den ÷= g
        # now add (e-1)
        # (e-1) + num/den = ((e-1)*den + num)/den
        n2 = (e-1)*den + num
        # finally add 'a' (assumed integer multiple of h; here a=0 and h=1)
        # For general a with integer h-multiple, we retain float at end; exact de-dup via (e-1) and j/p is sufficient here.
        return n2 // den
    end
    # mapping rational -> global index
    dict = Dict{Rational{Int}, Int}()
    conns = Vector{NTuple{p+1,Int}}(undef, mesh.nel)
    for e in 1:mesh.nel
        ids = ntuple(j->begin
            r = ratpos(e, j-1)
            if !haskey(dict, r)
                dict[r] = length(dict) + 1
            end
            dict[r]
        end, p+1)
        conns[e] = ids
    end
    # Assemble and sort nodes by coordinate
    xs_rat = collect(keys(dict))
    sort!(xs_rat)
    # remap dict to ascending x order
    reindex = Dict{Int,Int}()
    for (k, r) in enumerate(xs_rat)
        reindex[dict[r]] = k
    end
    # apply remap to connections
    for e in 1:length(conns)
        conns[e] = ntuple(j->reindex[conns[e][j]], p+1)
    end
    # final node coordinates (Float64)
    xnodes = Float64.(xs_rat) .+ mesh.a  # here mesh.a = 0
    return xnodes, conns
end

# ---------------------- Evaluate Global Basis and Derivatives -----------------
"""
elem_index(mesh, x) -> e

Return element index e (1..nel) such that x ∈ [xL[e], xL[e]+h], with the right
boundary belonging to the last element.
"""
function elem_index(mesh::Mesh1D, x::Real)
    if x <= mesh.a
        return 1
    elseif x >= mesh.b
        return mesh.nel
    else
        e = Int(floor((x - mesh.a) / mesh.h)) + 1
        return clamp(e, 1, mesh.nel)
    end
end

"""
eval_global_basis!(ϕ, mesh, p, xnodes, conn, x; t = ref_nodes(p))

Fill ϕ with values of all global basis functions at x.
Only the (p+1) DOFs of the containing element contribute at any interior x.
"""
function eval_global_basis!(ϕ::AbstractVector{<:Real}, mesh::Mesh1D, p::Int,
                           xnodes::AbstractVector{<:Real}, conn, x::Real; t = ref_nodes(p))
    fill!(ϕ, 0.0)
    e = elem_index(mesh, x)
    xL = mesh.xL[e]
    ξ = (x - xL) / mesh.h
    # Guard against slight floating error at right boundary
    ξ = clamp(ξ, 0.0, 1.0)
    for (a, gid) in enumerate(conn[e])
        ϕ[gid] = lagrange_eval(t, a-1, ξ)
    end
    return ϕ
end

"""
eval_global_basis_dx!(dϕ, ...) similar to eval_global_basis! but returns dφ/dx.
"""
function eval_global_basis_dx!(dϕ::AbstractVector{<:Real}, mesh::Mesh1D, p::Int,
                              xnodes::AbstractVector{<:Real}, conn, x::Real; t = ref_nodes(p))
    fill!(dϕ, 0.0)
    e = elem_index(mesh, x)
    xL = mesh.xL[e]
    ξ = (x - xL) / mesh.h
    ξ = clamp(ξ, 0.0, 1.0)
    invh = 1.0 / mesh.h
    for (a, gid) in enumerate(conn[e])
        dϕ[gid] = invh * lagrange_eval_deriv(t, a-1, ξ)
    end
    return dϕ
end

# ----------------------------- Quality Checks ---------------------------------
function partition_of_unity_error(mesh::Mesh1D, p::Int, xnodes, conn; npts=5001)
    t = ref_nodes(p)
    xs = range(mesh.a, mesh.b; length=npts)
    ndof = length(xnodes)
    ϕ = zeros(ndof)
    maxerr = 0.0
    vals = similar(xs) |> collect
    for (k,x) in enumerate(xs)
        eval_global_basis!(ϕ, mesh, p, xnodes, conn, x; t=t)
        s = sum(ϕ)
        vals[k] = s
        err = abs(s - 1)
        maxerr = max(maxerr, err)
    end
    return maxerr, xs, vals
end

function basis_support_lengths(mesh::Mesh1D, p::Int, xnodes, conn; npts=5001, tol=1e-12)
    t = ref_nodes(p)
    xs = range(mesh.a, mesh.b; length=npts)
    ndof = length(xnodes)
    ϕ = zeros(ndof)
    counts = zeros(Int, ndof)
    for x in xs
        eval_global_basis!(ϕ, mesh, p, xnodes, conn, x; t=t)
        for i in 1:ndof
            if abs(ϕ[i]) > tol
                counts[i] += 1
            end
        end
    end
    Δx = step(xs)
    lengths = counts .* Δx
    return lengths
end

function basis_nonnegativity_report(mesh::Mesh1D, p::Int, xnodes, conn; npts=5001, tol=1e-12)
    t = ref_nodes(p)
    xs = range(mesh.a, mesh.b; length=npts)
    ndof = length(xnodes)
    ϕ = zeros(ndof)
    mins = fill(+Inf, ndof)
    for x in xs
        eval_global_basis!(ϕ, mesh, p, xnodes, conn, x; t=t)
        for i in 1:ndof
            if abs(ϕ[i]) > tol
                mins[i] = min(mins[i], ϕ[i])
            end
        end
    end
    global_min = minimum(mins)
    return mins, global_min
end

function continuity_jump_report(mesh::Mesh1D, p::Int, xnodes, conn; eps=1e-8)
    t = ref_nodes(p)
    ndof = length(xnodes)
    ϕL = zeros(ndof)
    ϕR = zeros(ndof)
    maxjump = 0.0
    jumps = Float64[]
    for k in 1:(mesh.nel-1)
        xb = mesh.a + k*mesh.h
        eval_global_basis!(ϕL, mesh, p, xnodes, conn, xb - eps; t=t)
        eval_global_basis!(ϕR, mesh, p, xnodes, conn, xb + eps; t=t)
        for i in 1:ndof
            jmp = abs(ϕL[i] - ϕR[i])
            push!(jumps, jmp)
            maxjump = max(maxjump, jmp)
        end
    end
    return maxjump
end

# ---------------------------- Plotting Utilities ------------------------------
function plot_global_bases(mesh::Mesh1D, p::Int, xnodes, conn; npts=2001)
    t = ref_nodes(p)
    xs = range(mesh.a, mesh.b; length=npts) |> collect
    ndof = length(xnodes)
    Φ = zeros(ndof, length(xs))
    ϕ = zeros(ndof)
    for (k,x) in enumerate(xs)
        eval_global_basis!(ϕ, mesh, p, xnodes, conn, x; t=t)
        @inbounds Φ[:,k] .= ϕ
    end
    plt = plot(title = @sprintf("Global Lagrange bases (p=%d), nel=%d", p, mesh.nel), xlabel="x", ylabel="φ_i(x)", legend=false)
    for i in 1:ndof
        plot!(xs, Φ[i,:])
    end
    # vertical lines at element interfaces
    for k in 1:(mesh.nel-1)
        vline!([mesh.a + k*mesh.h], line=:dash, alpha=0.2)
    end
    return plt
end

function plot_partition_of_unity(mesh::Mesh1D, p::Int, xnodes, conn; npts=3001)
    maxerr, xs, vals = partition_of_unity_error(mesh, p, xnodes, conn; npts=npts)
    plt = plot(xs, vals, xlabel="x", ylabel="Σ_i φ_i(x)", label="sum of bases",
               title = @sprintf("Partition of unity (p=%d) | max |Σ−1| = %.2e", p, maxerr))
    hline!([1.0], line=:dash, label="1")
    return plt
end

function plot_derivatives(mesh::Mesh1D, p::Int, xnodes, conn; npts=2001)
    @assert p >= 2 "Derivative plot requested for p>=2"
    t = ref_nodes(p)
    xs = range(mesh.a, mesh.b; length=npts) |> collect
    ndof = length(xnodes)
    dΦ = zeros(ndof, length(xs))
    dϕ = zeros(ndof)
    for (k,x) in enumerate(xs)
        eval_global_basis_dx!(dϕ, mesh, p, xnodes, conn, x; t=t)
        @inbounds dΦ[:,k] .= dϕ
    end
    plt = plot(title = @sprintf("First derivatives dφ_i/dx (p=%d)", p), xlabel="x", ylabel="dφ_i/dx", legend=false)
    for i in 1:ndof
        plot!(xs, dΦ[i,:])
    end
    for k in 1:(mesh.nel-1)
        vline!([mesh.a + k*mesh.h], line=:dash, alpha=0.2)
    end
    return plt
end

# ---------------------------- Driver / Main -----------------------------------
function run_all(; a=0.0, b=10.0, nel=10, outdir="figs")
    mesh = make_mesh(a,b,nel)
    isdir(outdir) || mkpath(outdir)

    for p in (1,2,3)
        xnodes, conn = build_topology(mesh, p)
        ndof = length(xnodes)
        println("\n======================= p = $p =======================")
        println("#elements = $(mesh.nel), h = $(mesh.h)")
        println("Global DOFs (nodes) = $ndof")
        @printf("Nodes (first 10): %s ...\n", string(xnodes[1:min(end,10)]))

        # Plots: bases & PoU
        plt1 = plot_global_bases(mesh, p, xnodes, conn)
        savefig(plt1, joinpath(outdir, @sprintf("basis_p%d.png", p)))
        if Base.isinteractive() && haskey(ENV, "DISPLAY")
        display(plt1)
    end

        plt2 = plot_partition_of_unity(mesh, p, xnodes, conn)
        savefig(plt2, joinpath(outdir, @sprintf("pou_p%d.png", p)))
        if Base.isinteractive() && haskey(ENV, "DISPLAY")
        display(plt2)
    end

        if p >= 2
            pld = plot_derivatives(mesh, p, xnodes, conn)
            savefig(pld, joinpath(outdir, @sprintf("derivatives_p%d.png", p)))
            if Base.isinteractive() && haskey(ENV, "DISPLAY")
        display(pld)
    end
        end

        # Quality checks
        maxerr, = partition_of_unity_error(mesh, p, xnodes, conn)
        println(@sprintf("Partition-of-unity max |Σφ−1| = %.3e", maxerr))

        lens = basis_support_lengths(mesh, p, xnodes, conn)
        # Expected support length: 2h for interior vertices, 1h otherwise
        expected = similar(lens)
        for i in eachindex(xnodes)
            xi = xnodes[i]
            if isapprox(xi, mesh.a; atol=1e-12) || isapprox(xi, mesh.b; atol=1e-12)
                expected[i] = mesh.h                # boundary vertex: 1 element
            elseif isapprox(xi, round(xi); atol=1e-12)
                expected[i] = 2*mesh.h              # interior vertex: two elements
            else
                expected[i] = mesh.h                # element-internal nodes: one element
            end
        end
        sup_err = maximum(abs.(lens .- expected))
        println(@sprintf("Support length check: max |measured−expected| = %.2f", sup_err))

        mins, global_min = basis_nonnegativity_report(mesh, p, xnodes, conn)
        nonneg_ok = global_min >= -1e-12
        println(@sprintf("Non-negativity: min φ = %.3e  (%s)", global_min,
                         nonneg_ok ? "OK (within tol)" : "NOT satisfied for some bases"))

        jmp = continuity_jump_report(mesh, p, xnodes, conn)
        println(@sprintf("C0 continuity across interfaces: max jump = %.3e", jmp))
    end

    println("\nAll plots saved under $(abspath(outdir))")
    nothing
end

# Optional: run automatically only when executed as a script (safe for VS Code / REPL)
if Base.isinteractive() == false && abspath(Base.program_file()) == abspath(@__FILE__)
    run_all()
end
