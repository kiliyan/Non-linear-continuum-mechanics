# lagrange_2d_tensor.jl
# 2D C0 Lagrange tensor-product bases on [0,10]^2 with nel=10 per axis, p ∈ {1,2,3}.
# Plots: corner / edge / interior basis surfaces; for p=2,3 also d/dx and d/dy surfaces.

# --- Headless-safe plotting (GR file backend) ---
ENV["GKSwstype"] = "100"            # write directly to files
ENV["QT_QPA_PLATFORM"] = "offscreen"

using Printf
using Plots
using Measures: mm
using Base: gcd

gr()
default(show=false, size=(1100, 800), bottom_margin=8mm)

# ---------------------------- 1D Mesh ----------------------------
struct Mesh1D
    a::Float64
    b::Float64
    nel::Int
    h::Float64
    xL::Vector{Float64}   # left coordinate of each element (length nel)
end

function make_mesh(a::Real, b::Real, nel::Int)
    @assert b > a
    h = (b - a) / nel
    xL = [a + (e-1)*h for e in 1:nel]
    Mesh1D(float(a), float(b), nel, h, xL)
end

# ----------------------- Reference Lagrange ----------------------
ref_nodes(p::Int) = [j/p for j in 0:p]

function lagrange_eval(t::AbstractVector{<:Real}, i::Int, ξ::Real)
    p = length(t)-1; num = 1.0; den = 1.0; ti = t[i+1]
    @inbounds for j in 0:p
        j==i && continue
        tj = t[j+1]; num *= (ξ - tj); den *= (ti - tj)
    end
    return num/den
end

function lagrange_eval_deriv(t::AbstractVector{<:Real}, i::Int, ξ::Real)
    p = length(t)-1; den = 1.0; ti = t[i+1]
    @inbounds for j in 0:p
        j==i && continue
        den *= (ti - t[j+1])
    end
    sumprod = 0.0
    @inbounds for m in 0:p
        m==i && continue
        prod = 1.0
        for j in 0:p
            (j==i || j==m) && continue
            prod *= (ξ - t[j+1])
        end
        sumprod += prod
    end
    return sumprod/den
end

# ----------------------- 1D Topology & Eval ----------------------
# Build global nodes and connectivity for 1D C0 Lagrange (vertices shared; interior element nodes unique)
function build_topology_1d(mesh::Mesh1D, p::Int)
    # exact de-dup using rationals at positions (e-1) + j/p (independent of h)
    function ratpos(e::Int, j::Int)
        num, den = j, p
        g = gcd(num, den); num ÷= g; den ÷= g
        n2 = (e-1)*den + num   # (e-1) + num/den
        return n2 // den
    end
    dict = Dict{Rational{Int}, Int}()
    conn = Vector{NTuple{p+1,Int}}(undef, mesh.nel)

    for e in 1:mesh.nel
        ids = ntuple(j->begin
            r = ratpos(e, j-1)
            if !haskey(dict, r)
                dict[r] = length(dict) + 1
            end
            dict[r]
        end, p+1)
        conn[e] = ids
    end

    xs_rat = collect(keys(dict)); sort!(xs_rat)

    # remap to ascending x order
    reindex = Dict{Int,Int}()
    for (k, r) in enumerate(xs_rat)
        reindex[dict[r]] = k
    end
    for e in 1:length(conn)
        conn[e] = ntuple(j->reindex[conn[e][j]], p+1)
    end

    # final node coordinates (Float64): x = a + h * ((e-1)+j/p)
    xnodes = mesh.a .+ mesh.h .* Float64.(xs_rat)
    return xnodes, conn
end


# element lookup and evaluation
function elem_index(mesh::Mesh1D, x::Real)
    if x <= mesh.a; return 1
    elseif x >= mesh.b; return mesh.nel
    else; e = Int(floor((x - mesh.a)/mesh.h)) + 1; return clamp(e, 1, mesh.nel)
    end
end

function eval_global_basis_1d!(ϕ::AbstractVector{<:Real}, mesh::Mesh1D, p::Int,
                               xnodes::AbstractVector{<:Real}, conn, x::Real; t=ref_nodes(p))
    fill!(ϕ, 0.0)
    e = elem_index(mesh, x)
    ξ = clamp((x - mesh.xL[e])/mesh.h, 0.0, 1.0)
    @inbounds for (a, gid) in enumerate(conn[e])
        ϕ[gid] = lagrange_eval(t, a-1, ξ)
    end
    return ϕ
end

function eval_global_basis_dx_1d!(dϕ::AbstractVector{<:Real}, mesh::Mesh1D, p::Int,
                                  xnodes::AbstractVector{<:Real}, conn, x::Real; t=ref_nodes(p))
    fill!(dϕ, 0.0)
    e = elem_index(mesh, x)
    ξ = clamp((x - mesh.xL[e])/mesh.h, 0.0, 1.0)
    invh = 1.0 / mesh.h
    @inbounds for (a, gid) in enumerate(conn[e])
        dϕ[gid] = invh * lagrange_eval_deriv(t, a-1, ξ)
    end
    return dϕ
end

# convenient samplers for a chosen global index
function line_values_for_basis(mesh::Mesh1D, p::Int, xnodes, conn, idx::Int, xs::AbstractVector)
    ndof = length(xnodes); ϕ = zeros(ndof); vals = similar(xs) |> collect
    for (k,x) in enumerate(xs)
        eval_global_basis_1d!(ϕ, mesh, p, xnodes, conn, x)
        vals[k] = ϕ[idx]
    end
    vals
end
function line_values_for_basis_dx(mesh::Mesh1D, p::Int, xnodes, conn, idx::Int, xs::AbstractVector)
    ndof = length(xnodes); dϕ = zeros(ndof); vals = similar(xs) |> collect
    for (k,x) in enumerate(xs)
        eval_global_basis_dx_1d!(dϕ, mesh, p, xnodes, conn, x)
        vals[k] = dϕ[idx]
    end
    vals
end

# nearest global node index to a coordinate
nearest_index(xnodes::AbstractVector{<:Real}, x0::Real) = findmin(abs.(xnodes .- x0))[2]

# ---------------------------- 2D Tensor Φ(x,y) ----------------------------
# Φ_ij(x,y) = φ_i(x) φ_j(y); ∂xΦ = φ'_i(x) φ_j(y); ∂yΦ = φ_i(x) φ'_j(y)
function surface_for_basis(meshx::Mesh1D, meshy::Mesh1D, p::Int,
                           xnodes_x, conn_x, xidx::Int,
                           xnodes_y, conn_y, yidx::Int;
                           nx=201, ny=201)
    xs = range(meshx.a, meshx.b; length=nx) |> collect
    ys = range(meshy.a, meshy.b; length=ny) |> collect
    Nx = line_values_for_basis(meshx, p, xnodes_x, conn_x, xidx, xs)     # length nx
    Ny = line_values_for_basis(meshy, p, xnodes_y, conn_y, yidx, ys)     # length ny
    Z  = Nx * transpose(Ny)                                              # (nx × ny)
    return xs, ys, Z
end

function surface_for_derivative(meshx::Mesh1D, meshy::Mesh1D, p::Int,
                                xnodes_x, conn_x, xidx::Int,
                                xnodes_y, conn_y, yidx::Int; dir::Symbol=:x,
                                nx=201, ny=201)
    @assert dir == :x || dir == :y
    xs = range(meshx.a, meshx.b; length=nx) |> collect
    ys = range(meshy.a, meshy.b; length=ny) |> collect
    if dir == :x
        Dx = line_values_for_basis_dx(meshx, p, xnodes_x, conn_x, xidx, xs)
        Ny = line_values_for_basis(meshy, p, xnodes_y, conn_y, yidx, ys)
        Z  = Dx * transpose(Ny)
    else
        Nx = line_values_for_basis(meshx, p, xnodes_x, conn_x, xidx, xs)
        Dy = line_values_for_basis_dx(meshy, p, xnodes_y, conn_y, yidx, ys)
        Z  = Nx * transpose(Dy)
    end
    return xs, ys, Z
end

# ---------------------------- Plot helpers ----------------------------
function save_surface(xs, ys, Z, path; ttl="Lagrange 2D basis", symmetric=false)
    plt = plot(camera=(30,30), xlabel="x", ylabel="y", zlabel="Φ(x,y)",
               title=ttl, bottom_margin=8mm)
    if symmetric
        m = maximum(abs.(Z)); m = (m == 0 ? 1.0 : m)
        surface!(plt, xs, ys, Z; color=cgrad(:balance), clim=(-m, m))
    else
        surface!(plt, xs, ys, Z; color=:viridis)
    end
    savefig(plt, path)
end

# ---------------------------- Driver ----------------------------
function run_all(; ax=0.0, bx=10.0, ay=0.0, by=10.0, nelx=10, nely=10, outdir="figs_lagrange_2d")
    meshx = make_mesh(ax, bx, nelx)
    meshy = make_mesh(ay, by, nely)
    isdir(outdir) || mkpath(outdir)

    for p in (1,2,3)
        xnodes_x, conn_x = build_topology_1d(meshx, p)
        xnodes_y, conn_y = build_topology_1d(meshy, p)

        # pick basis indices
        ix_corner = nearest_index(xnodes_x, ax)     # x=0
        iy_corner = nearest_index(xnodes_y, ay)     # y=0

        ix_edge   = nearest_index(xnodes_x, 5.0)    # x≈5
        iy_edge   = nearest_index(xnodes_y, ay)     # y=0 edge

        ix_int    = nearest_index(xnodes_x, 5.0)    # x≈5 interior
        iy_int    = nearest_index(xnodes_y, 5.0)    # y≈5 interior

        @printf "\n=== p=%d ===\n" p
        @printf "Corner node coords: (%.3f, %.3f)\n" xnodes_x[ix_corner] xnodes_y[iy_corner]
        @printf "Edge   node coords: (%.3f, %.3f)\n" xnodes_x[ix_edge]   xnodes_y[iy_edge]
        @printf "Interior node coords: (%.3f, %.3f)\n" xnodes_x[ix_int]  xnodes_y[iy_int]

        # --- Base surfaces
        for (label, ix, iy) in [("corner", ix_corner, iy_corner),
                                ("edge",   ix_edge,   iy_edge),
                                ("interior", ix_int,  iy_int)]
            xs, ys, Z = surface_for_basis(meshx, meshy, p, xnodes_x, conn_x, ix,
                                          xnodes_y, conn_y, iy)
            ttl = "Φ(x,y): $(label) basis (p=$p)"
            fn  = joinpath(outdir, @sprintf("lag2d_p%d_%s.png", p, label))
            save_surface(xs, ys, Z, fn; ttl=ttl, symmetric=false)
        end

        # --- First derivatives for p ≥ 2
        if p ≥ 2
            for (label, ix, iy) in [("corner", ix_corner, iy_corner),
                                    ("edge",   ix_edge,   iy_edge),
                                    ("interior", ix_int,  iy_int)]
                xs, ys, Zx = surface_for_derivative(meshx, meshy, p, xnodes_x, conn_x, ix,
                                                    xnodes_y, conn_y, iy; dir=:x)
                ttlx = "∂Φ/∂x: $(label) basis (p=$p)"
                fnx  = joinpath(outdir, @sprintf("lag2d_p%d_%s_dx.png", p, label))
                save_surface(xs, ys, Zx, fnx; ttl=ttlx, symmetric=true)

                xs, ys, Zy = surface_for_derivative(meshx, meshy, p, xnodes_x, conn_x, ix,
                                                    xnodes_y, conn_y, iy; dir=:y)
                ttly = "∂Φ/∂y: $(label) basis (p=$p)"
                fny  = joinpath(outdir, @sprintf("lag2d_p%d_%s_dy.png", p, label))
                save_surface(xs, ys, Zy, fny; ttl=ttly, symmetric=true)
            end
        end


    end

    println("\nSaved surfaces in: $(abspath(outdir))/")
end

# Auto-run if executed as a script
if Base.isinteractive() == false && abspath(Base.program_file()) == abspath(@__FILE__)
    run_all()
end
