         # Complete OCP Optimal Algae Control Model

using OptimalControl
using NLPModelsIpopt
using Plots
using MadNLP

# ------------------ Parameters ------------------
kP = 1.6
K = 140.0
I0 = 300.0
α = 0.1
kR_bar = 1.5
z_bar = 35.0
n = 3
tf = 30.0

# ------------------ Biological Functions ------------------
function I_light(z)
    return I0 * exp(-α * z)
end

function vP(p, z)
    return kP * p * I_light(z) / (K + p * I_light(z))
end

function kR(z)
    return kR_bar * z^n / (z_bar^n + z^n)
end

# ------------------ OptimalControl Solution ------------------
function OCP(c0=0.1,p0=0.8,z0=5,L=50)
    ocp = @def begin
        t ∈ [0, tf], time
        x = (c, p, z) ∈ R^3, state
        U ∈ R^2, control
        c(0) == c0
        p(0) == p0
        z(0) == z0
        0 ≤ U(t)[1] ≤ 1
        -34 ≤ U(t)[2] ≤ 34
        0 ≤ c(t) ≤ 1
        0 ≤ p(t) ≤ 1
        0 ≤ z(t) ≤ L
        c(t) + p(t) ≤ 1
        ẋ(t) == [
            vP(p(t), z(t)) * (1 - c(t)) - kR(z(t)) * c(t) * (1 - c(t) - p(t)),
            U(t)[1] * kR(z(t)) * c(t) * (1 - c(t) - p(t)) - vP(p(t), z(t)) * p(t),
            U(t)[2]
        ]
        ∫(vP(p(t), z(t))) → max
end

    sol1 = solve(ocp, :direct, :adnlp, :madnlp; disc_method = :midpoint, grid_size = 50, display = false)
    sol2 = solve(ocp, :direct, :adnlp, :madnlp; disc_method = :midpoint, init = sol1, grid_size = 100, display = false)
    sol3 = solve(ocp, :direct, :adnlp, :madnlp; disc_method = :midpoint, init = sol2, grid_size = 200, display = false)
    sol4 = solve(ocp, :direct, :adnlp, :ipopt; disc_method = :midpoint, init = sol3, grid_size = 300, display = false)
    sol5 = solve(ocp, :direct, :adnlp, :ipopt; disc_method = :midpoint, init = sol4, grid_size = 400, display = false)
    sol6 = solve(ocp, :direct, :adnlp, :ipopt; disc_method = :midpoint, init = sol5, grid_size = 500, display = false)
    sol7 = solve(ocp, :direct, :adnlp, :ipopt; disc_method = :midpoint, init = sol6, grid_size = 600, display = false)
    sol8 = solve(ocp, :direct, :adnlp, :ipopt; disc_method = :midpoint, init = sol7, grid_size = 700, display = false)
    sol  = solve(ocp, :direct, :adnlp, :ipopt; disc_method = :midpoint, init = sol8, grid_size = 800, display = true)


    # ------------------ Results Visualization ------------------
    plt = plot(sol)
    display(plt)

    
    # ------------------ Extract and Print Solution Details ------------------
    #=
    λ = costate(sol)
    x = state(sol)
    U = control(sol)
    u = t -> U(t)[1]
    v = t -> U(t)[2]
    T = time_grid(sol)

    for i in eachindex(T)
        t = T[i]
        λ_val = λ(t)
        x_val = x(t)
        u_val = u(t)
        v_val = v(t)
        println("t = $(round(t, digits=2)) │ λ = $(round.(λ_val, digits=4)) │ c = $(round(x_val[1], digits=4)) │ p = $(round(x_val[2], digits=4)) │ z = $(round(x_val[3], digits=4)) │ u = $(round(u_val, digits=4)) │ v = $(round(v_val, digits=4))")
    end
    println("Maximum objective value ∫vP = ", objective(sol))  
    =# 
end 