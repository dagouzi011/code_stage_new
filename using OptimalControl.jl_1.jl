using OptimalControl
using NLPModelsIpopt
using Plots
using MadNLP


kR = 2.1
kP = 1.6
K = 140
I = 200
kg = 0.00001
g = kg * (I^2)
tf = 15
function vP(p)
    return kP * p * I / (K + p * I)
end

plot(vP, 0, 1)

ocp = @def begin
    t ∈ [0, tf], time
    x = (c, p) ∈ R^2, state
    u ∈ R, control

    c(0) == 0.1
    p(0) == 0.8

    0 ≤ u(t) ≤ 1

    c(t) ≥ 0
    p(t) ≥ 0
    c(t) + p(t) ≤ 1

    ẋ(t) == [vP(p(t)) * (1 - c(t)) - kR * c(t) * (1 - c(t) - p(t)) + g * c(t) * p(t),
        u(t) * kR * c(t) * (1 - c(t) - p(t)) - p(t) * vP(p(t)) - g * p(t) * (1 - p(t))
    ]

    ∫(vP(p(t)) - g * p(t)) → max
end



sol1 = solve(ocp, :direct, :adnlp, :madnlp, grid_size=50, display=false)
sol2 = solve(ocp, :direct, :adnlp, :madnlp, init=sol1, grid_size=100, display=false)
sol3 = solve(ocp, :direct, :adnlp, :madnlp, init=sol2, grid_size=200, display=false)
sol4 = solve(ocp, :direct, :adnlp, :ipopt, init=sol3, grid_size=300, display=false)
sol5 = solve(ocp, :direct, :adnlp, :ipopt, init=sol4, grid_size=400, display=false)
sol6 = solve(ocp, :direct, :adnlp, :ipopt, init=sol5, grid_size=500, display=false)
sol7 = solve(ocp, :direct, :adnlp, :ipopt, init=sol6, grid_size=600, display=false)
sol8 = solve(ocp, :direct, :adnlp, :ipopt, init=sol7, grid_size=700, display=false)
sol = solve(ocp, :direct, :adnlp, :ipopt, init=sol8, grid_size=800, display=true)

plot(sol)
plt = plot(sol)            # 生成图像对象
savefig(plt, "code_julia11.png")

λ = costate(sol)           # 协变量函数 λ(t)
x = state(sol)             # 状态变量函数 x(t) = [c(t), p(t)]
u = control(sol)           # 控制变量函数 u(t)
T = time_grid(sol)         # 时间网格

for i in eachindex(T)
   t = T[i]
    λ_val = λ(t)
    x_val = x(t)
    u_val = u(t)
    println("t = $(round(t, digits=2)) │ λ = $(round.(λ_val, digits=4)) │ c = $(round(x_val[1], digits=4)) │ p = $(round(x_val[2], digits=4)) │ u = $(round(u_val, digits=4))")

end


