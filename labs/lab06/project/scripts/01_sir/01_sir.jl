using DrWatson
@quickactivate "project"

using DifferentialEquations
using Plots
using DataFrames
using CSV
using JLD2

script_name = isempty(PROGRAM_FILE) ? "interactive" : splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

N = 11400.0
I0 = 250.0
R0 = 47.0
S0 = N - I0 - R0

u0 = [S0, I0, R0]

t0 = 0.0
tmax = 200.0
tspan = (t0, tmax)

a = 0.11
b = 0.02

p = (a = a, b = b)

function sir_case_1!(dy, y, p, t)
    dy[1] = 0.0
    dy[2] = p.b * y[2]
    dy[3] = -p.b * y[2]
end

function sir_case_2!(dy, y, p, t)
    dy[1] = -p.a * y[1]
    dy[2] =  p.a * y[1] - p.b * y[2]
    dy[3] =  p.b * y[2]
end

function run_case(case_name; model!, u0, tspan, p, nt = 1000)

    tgrid = collect(LinRange(tspan[1], tspan[2], nt))

    prob = ODEProblem(model!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat = tgrid)

    df = DataFrame(
        t = sol.t,
        S = [u[1] for u in sol.u],
        I = [u[2] for u in sol.u],
        R = [u[3] for u in sol.u]
    )

    plt_time = plot(
        sol,
        xlabel = "t",
        ylabel = "Численность",
        label = ["S(t) — восприимчивые" "I(t) — инфицированные" "R(t) — выбывшие"],
        lw = 2,
        title = "Динамика SIR — $case_name",
        legend = :topright
    )

    plt_phase_si = plot(
        sol,
        idxs = (1, 2),
        xlabel = "S",
        ylabel = "I",
        lw = 2,
        label = "Фазовая траектория I(S)",
        title = "Фазовый портрет S-I — $case_name",
        legend = :topright
    )

    plt_phase_ir = plot(
        sol,
        idxs = (2, 3),
        xlabel = "I",
        ylabel = "R",
        lw = 2,
        label = "Фазовая траектория R(I)",
        title = "Фазовый портрет I-R — $case_name",
        legend = :topright
    )

    savefig(plt_time, plotsdir(script_name, "$(case_name)_time.png"))
    savefig(plt_phase_si, plotsdir(script_name, "$(case_name)_phase_SI.png"))
    savefig(plt_phase_ir, plotsdir(script_name, "$(case_name)_phase_IR.png"))

    CSV.write(datadir(script_name, "table_$(case_name).csv"), df)

    @save datadir(script_name, "data_$(case_name).jld2") df p u0 tspan nt

    return (
        sol = sol,
        df = df,
        plt_time = plt_time,
        plt_phase_si = plt_phase_si,
        plt_phase_ir = plt_phase_ir
    )
end

res_1 = run_case(
    "sir_case_1";
    model! = sir_case_1!,
    u0 = u0,
    tspan = tspan,
    p = p,
    nt = 1000
)

res_2 = run_case(
    "sir_case_2";
    model! = sir_case_2!,
    u0 = u0,
    tspan = tspan,
    p = p,
    nt = 1000
)

println("Первые 5 строк таблицы результатов для первой модели:")
println(first(res_1.df, 5))

println()
println("Первые 5 строк таблицы результатов для второй модели:")
println(first(res_2.df, 5))

res_1.plt_time
