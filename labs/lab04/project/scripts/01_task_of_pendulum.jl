using DrWatson
@quickactivate "project"

using DifferentialEquations
using Plots
using DataFrames
using JLD2

script_name = isempty(PROGRAM_FILE) ? "interactive" : splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

# ## Начальные условия и временной интервал
x0 = 0.5
y0 = -1.5
u0 = [x0, y0]

t0 = 0.0
tmax = 59.0
tspan = (t0, tmax)
tgrid = collect(LinRange(t0, tmax, 1000))

# ## Первая модель: незатухающие колебания
w1 = 5.2
p1 = (w = w1,)

function syst1!(dy, y, p, t)
    dy[1] = y[2]
    dy[2] = -p.w * y[1]
end

# ## Вторая модель: затухающие колебания
w2 = 0.5
g2 = 14.0
p2 = (w = w2, g = g2)

function syst2!(dy, y, p, t)
    dy[1] = y[2]
    dy[2] = -p.g * y[2] - p.w * y[1]
end

# ## Третья модель: затухающие колебания с внешним воздействием
w3 = 0.3
g3 = 13.0
p3 = (w = w3, g = g3)

function P(t)
    return 0.8 * sin(9 * t)
end

function syst3!(dy, y, p, t)
    dy[1] = y[2]
    dy[2] = -p.g * y[2] - p.w * y[1] + P(t)
end

# ## Утилита для решения, построения графиков и сохранения результатов
function run_model(case_name; model!, u0, tspan, p, tgrid, vel_file, phase_file)
    prob = ODEProblem(model!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat = tgrid)

    df = DataFrame(
        t = sol.t,
        x = getindex.(sol.u, 1),
        y = getindex.(sol.u, 2)
    )

    plt1 = plot(
        sol,
        idxs = (2),
        xlabel = "t",
        ylabel = "y(t)",
        title = "Модель $case_name: зависимость y(t)",
        label = "y(t)",
        lw = 2,
        legend = :topright
    )
    savefig(plt1, plotsdir(script_name, vel_file))

    plt2 = plot(
        sol,
        idxs = (1, 2),
        xlabel = "x",
        ylabel = "y",
        title = "Модель $case_name: фазовый портрет",
        label = "y(x)",
        lw = 2,
        legend = :topright
    )
    savefig(plt2, plotsdir(script_name, phase_file))

    @save datadir(script_name, "data_$(case_name).jld2") df p u0 tspan tgrid

    return (sol = sol, df = df, plt_time = plt1, plt_phase = plt2)
end

# ## Запуск первой модели
res1 = run_model(
    "1";
    model! = syst1!,
    u0 = u0,
    tspan = tspan,
    p = p1,
    tgrid = tgrid,
    vel_file = "01j.png",
    phase_file = "02j.png"
)

println("Модель 1 — первые 5 строк таблицы:")
println(first(res1.df, 5))

# ## Запуск второй модели
res2 = run_model(
    "2";
    model! = syst2!,
    u0 = u0,
    tspan = tspan,
    p = p2,
    tgrid = tgrid,
    vel_file = "03j.png",
    phase_file = "04j.png"
)

println("\nМодель 2 — первые 5 строк таблицы:")
println(first(res2.df, 5))

# ## Запуск третьей модели
res3 = run_model(
    "3";
    model! = syst3!,
    u0 = u0,
    tspan = tspan,
    p = p3,
    tgrid = tgrid,
    vel_file = "05j.png",
    phase_file = "06j.png"
)

println("\nМодель 3 — первые 5 строк таблицы:")
println(first(res3.df, 5))

# (опционально) показать последний график в интерактивной среде
res3.plt_phase
