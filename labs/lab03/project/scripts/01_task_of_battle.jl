# # Две системы ОДУ: численное решение и визуализация
# **Цель:** решить две системы ОДУ, построить графики решений,
# сохранить изображения и таблицы с результатами.

# ## Инициализация проекта и загрузка пакетов
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
x0 = 32888.0
y0 = 17777.0
t0 = 0.0
tmax = 1.0

u0 = [x0, y0]
tspan = (t0, tmax)

# ## Параметры первой системы
a = 0.55
b = 0.77
c = 0.66
d = 0.44

p1 = (a = a, b = b, c = c, d = d)

# ## Параметры второй системы
a2 = 0.27
b2 = 0.88
c2 = 0.68
d2 = 0.37

p2 = (a = a2, b = b2, c = c2, d = d2)

# ## Внешние воздействия для первой системы
function P(t)
    return 1.5 * sin(3 * t + 1)
end

function Q(t)
    return 1.2 * cos(t + 1)
end

# ## Внешние воздействия для второй системы
function P2(t)
    return sin(20 * t)
end

function Q2(t)
    return cos(10 * t) + 1
end

# ## Определение моделей

# Первая система:
# dx/dt = -a*x - b*y + P(t)
# dy/dt = -c*x - d*y + Q(t)
function syst1!(dy, y, p, t)
    dy[1] = -p.a * y[1] - p.b * y[2] + P(t)
    dy[2] = -p.c * y[1] - p.d * y[2] + Q(t)
end

# Вторая система:
# dx/dt = -a*x - b*y + P2(t)
# dy/dt = -c*x*y - d*y + Q2(t)
function syst2!(dy, y, p, t)
    dy[1] = -p.a * y[1] - p.b * y[2] + P2(t)
    dy[2] = -p.c * y[1] * y[2] - p.d * y[2] + Q2(t)
end

# ## Утилита: один прогон (решение, график, таблица, сохранение)
function run_case(case_name; model!, u0, tspan, p, nt=100)
    # --- Сетка по времени ---
    tgrid = collect(LinRange(tspan[1], tspan[2], nt))

    # --- Решение ОДУ ---
    prob = ODEProblem(model!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tgrid)

    # --- Таблица с результатами ---
    df = DataFrame(
        t = sol.t,
        x = first.(sol.u),
        y = last.(sol.u)
    )

    # --- Визуализация ---
    plt = plot(sol,
        xlabel = "t",
        ylabel = "Значение",
        label = ["x(t)" "y(t)"],
        lw = 2,
        title = "Решение системы — $case_name",
        legend = :topright
    )

    # --- Сохранение ---
    savefig(plt, plotsdir(script_name, "$case_name.png"))
    @save datadir(script_name, "data_$case_name.jld2") df p u0 tspan nt

    return (plt = plt, df = df, sol = sol)
end

# ## Запуск 1: первая система
res1 = run_case("system1"; model! = syst1!, u0 = u0, tspan = tspan, p = p1, nt = 100)

println("Система 1 — первые 5 строк:")
println(first(res1.df, 5))

# ## Запуск 2: вторая система
res2 = run_case("system2"; model! = syst2!, u0 = u0, tspan = tspan, p = p2, nt = 100)

println("\nСистема 2 — первые 5 строк:")
println(first(res2.df, 5))

# (опционально) показать последний график в интерактивной среде
res2.plt
