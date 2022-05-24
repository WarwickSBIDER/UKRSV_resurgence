cd("./RSV_resurgence/")
using Pkg
Pkg.activate(".")
##

using OrdinaryDiffEq, StatsPlots, Statistics, PlutoUI, CSV, FileIO, DataFrames, Dates, LinearAlgebra, DiffEqFlux, Tullio
using DataInterpolations, DiffEqSensitivity, ForwardDiff, Zygote, JLD2, Plots.PlotMeasures

##Declare risk per infection

dis_risk₁ = [mean([31.2, 28.6]); mean([20.0, 13.0]); fill(2.0, 4)] / 100
dis_risk₂ = [fill(5.0, 5); 0.0] / 100
hosp_risk = [mean([32.76, 33.07, 21.9, 20.74, 18.86, 12, 27]), mean([9.4, 10.76, 9.1, 12.11, 9, 8.7, 6.7]),
    mean([6.7, 7.11, 7.78, 7.43, 4.13, 4.1, 10.52, 20, 14.84, 13.95, 10.18, 2.36, 8.41]),
    mean([3.76, 1.08, 0.19]), 0.0, 0.0] / 100

## Load data
# Add time series

datamart = CSV.File("datamart_other_resp_from_wk402013.csv") |> DataFrame
datamart.date = Date.(datamart.date, DateFormat("dd/mm/yyyy"));
datamart_agrp = CSV.File("datamart_age_2021_2022.csv") |> DataFrame;
sariwatch = CSV.File("SARI_watch_RSV.csv") |> DataFrame;

ref_date = Date(2017, 1, 1)
sariwatch.t = [Float64((d - ref_date).value) for d in sariwatch.date]
datamart.t = [Float64((d - ref_date).value) for d in datamart.date]
datamart_agrp.t = [Float64((d - ref_date).value) for d in datamart_agrp.date];

## Linear fit to project Sariwatch data into past

comparison_times = intersect(sariwatch.t, datamart.t[.~ismissing.(datamart.RSV_0_4_perc)])
comparison_datamart = datamart.RSV_0_4_perc[[t ∈ comparison_times for t in datamart.t]]
comparison_sariwatch = sariwatch.RSV_hosp_ICU_per_100_000[[t ∈ comparison_times for t in sariwatch.t]]
β_fit = comparison_datamart \ comparison_sariwatch

## Also projection each week with <= 0.5% datamart RSV as low RSV hosp incidence

projection_times = setdiff(datamart.t[.~ismissing.(datamart.RSV_0_4_perc)], comparison_times)
projection_idxs = [t ∈ projection_times for t in datamart.t]
projection_times2 = setdiff(setdiff(datamart.t[[p <= 0.5 for p in datamart.RSV_perc]], sariwatch.t), collect(1000.0:1.0:2000.0))
projection_idxs2 = [t ∈ projection_times2 for t in datamart.t]

ts_data_unsrted = [datamart.t[projection_idxs]; datamart.t[projection_idxs2]; sariwatch.t]
projection_unsrted = [datamart.RSV_0_4_perc[projection_idxs] .* β_fit; fill(0.05, length(datamart.t[projection_idxs2])); sariwatch.RSV_hosp_ICU_per_100_000]
idxs = sortperm(ts_data_unsrted)
ts = Float32.(ts_data_unsrted[idxs])
rsv_hosp_rate = Float32.(projection_unsrted[idxs])

## Plot data

T = 365.25f0
scatter(sariwatch.t ./ T, sariwatch.RSV_hosp_ICU_per_100_000, lab="Sariwatch data", ms=3,
    xlabel="Days (rel. 01/01/2017)",
    ylabel="RSV Hosp. Rate",
    size=(1000, 500))
scatter!(datamart.t[projection_idxs] ./ T, datamart.RSV_0_4_perc[projection_idxs] .* β_fit, ms=3, lab="Projected hosp. rate", color=2)
scatter!(datamart.t[projection_idxs2] ./ T, fill(0.05, length(datamart.t[projection_idxs2])), ms=3, lab="", color=2)
plot!(ts ./ T, rsv_hosp_rate)
vline!([3.2], lw=3, ls=:dash, lab="Measures start")
vline!([4.0], lw=3, ls=:dash, lab="Measures end")

## Plot 

influ_perc = ConstantInterpolation((datamart.Influenza_A_perc .+ datamart.Influenza_B_perc) ./ 100 .|> Float32, datamart.t .|> Float32)
rhino_perc = ConstantInterpolation(datamart.Rhinovirus_perc ./ 100 .|> Float32, datamart.t .|> Float32)

plot(datamart.t, (datamart.Influenza_A_perc .+ datamart.Influenza_B_perc) ./ 100 .|> Float32,)
plot!(datamart.t, (datamart.Rhinovirus_perc) ./ 100 .|> Float32)

## Neural network representation

β_nn = FastChain((x, p) -> x,
    FastDense(3, 50, gelu),
    FastDense(50, 1, sigmoid))

β_periodic = FastChain((x, p) -> x,
    FastDense(1, 33, x -> x .+ cos(x)^2),
    FastDense(33, 1, sigmoid))
p_init = initial_params(β_nn)
@time β_nn([0.0f0, influ_perc(0), rhino_perc(0)], p_init)[1]

incidence_nn_fitted = FastChain((x, p) -> x,
    FastDense(3, 64, x -> x .+ (1 / (2 * pi)) .* cos.((2 * pi) .* x) .^ 2),
    FastDense(64, 64, x -> x .+ (1 / (2 * pi)) .* cos.((2 * pi) .* x) .^ 2),
    FastDense(64, 1, x -> x .+ (1 / (2 * pi)) .* cos.((2 * pi) .* x) .^ 2))
# p_init = initial_params(β_periodic)
# @time β_periodic([0.0f0], p_init)[1]

function beta_t(t, p, a)
    β_periodic = FastChain((x, p) -> x,
        FastDense(1, 33, x -> x .+ (1 / a) * cos.(a .* x) .^ 2),
        FastDense(33, 1, sigmoid))
    return β_periodic(t, p)
end

@time beta_t([0.0f0], p_init, 1.0f0)

## Declare fixed parameters

proportion_of_population = [3 / 10; 3 / 10; 3 / 5; 3 * 3 / 5; 6.1; 90.9] ./ 100 .|> Float32
α = 1 / 2 #2 day incubation period
γ = 1 / 5 # 5 day infectious period
ω = 1 / 365 #6 months before return to potential susceptibility
σ = 0.65 #Reduction in susceptibility due to first episode
ι = 0.25 #Reduction in transmissibility due to first episode
N = 50e6#Pop size England
L = 80 * T #Mean life expectancy
μ = 1 / (L - 14 * T) #mortality rate assuming no mortality in first 14 years of life
B = μ * (N * proportion_of_population[end])#Daily Replacement rate of births
ϵ = 5e-6 #~1-10 infectious contacts from outside London per day not in measures

## Declare model

l_ages = [0.5 * T, 0.5 * T, T, 3 * T, 11 * T, Inf]
r_ages = 1.0 ./ l_ages
aging_mat = Bidiagonal(-r_ages, r_ages[1:(end-1)], :L)
aging_rates(A, u) = @tullio C[a, s, n] := A[a, b] * u[b, s, n]
function aging_rates!(X, A, u)
    @tullio X[a, s, n] = A[a, b] * u[b, s, n]
    return nothing
end

function ct_lkd(t, minpoint_y;
    measures_x=1177.0, minpoint_x=1184.0, returnpoint_x=1916.0)
    ct = 1.0 * (t < measures_x)
    ct += (1.0 + ((t - measures_x) * (minpoint_y - 1.0) / (minpoint_x - measures_x))) * (t ≥ measures_x && t < minpoint_x)
    ct += (minpoint_y + ((t - minpoint_x) * (1.0 - minpoint_y) / (returnpoint_x - minpoint_x))) * (t ≥ minpoint_x && t < returnpoint_x)
    ct += 1.0 * (t ≥ returnpoint_x)
end

plot(t -> ct_lkd(t, 0.4), 600, 2000)
##

T = 365.25f0
function f_RSV_nn(du, u, p, t)
    #Gather parameters and states
    β_school, β_LD, α, γ, σ, ι, B, N, μ, ϵ = p[1:10]
    # β_school, β_LD, α, γ, σ, ι, B, N, μ, ϵ = p[1:10]

    nn_params = p[11:end]
    S₁ = @view u[:, 1, 1]
    S₂ = @view u[:, 1, 2]
    I₁ = @view u[:, 2, 1]
    I₂ = @view u[:, 2, 2]
    R₁ = @view u[:, 3, 1]
    R₂ = @view u[:, 3, 2]
    #Time varying infection rates
    # β₀ = β_nn([cospi(2 * (t - ϕ) / T), influ_perc(t) / 10.0f0, rhino_perc(t) / 10.0f0], nn_params)[1]
    τ = t < ts[1] ? mod(t, T) : t
    # β₀ = β_nn([mod(t, T) / T, influ_perc(τ) / 10.0f0, rhino_perc(τ) / 10.0f0], nn_params)[1]
    β₀ = 1.5 * sigmoid(incidence_nn_fitted([t / T, influ_perc(τ), rhino_perc(τ)], nn_params)[1])
    # β₀ = beta_t([t / T], nn_params, ϕ)[1]
    inlockdown = (t >= T * 3.3 && t <= T * 3.8)
    schoolopen = (mod(t, T) < 0.553 * T || mod(t, T) > 0.668 * T)
    # β = 1.5 * β₀ * ct_lkd(t, β_LD)
    β = β₀ * ct_lkd(t, β_LD)

    β_s = β_school * (1 - inlockdown) * schoolopen * ct_lkd(t, β_LD)
    λ = (β * (sum(I₁) + ι * sum(I₂)) / N) + ϵ * (1 - inlockdown)#This assumes homogeneous mixing proportional to the size of each group
    λ_schools = β_s * (I₁[5] + ι * I₂[5]) / sum(u[5, :, :])  #School mixing
    #Transmission Dynamics
    du[:, 1, 1] .= -S₁ .* λ
    du[:, 1, 2] .= -σ .* S₂ .* λ .+ ω .* (R₁ .+ R₂)
    du[:, 2, 1] .= S₁ .* λ .- γ .* I₁
    du[:, 2, 2] .= σ .* S₂ .* λ .- γ .* I₂
    du[:, 3, 1] .= γ .* I₁ .- ω .* R₁
    du[:, 3, 2] .= γ .* I₂ .- ω .* R₂
    #Add schools based transmission
    du[5, 1, 1] += -S₁[5] * λ_schools
    du[5, 2, 1] += S₁[5] * λ_schools
    du[5, 1, 2] += -σ * S₂[5] * λ_schools
    du[5, 2, 2] += σ * S₂[5] * λ_schools
    #Add Demography
    du[1, 1, 1] += B # Births with maternal immunity
    du[end, :, :] .+= -μ .* u[end, :, :] #Deaths in the final age group
    #Add Aging
    du .+= aging_rates(aging_mat, u)
    return nothing
end
# function f_RSV_nn(du, u, p, t)
#     #Gather parameters and states
#     # β_school, β_LD, ϕ, α, γ, σ, ι, B, N, μ, ϵ = p[1:11]
#     β_school, β_LD, α, γ, σ, ι, B, N, μ, ϵ = p[1:10]

#     nn_params = p[11:end]
#     S₁ = @view u[:, 1, 1]
#     S₂ = @view u[:, 1, 2]
#     I₁ = @view u[:, 2, 1]
#     I₂ = @view u[:, 2, 2]
#     R₁ = @view u[:, 3, 1]
#     R₂ = @view u[:, 3, 2]
#     #Time varying infection rates
#     # β₀ = β_nn([cospi(2*(t-ϕ)/T), influ_perc(t)/10f0, rhino_perc(t)/10f0], nn_params)[1]
#     τ = t < ts[1] ? mod(t, T) : t
#     β₀ = β_nn([mod(0.5 + (t / T), 1.0f0), influ_perc(τ) / 10.0f0, rhino_perc(τ) / 10.0f0], nn_params)[1]
#     inlockdown = (t >= T * 3.3 && t <= T * 3.8)
#     schoolopen = (mod(t, T) < 0.553 * T || mod(t, T) > 0.668 * T)
#     β = 1.5 * β₀ * ct_lkd(t, β_LD)
#     β_s = β_school * (1 - inlockdown) * schoolopen * ct_lkd(t, β_LD)
#     λ = (β * (sum(I₁) + ι * sum(I₂)) / N) + ϵ * (1 - inlockdown)#This assumes homogeneous mixing proportional to the size of each group
#     λ_schools = β_s * (I₁[5] + ι * I₂[5]) / sum(u[5, :, :])  #School mixing
#     #Transmission Dynamics
#     du[:, 1, 1] .= -S₁ .* λ
#     du[:, 1, 2] .= -σ .* S₂ .* λ .+ ω .* (R₁ .+ R₂)
#     du[:, 2, 1] .= S₁ .* λ .- γ .* I₁
#     du[:, 2, 2] .= σ .* S₂ .* λ .- γ .* I₂
#     du[:, 3, 1] .= γ .* I₁ .- ω .* R₁
#     du[:, 3, 2] .= γ .* I₂ .- ω .* R₂
#     #Add schools based transmission
#     du[5, 1, 1] += -S₁[5] * λ_schools
#     du[5, 2, 1] += S₁[5] * λ_schools
#     du[5, 1, 2] += -σ * S₂[5] * λ_schools
#     du[5, 2, 2] += σ * S₂[5] * λ_schools
#     #Add Demography
#     du[1, 1, 1] += B # Births with maternal immunity
#     du[end, :, :] .+= -μ .* u[end, :, :] #Deaths in the final age group
#     #Add Aging
#     du .+= aging_rates(aging_mat, u)
#     return nothing
# end

## Default parameter choices

β_school = 0.1f0
β_LD = 0.3
β_base = 0.5f0
amp = 0.2f0
phase = 0.0f0
ϕ_log = 0.0f0
# p₀_nn = [[β_school, β_LD, α, γ, σ, ι, B, N, μ, ϵ]; p_init] .|> Float32
p₀_nn = [[β_school, β_LD, α, γ, σ, ι, B, N, μ, ϵ]; fit_snake.u[2:end]] .|> Float32

## Test vector field function

u0 = zeros(Float32, 6, 3, 2)
u0[1:3, 1, 1] .= proportion_of_population[1:3] .* N
u0[4:end, 3, 2] .= proportion_of_population[4:end] .* N
u0[:, 2, 2] .+= 1e4 #Add some infected

du = similar(u0)
@time f_RSV_nn(du, u0, p₀_nn, 0.0f0)

##

tspan = (Float32(-3136), Float32(ts[end]))
prob_RSV_nn = ODEProblem(f_RSV_nn, u0, tspan, p₀_nn)

@time sol = solve(prob_RSV_nn, Tsit5(); saveat=7);
# @time sol = solve(prob_RSV_nn, AutoTsit5(Rodas5()); saveat=ts);
# @time sol = solve(prob_RSV_nn, AutoTsit5(Rosenbrock23()); saveat=ts);

pred_hosps = 7.0 .* γ .* [sum(u[:, 2, 1] .* dis_risk₁ .* hosp_risk .+ u[:, 2, 2] .* dis_risk₂ .* hosp_risk) for u in sol.u] .* (100_000 / N)
plot(pred_hosps)

## Optimisation problem -- set up initial condition arrays

u0 = zeros(Float32, 6, 3, 2)
u0[1, 1, 1] = proportion_of_population[1] .* N
u0[2, 1, 1] = 0.75 * proportion_of_population[2] .* N
u0[2, 1, 2] = 0.25 * proportion_of_population[2] .* N
u0[3, 1, 1] = 0.5 * proportion_of_population[3] .* N
u0[3, 1, 2] = 0.5 * proportion_of_population[3] .* N
u0[4, 1, 1] = 0.1 * proportion_of_population[4] .* N
u0[4, 1, 2] = 0.9 * proportion_of_population[4] .* N
u0[5:end, 1, 2] .= proportion_of_population[5:end] .* N
u0 = u0 * 0.99f0
u0[:, 2, 2] .+= 0.01f0 * proportion_of_population .* N
sum(u0) ≈ N

change_mat = zeros(Float32, 6, 3, 2)
change_mat[:, 3, 2] .= proportion_of_population .* N

##

function loss_func_nn(θ, prob, hosp_data, sensealg)
    β_school_logit, β_LD_logit, p_R_logit, obs_scale_log, risk_rhino_log, risk_flu_log = θ[1:6]
    β_school = 1.5 * sigmoid(β_school_logit)
    β_LD = sigmoid(β_LD_logit)
    p_R = sigmoid(p_R_logit)
    obs_scale = [exp(obs_scale_log + risk_rhino_log * rhino_perc(t) + risk_flu_log * influ_perc(t)) for t in ts]

    _p = [[β_school, β_LD, α, γ, σ, ι, B, N, μ, ϵ]; θ[7:end]]
    _u0 = u0 .* (1 - p_R) .+ p_R .* change_mat

    _prob = remake(prob; p=_p, u0=_u0)
    sol = solve(_prob, Tsit5(); saveat=ts, sensealg=sensealg)
    failure = length(sol.u) < length(ts)

    if failure
        return Inf, zeros(100)
    else
        pred_hosps = 7.0 * γ * obs_scale .* [sum(u[:, 2, 1] .* dis_risk₁ .* hosp_risk .+ u[:, 2, 2] .* dis_risk₂ .* hosp_risk) for u in sol.u] .* (100_000 / N)
        pred_1_seropos = [(sum(u[3, :, 2]) + sum(u[3, 2:3, 1])) / (sum(u[3, :, :])) for u in sol.u[[24, 46, 67]]]
        pred_2plus_seropos = [(sum(u[4, :, 2]) + sum(u[4, 2:3, 1])) / (sum(u[4, :, :])) for u in sol.u[[24, 46, 67]]]
        loss = sum(abs, pred_hosps .- hosp_data) / 0.25 + sum(abs, pred_1_seropos .- 0.5) / 0.1 + sum(abs, pred_2plus_seropos .- 0.9) / 0.1

        return loss + reg, pred_hosps
        # return sum(abs, pred_hosps .- hosp_data), pred_hosps
    end
end

@time l, pred = loss_func_nn([[0.0f0, 0.0f0, -1.0f0, 0.0f0, 0.0f0, 0.0f0]; p_init], prob_RSV_nn, rsv_hosp_rate, ForwardDiffSensitivity())
plot(pred)
# @time grad_fd_fs = ForwardDiff.gradient(θ -> loss(θ, prob_RSV, rsv_hosp_rate, ForwardSensitivity()), [β_school; p_init])
# @time grad_fd_fds = ForwardDiff.gradient(θ -> loss_func_seas(θ, prob_RSV, rsv_hosp_rate, ForwardDiffSensitivity())[1], [β_base, amp, phase, β_school, β_LD, 0.001f0])# ~28 secs

# @time grad_zy_fds = Zygote.gradient(θ -> loss_func_seas(θ, prob_RSV, rsv_hosp_rate, ForwardDiffSensitivity())[1], [β_base, amp, phase, β_school, β_LD, 0.001f0]) #~3.1secs
# @time grad_zy_fds = Zygote.gradient(θ -> loss_func_nn(θ, prob_RSV_nn, rsv_hosp_rate, ForwardDiffSensitivity())[1], [[β_school, β_LD, 0.001f0]; p_init]) #~3.1secs

# # @time grad_zy_fs = Zygote.gradient(θ -> loss(θ, prob_RSV, rsv_hosp_rate, ForwardSensitivity()), [β_school; p_init])
# @time grad_zy_bs = Zygote.gradient(θ -> loss(θ, prob_RSV, rsv_hosp_rate, BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))), [β_school; p_init]) #unstable
# @time grad_zy_inter = Zygote.gradient(θ -> loss(θ, prob_RSV, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))), [β_school; p_init]) #~4.4 secs
# @time grad_zy_inter_false = Zygote.gradient(θ -> loss(θ, prob_RSV, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))), [β_school; p_init]) #~4.4secs
# @time grad_zy_quad = Zygote.gradient(θ -> loss(θ, prob_RSV, rsv_hosp_rate, QuadratureAdjoint(autojacvec=ReverseDiffVJP(false))), [β_school; p_init]) #slow, looks unstable



## Declare 

function plot_hosps(θ, l, pred)
    println("loss = ", l)
    plt = plot(ts, pred, lab="pred")
    scatter!(plt, ts, rsv_hosp_rate, ms=3, lab="data", legend=:topleft)
    display(plt)
    return false
end

## Run fit
fit = DiffEqFlux.sciml_train(θ -> loss_func_nn(θ, prob_RSV_nn, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))),
    [[0.0f0, 0.0f0, -1.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0]; fit_snake.u[2:end]],
    callback=plot_hosps)
##
fit2 = DiffEqFlux.sciml_train(θ -> loss_func_nn(θ, prob_RSV_nn, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))),
    fit.u,
    callback=plot_hosps)
##
fit3 = DiffEqFlux.sciml_train(θ -> loss_func_nn(θ, prob_RSV_nn, rsv_hosp_rate, ForwardDiffSensitivity()),
    fit2.u,
    callback=plot_hosps)


## Save fit

@save("rsv_model_fit_other.jld2", fit2)

## Some plots about the fit

β_school_logit, β_LD_logit, p_R_logit, obs_scale_log, risk_rhino_log, risk_flu_log = fit2.u[1:6]
p_nn_fit = fit2.u[7:end]
β_school_fit = 1.5 * sigmoid(β_school_logit)
β_LD_fit = sigmoid(β_LD_logit)
p_R = sigmoid(p_R_logit)
obs_scale_fit = [exp(obs_scale_log + risk_rhino_log * rhino_perc(t) + risk_flu_log * influ_perc(t)) for t in -1500:2000]
plot(obs_scale_fit)
β₀_fit = [1.5 * β_nn([mod((t / T), 1), influ_perc(t) / 10, rhino_perc(t) / 10], p_nn_fit)[1] * ct_lkd(t, β_LD_fit) for t in -1500:2000]
plot(-1500:2000, β₀_fit)
vline!([T * (i) for i = -5:5])

β_sch = [(1 - (t >= T * 3.3 && t <= T * 3.8)) * (mod(t / T, 1) < 0.553 || mod(t / T, 1) > 0.668) * β_school_fit * ct_lkd(t, β_LD_fit) for t = -1500:2000]
plot(-1500:2000, β_sch, lab="")

plot(-1500:2000, β_sch .+ β₀_fit, lab="")

β₀_rhino = [β_nn([0, 0 / 10, r / 10], p_nn_fit)[1] for r = 0:0.01:1]
plot(0:0.01:1, β₀_rhino)

β_rhino_flu = [1.5 * β_nn([0, f / 10, r / 10], p_nn_fit)[1] for f = 0:0.01:1, r = 0:0.01:1]
heatmap(0:1:100, 0:1:100, β_rhino_flu,
    xlabel="Rhinovirus %",
    ylabel="Flu %")

@time l, pred = loss_func_nn(fit2.u, prob_RSV_nn, rsv_hosp_rate, ForwardDiffSensitivity())

## Fit to RSV hosp
ref_date = Date(2017, 1, 1)
xticks_pos = [(Date(y, 1, 1) - ref_date).value for y = 2014:2022]
xticks_labs = 2013:2022
scatter(ts, rsv_hosp_rate,
    lab="Data",
    xticks=(xticks_pos, xticks_labs),
    ylabel="Weekly hospitalisation rate",
    xlabel="Year",
    title="Sari-watch RSV reporting vs Model",
    size=(800, 500), dpi=200,
    left_margin=5mm,
    guidefont=14, tickfont=11, titlefont=18)
plot!(ts, pred, lw=3,
    lab="model")

# savefig("RSV_fit_to_data.png")
## Tranmission rates
β₀_fit = [1.5 * β_nn([mod((t / T), 1), influ_perc(t) / 10, rhino_perc(t) / 10], p_nn_fit)[1] * ct_lkd(t, β_LD_fit) for t in -1500:2000]
β_sch = [(1 - (t >= T * 3.3 && t <= T * 3.8)) * (mod(t / T, 1) < 0.553 || mod(t / T, 1) > 0.668) * β_school_fit * ct_lkd(t, β_LD_fit) for t = -1500:2000]
plot(-1500:2000, β_sch .+ β₀_fit, lab="",
    xticks=(xticks_pos, xticks_labs),
    ylabel="beta(t,flu_perc,rhino_perc)",
    xlabel="Year",
    title="Model inferred transmission rates",
    size=(800, 500), dpi=200,
    left_margin=5mm,
    guidefont=14, tickfont=11, titlefont=18)
# savefig("RSV_transmission_ovr_time.png")

##
β_rhino_flu = [β_nn([0.0, f / 10, r / 10], p_nn_fit)[1] / β_nn([0.0, 0, 0], p_nn_fit)[1] for f = 0:0.01:0.3, r = 0:0.01:0.3]
heatmap(0:1:30, 0:1:30, β_rhino_flu,
    xlabel="Rhinovirus %",
    ylabel="Flu %",
    title="Effect of other viruses on transmission (Jan 1st)",
    size=(800, 500), dpi=200,
    left_margin=5mm, right_margin=7mm,
    guidefont=14, tickfont=11, titlefont=18)
# savefig("Other_viruses_transmission.png")
##
rel_risk_rhino_flu = [exp(risk_rhino_log * r + risk_flu_log * f) for f = 0:0.01:0.3, r = 0:0.01:0.3]
heatmap(0:1:30, 0:1:30, rel_risk_rhino_flu,
    xlabel="Rhinovirus %",
    ylabel="Flu %",
    title="Effect of other viruses on risk of hospitalisation",
    size=(800, 500), dpi=200,
    left_margin=5mm, right_margin=5mm,
    guidefont=14, tickfont=11, titlefont=18)

# savefig("Other_viruses_risk.png")
##

plot(datamart.t, (datamart.Influenza_A_perc .+ datamart.Influenza_B_perc), lab="Flu",
    xticks=(xticks_pos, xticks_labs), lw=3,
    ylabel="% test pos",
    xlabel="Year",
    title="Datamart % positivity",
    size=(800, 500), dpi=200,
    left_margin=5mm,
    guidefont=14, tickfont=11, titlefont=18)
plot!(datamart.t, (datamart.Rhinovirus_perc), lab="Rhinovirus", lw=3)
savefig("flu_rhino.png")

## testing learning directly --- using semi-periodic layers
a = 1

incidence_nn = FastChain((x, p) -> x,
    FastDense(3, 64, x -> x .+ (1 / a) .* cos.(a .* x) .^ 2),
    FastDense(64, 64, x -> x .+ (1 / a) .* cos.(a .* x) .^ 2),
    FastDense(64, 1, x -> x .+ (1 / a) .* cos.(a .* x) .^ 2))
snake_p₀ = initial_params(incidence_nn)
# @time incidence_nn([0,0,0],snake_p₀)[1]

function snake_model(x, p, a)
    incidence_nn = FastChain((x, p) -> x,
        FastDense(3, 64, x -> x .+ (1 / a) .* cos.(a .* x) .^ 2),
        FastDense(64, 64, x -> x .+ (1 / a) .* cos.(a .* x) .^ 2),
        FastDense(64, 1, x -> x .+ (1 / a) .* cos.(a .* x) .^ 2))
    return incidence_nn(x, p)
end

θ₀ = [log(0.56f0); snake_p₀]
@time snake_model(rand(1), snake_p₀, exp(θ₀[1]))
# pred = [snsnake_model(x, p, a)ake_model([t/T,influ_perc(t)* 10.0f0, rhino_perc(t) * 10.0f0],snake_p₀,1f0)[1] for t in ts]

ts2 = ts[rsv_hosp_rate.>0]
rsv_hosp_rate2 = rsv_hosp_rate[rsv_hosp_rate.>0]

function loss_snake(θ)
    a = exp(θ[1])
    p = θ[2:end]
    pred = [snake_model([t / T, influ_perc(t), rhino_perc(t)], p, a)[1] for t in ts2[1:190]]
    sum(abs2, pred .- log.(rsv_hosp_rate2[1:190])) + (a - 2 * pi)^2, pred
end

loss_snake(θ₀)
function plot_snake(x, l, pred)
    plt = scatter(ts2[1:190], log.(rsv_hosp_rate2[1:190]), lab="Data", legend=:topleft)
    plot!(plt, ts2[1:190], pred[1:190], lab="pred")
    display(plt)
    println("loss = $(l)")
    return false
end

## 
fit_snake = DiffEqFlux.sciml_train(loss_snake, θ₀, callback=plot_snake)
ts_extrap = [ts; ts[end]+7:7:7*(150)]
log_inc_extrap = [snake_model([t / T, influ_perc(t), rhino_perc(t)], fit_snake.u[2:end], exp(fit_snake.u[1]))[1] for t in ts_extrap]
log_inc_extrap = [incidence_nn_fitted([t / T, influ_perc(t), rhino_perc(t)], fit_snake.u[2:end])[1] for t in ts_extrap]

plot(ts_extrap,1.5*sigmoid.(log_inc_extrap))
scatter(ts, log.(rsv_hosp_rate))
plot!(ts_extrap, log_inc_extrap)

vline!(ts[rsv_hosp_rate.==0])
vline!([ts2[190]])
