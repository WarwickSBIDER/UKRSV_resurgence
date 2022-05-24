cd("./RSV_resurgence/")
using Pkg
Pkg.activate(".")
##

using OrdinaryDiffEq, StatsPlots, Statistics, CSV, FileIO, DataFrames, Dates, LinearAlgebra, DiffEqFlux, Tullio
using DataInterpolations, DiffEqSensitivity, ForwardDiff, Zygote, JLD2, Plots.PlotMeasures, RCall
using Distributions

##Declare risk per infection

dis_risk₁ = [mean([31.2, 28.6]); mean([20.0, 13.0]); fill(2.0, 4)] / 100
dis_risk₂ = [fill(5.0, 5); 0.0] / 100
hosp_risk = [mean([32.76, 33.07, 21.9, 20.74, 18.86, 12, 27]), mean([9.4, 10.76, 9.1, 12.11, 9, 8.7, 6.7]),
    mean([6.7, 7.11, 7.78, 7.43, 4.13, 4.1, 10.52, 20, 14.84, 13.95, 10.18, 2.36, 8.41]),
    mean([3.76, 1.08, 0.19]), 0.0, 0.0] / 100

## Load data

## Load contact matrices from Prem and Jit 

R"""
library('socialmixr')
data("polymod")
ct_plymd_info = socialmixr::contact_matrix(polymod, countries = "United Kingdom", age.limits = c(0, 1,2, 4, 19), symmetric = TRUE)
ct_plymd_info2 = socialmixr::contact_matrix(polymod, countries = "United Kingdom", age.limits = seq(0,75,5), symmetric = TRUE)

library('contactdata')
pop_data = contactdata::age_df_countries(c("UK"))
ct_all = contactdata::contact_matrix('UK', location = c("all"))
ct_school = contactdata::contact_matrix('UK', location = c("school"))
# ct_plymd_info = contact_matrix(polymod, countries = "United Kingdom", age.limits = c(0, 1,2, 4, 14), symmetric = TRUE)
"""
ct_all_fromto = @rget ct_all
ct_school_fromto = @rget ct_school
ct_plymd_dict = @rget ct_plymd_info
ct_plymd2_dict = @rget ct_plymd_info2

pop_data = @rget pop_data

proportion_of_population = [fill(ct_plymd_dict[:demography].proportion[1] / 2, 2); ct_plymd_dict[:demography].proportion[2:5]] .|> Float32
proportion_of_population_5yrs = ct_plymd2_dict[:demography].proportion

# ct_ploy = ct_plymd_dict[:matrix]

## Define conversion matrices from Prem 16 age groups to 6 age groups in this model

# Zygote.gradient(x -> sum(abs2,x),[1.0,2.0])

##
P_Aa = zeros(6, 16)
P_Aa[1, 1] = 1
P_Aa[2, 1] = 1
P_Aa[3, 1] = 1
P_Aa[4, 1] = 1
P_Aa[5, 2] = 1 / 3
P_Aa[5, 3] = 1 / 3
P_Aa[5, 4] = 1 / 3
P_Aa[6, 5:end] = proportion_of_population_5yrs[5:end] ./ sum(proportion_of_population_5yrs[5:end])

##
P_bB = zeros(16, 6)
P_bB[1, 1] = 0.5 / 5
P_bB[1, 2] = 0.5 / 5
P_bB[1, 3] = 1 / 5
P_bB[1, 4] = 3 / 5
P_bB[2:4, 5] .= 1.0
P_bB[5:end, 6] .= 1.0

## Create normalized mixing matrices with leading eigen value 1
#Split 0-1 into 0-6 mnth and 7-12 month


ct_all_un = Matrix((P_Aa * ct_all_fromto * P_bB)')
ct_school_un = Matrix((P_Aa * ct_school_fromto * P_bB)')
ct_other_un = ct_all_un .- ct_school_un

max_eigval_ct = maximum(abs.(eigvals(ct_all_un)))
ct_mat_other = ct_other_un ./ max_eigval_ct .|> Float32
ct_mat_schools = ct_school_un ./ max_eigval_ct .|> Float32
##
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


## Declare fixed parameters

γ₁ = 40.6 / T# mean 9 day infectious period
γ₂ = 93.7 / T# mean 3.9 day infectious period
ω = 2 / T #6 months before return to potential susceptibility
σ = 0.65 #Reduction in susceptibility after first episode
ι = 0.25 #Reduction in transmissibility during subsequent episodes
N = 50e6#Pop size England
L = 80 * T #Mean life expectancy
μ = 1 / (L - 14 * T) #mortality rate assuming no mortality in first 14 years of life
B = μ * (N * proportion_of_population[end])#Daily Replacement rate of births: 631k per annum in this model
ϵ = 5e-6 #~250 RSV infectious contacts from outside UK per day when not in measures
N_vec = N .* proportion_of_population .|> Float32
## Declare model

l_ages = [0.5 * T, 0.5 * T, T, 3 * T, 15 * T, Inf]
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

##

n_wk = 278 + 165
logit_wk₀ = zeros(Float32, n_wk)


# function h(t)
#     τ = t < ts[1] ? mod(t, T) : t
#     logit_wk₀[min(floor(Int64, (τ - ts[1]) / 7) + 1, n_wk)]
# end
# plot(h, ts[1] - 1000, ts[end] + 600, lab="")
# vline!([ts[1]])
T = 365.25f0

function f_RSV_wk(du, u, p, t)
    #Gather parameters and states
    β_school, β_LD, γ₀, γ₁, σ, ι, B, N, μ, ϵ = p[1:10]
    β_wk = p[(10+1):(10+n_wk)] #Weekly continuously varying transmission rates

    S₁ = @view u[:, 1, 1]
    S₂ = @view u[:, 1, 2]
    I₁ = @view u[:, 2, 1]
    I₂ = @view u[:, 2, 2]
    R₁ = @view u[:, 3, 1]
    R₂ = @view u[:, 3, 2]

    #Time varying infection rates
    τ = t < ts[1] ? mod(t, T) : t
    inlockdown = (t >= T * 3.3 && t <= T * 3.8)
    schoolopen = (mod(t, T) < 0.553 * T || mod(t, T) > 0.668 * T)
    β₀ = β_wk[min(floor(Int64, (τ - ts[1]) / 7) + 1, n_wk)]# * β_viruses([influ_perc(τ), rhino_perc(τ)], nn_params)[1]
    β_s = β_school * (1 - inlockdown) * schoolopen * ct_lkd(t, β_LD)
    # λ = ((β₀ .* ct_mat_other + β_s .* ct_mat_schools) * (I₁ .+ ι .* I₂)) ./ N_vec .+ ϵ * (1 - inlockdown)#This assumes homogeneous mixing proportional to the size of each group
    λ =  (β₀ * ct_mat_other .+ β_s * ct_mat_schools) * (I₁ .+ ι .* I₂) ./ N_vec #This assumes homogeneous mixing proportional to the size of each group

    # λ = (β₀ * (sum(I₁) + ι * sum(I₂)) / N) + ϵ * (1 - inlockdown)
    # λ_schools = β_s .* (ct_mat_schools * (I₁ .+ ι .* I₂)) ./ N_vec  #School mixing
    #Transmission Dynamics
    du[:, 1, 1] .= -S₁ .* (λ .+ ϵ * (1 - inlockdown))
    du[:, 1, 2] .= -σ .* S₂ .* (λ .+ ϵ * (1 - inlockdown)) .+ ω .* (R₁ .+ R₂)
    du[:, 2, 1] .= S₁ .* (λ .+ ϵ * (1 - inlockdown)) .- γ₁ .* I₁
    du[:, 2, 2] .= σ .* S₂ .* (λ .+ ϵ * (1 - inlockdown)) .- γ₂ .* I₂
    du[:, 3, 1] .= γ₁ .* I₁ .- ω .* R₁
    du[:, 3, 2] .= γ₂ .* I₂ .- ω .* R₂
    #Add Demography
    du[1, 1, 1] += B # Births with maternal immunity
    du[end, :, :] .+= -μ .* u[end, :, :] #Deaths in the final age group
    #Add Aging
    du .+= aging_rates(aging_mat, u)
    return nothing
end

## Default parameter choices

β_school = 1.0f0
β_LD = 0.3
β_base = 1.0f0
amp = 0.2f0
phase = 0.0f0
ϕ_log = 0.0f0
wk₀ = ones(Float32, n_wk)
# p₀_nn = [[β_school, β_LD, α, γ, σ, ι, B, N, μ, ϵ]; p_init] .|> Float32
# p₀ = [[β_school, β_LD, α, γ, σ, ι, B, N, μ, ϵ]; logit_wk₀; p_init_viruses]
p₀ = [[β_school, β_LD, γ₁, γ₂, σ, ι, B, N, μ, ϵ]; wk₀]

## Test vector field function

u0 = zeros(Float32, 6, 3, 2)
u0[1:3, 1, 1] .= proportion_of_population[1:3] .* N
u0[4:end, 3, 2] .= proportion_of_population[4:end] .* N
u0[:, 2, 2] .+= 1e4 #Add some infected

du = similar(u0)
@time f_RSV_wk(du, u0, p₀, 0.0f0)

# S₁ = @view u0[:, 1, 1]
# S₂ = @view u0[:, 1, 2]
# I₁ = @view u0[:, 2, 1]
# I₂ = @view u0[:, 2, 2]
# R₁ = @view u0[:, 3, 1]
# R₂ = @view u0[:, 3, 2]

# λ =  (ct_mat_other .+ ct_mat_schools) * (I₁ .+ ι .* I₂) ./ N_vec #This assumes homogeneous mixing proportional to the size of each group



##some analysis
function leading_eval(β₀, β_s)
    maximum(abs.(eigvals(β₀ * ct_mat_schools + β_s * ct_mat_schools)))
end

rhos = [leading_eval(β₀, β_s) for β₀ in 0:0.01:3, β_s in 0:0.01:3]
heatmap(0:0.01:3, 0:0.01:3, rhos)

##

tspan = (Float32(-3136), Float32(ts[end]))
prob_RSV_wk = ODEProblem(f_RSV_wk, u0, tspan, p₀)

@time sol = solve(prob_RSV_wk, Tsit5(); saveat=7);
# @time sol = solve(prob_RSV_nn, AutoTsit5(Rodas5()); saveat=ts);
# @time sol = solve(prob_RSV_nn, AutoTsit5(Rosenbrock23()); saveat=ts);

pred_hosps = 7.0 .* [γ₁ .* sum(u[:, 2, 1] .* dis_risk₁ .* hosp_risk .+ γ₂ .* u[:, 2, 2] .* dis_risk₂ .* hosp_risk) for u in sol.u] .* (100_000 / N)
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

function loss_func_wk(θ, prob, hosp_data, sensealg)
    β_school_logit, β_LD_logit, p_R_logit, obs_scale_log, risk_rhino_log, risk_flu_log = θ[1:6]
    # p_R_logit, obs_scale_log, risk_rhino_log, risk_flu_log = θ[1:4]
    logit_wk = θ[(6+1):(6+n_wk)]
    # prec_logdiff_wk = exp(log_prec_logdiff_wk)
    # prec_hosprate = exp(log_prec_hosprate)
    prec_logdiff_wk = 1/0.1^2
    prec_hosprate= 1 / 0.25
    # p_v = θ[(6+n_wk+1):(6+n_wk+n_v)]
    β_school = 3.0 * sigmoid(β_school_logit)
    β_LD = sigmoid(β_LD_logit)
    p_R = sigmoid(p_R_logit)
    wks = 3.0 * sigmoid.(logit_wk)
    obs_scale = [exp(obs_scale_log + risk_rhino_log * rhino_perc(t) + risk_flu_log * influ_perc(t)) for t in ts]

    _p = [[β_school, β_LD, γ₁, γ₂, σ, ι, B, N, μ, ϵ]; wks]
    _u0 = u0 .* (1 - p_R) .+ p_R .* change_mat

    _prob = remake(prob; p=_p, u0=_u0)
    sol = solve(_prob, Tsit5(); saveat=ts, sensealg=sensealg)
    failure = length(sol.u) < length(ts)
    pred_hosps = 7.0 * obs_scale .* [sum(γ₁ .* u[:, 2, 1] .* dis_risk₁ .* hosp_risk .+ γ₂ * u[:, 2, 2] .* dis_risk₂ .* hosp_risk) for u in sol.u] .* (100_000 / N)
    pred_1_seropos = [(sum(u[3, :, 2]) + sum(u[3, 2:3, 1])) / (sum(u[3, :, :])) for u in sol.u[[24, 46, 67]]]
    pred_2plus_seropos = [(sum(u[4, :, 2]) + sum(u[4, 2:3, 1])) / (sum(u[4, :, :])) for u in sol.u[[24, 46, 67]]]
    loss = sum(abs, pred_hosps .- hosp_data) * prec_hosprate
    loss += sum([-logpdf(Beta(50,50),pred) for pred in pred_1_seropos])
    loss += sum([-logpdf(Beta(90,10),pred) for pred in pred_2plus_seropos])
    reg = sum(abs2, diff(log.(wks))) * prec_logdiff_wk * 0.5 
    reg += 0.5 * sum(abs2, [obs_scale_log, risk_rhino_log, risk_flu_log])
    reg += -logpdf(Gamma(20,10/20),0.5* prec_hosprate^2)
    reg += -logpdf(Gamma(20,100/20), prec_logdiff_wk)
    return loss + reg, pred_hosps
    # if failure
    #     return Inf, zeros(100)
    # else
    #     pred_hosps = 7.0 * obs_scale .* [γ₁ .* sum(u[:, 2, 1] .* dis_risk₁ .* hosp_risk .+ γ₂ * u[:, 2, 2] .* dis_risk₂ .* hosp_risk) for u in sol.u] .* (100_000 / N)
    #     pred_1_seropos = [(sum(u[3, :, 2]) + sum(u[3, 2:3, 1])) / (sum(u[3, :, :])) for u in sol.u[[24, 46, 67]]]
    #     pred_2plus_seropos = [(sum(u[4, :, 2]) + sum(u[4, 2:3, 1])) / (sum(u[4, :, :])) for u in sol.u[[24, 46, 67]]]
    #     loss = sum(abs, pred_hosps .- hosp_data) * prec_hosprate
    #     loss += sum([-logpdf(Beta(50,50),pred) for pred in pred_1_seropos])
    #     loss += sum([-logpdf(Beta(90,10),pred) for pred in pred_2plus_seropos])
    #     reg = sum(abs2, diff(log.(wks))) * prec_logdiff_wk * 0.5 
    #     reg += 0.5 * sum(abs2, [obs_scale_log, risk_rhino_log, risk_flu_log])
    #     reg += -logpdf(Gamma(4,1/4),0.5* prec_hosprate^2)
    #     reg += -logpdf(Gamma(4,100/4), prec_logdiff_wk)
    #     return loss + reg, pred_hosps
    #     # return sum(abs, pred_hosps .- hosp_data), pred_hosps
    # end
end

logit_wk₀ =  rand(Float32, n_wk)
##
# [[β_school, β_LD, α, γ, σ, ι, B, N, μ, ϵ]; logit_wk₀; p_init_viruses]
# @time l, pred = loss_func_wk([[0.0f0, 0.0f0, -1.0f0, 0.0f0, 0.0f0, 0.0f0]; logit_wk₀; p_init_viruses], prob_RSV_wk, rsv_hosp_rate, ForwardDiffSensitivity())
θ₀ = [[-0.0,0.0,-1.0, 0.0f0, 0.0f0, 0.0f0]; logit_wk₀] .|> Float64
@time l, pred = loss_func_wk(θ₀, prob_RSV_wk, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
# @time loss_func_wk(θ₀, prob_RSV_wk, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
plot(pred)
# 
##
Zygote.gradient(θ -> loss_func_wk(θ, prob_RSV_wk, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))[1],
    θ₀)


## Declare Optimisation problem and callback plotting

function plot_hosps(θ, l, pred)
    println("loss = ", l)
    # σ_hosprate = 1/exp(θ[6])
    σ_hosprate = sqrt(2)*0.25
    plt = plot(ts, pred,ribbon = σ_hosprate,color = :red, lab="pred",lw = 3, fillalpha = 0.3)
    scatter!(plt, ts, rsv_hosp_rate, ms=3, lab="data", legend=:topleft)
    display(plt)
    return false
end

## Run fit
# logit_wk₀ = fit.u[(6+1):(6+n_wk)]
fit = DiffEqFlux.sciml_train(θ -> loss_func_wk(θ, prob_RSV_wk, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))),
    θ₀,
    callback=plot_hosps)
##
fit2 = DiffEqFlux.sciml_train(θ -> loss_func_wk(θ, prob_RSV_wk, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))),
    fit.u,
    callback=plot_hosps)
##
θ₀ = fit2.u[3:end]
fit3 = DiffEqFlux.sciml_train(θ -> loss_func_wk(θ, prob_RSV_wk, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))),
    θ₀,
    callback=plot_hosps)

##
@save("fit_wk_mat.jld2", fit)

# fit3 = DiffEqFlux.sciml_train(θ -> loss_func_nn(θ, prob_RSV_nn, rsv_hosp_rate, ForwardDiffSensitivity()),
#     fit2.u,
#     callback=plot_hosps)

##
exp.(fit.u[(end-2):end])
β_wk_fit =  3.0* sigmoid.(fit.u[(6+1):(6+n_wk)])
β_school_fit = 3.0 * sigmoid(fit.u[1])
β_LD_fit = sigmoid(fit.u[2])

scatter(β_wk_fit, ms=3, color=:black)
vline!([i * 52 for i = 0:8])

β_sch = [(1 - (t >= T * 3.3 && t <= T * 3.8)) * (mod(t / T, 1) < 0.553 || mod(t / T, 1) > 0.668) * β_school_fit * ct_lkd(t, β_LD_fit) for t = -1500:2000]

β₀ = [β_wk_fit[min(floor(Int64, (t < ts[1] ? mod(t, T) : t - ts[1]) / 7) + 1, n_wk)] for t in -1500:2000]
scatter(β_sch + β₀)

## Fit to RSV hosp
function make_pred(θ, prob, new_ts)
    β_school_logit, β_LD_logit, p_R_logit, obs_scale_log, risk_rhino_log, risk_flu_log = θ[1:6]
    logit_wk = θ[(6+1):(6+n_wk)]
    # p_v = θ[(6+n_wk+1):(6+n_wk+n_v)]
    β_school = 1.5 * sigmoid(β_school_logit)
    β_LD = sigmoid(β_LD_logit)
    p_R = sigmoid(p_R_logit)
    obs_scale = [exp(obs_scale_log + risk_rhino_log * rhino_perc(t) + risk_flu_log * influ_perc(t)) for t in new_ts]

    # _p = [[β_school, β_LD, α, γ, σ, ι, B, N, μ, ϵ]; logit_wk; p_v]
    _p = [[β_school, β_LD, α, γ, σ, ι, B, N, μ, ϵ]; logit_wk]
    _u0 = u0 .* (1 - p_R) .+ p_R .* change_mat

    _prob = remake(prob; p=_p, u0=_u0)
    sol = solve(_prob, Tsit5(); saveat=new_ts)
    pred_hosps = 7.0 * γ * obs_scale .* [sum(u[:, 2, 1] .* dis_risk₁ .* hosp_risk .+ u[:, 2, 2] .* dis_risk₂ .* hosp_risk) for u in sol.u] .* (100_000 / N)
    # pred_1_seropos = [(sum(u[3, :, 2]) + sum(u[3, 2:3, 1])) / (sum(u[3, :, :])) for u in sol.u[[24, 46, 67]]]
    # pred_2plus_seropos = [(sum(u[4, :, 2]) + sum(u[4, 2:3, 1])) / (sum(u[4, :, :])) for u in sol.u[[24, 46, 67]]]
    # loss = sum(abs, pred_hosps .- hosp_data) / 0.25 + sum(abs, pred_1_seropos .- 0.5) / 0.1 + sum(abs, pred_2plus_seropos .- 0.9) / 0.1
    # reg = sum(abs2, diff(logit_wk)) / 0.1
    return pred_hosps
end

pred = make_pred(fit2.u, prob_RSV_wk, ts[1]:7:tspan[end])

ref_date = Date(2017, 1, 1)
xticks_pos = [(Date(y, 1, 1) - ref_date).value for y = 2014:2022]
xticks_labs = 2014:2022
scatter(ts, rsv_hosp_rate,
    lab="Data", color=:black,
    xticks=(xticks_pos, xticks_labs), marker=:x,
    ylabel="Weekly hospitalisation rate",
    xlabel="Year",
    title="Sari-watch RSV reporting vs Model",
    size=(800, 500), dpi=200,
    left_margin=5mm,
    guidefont=14, tickfont=11, titlefont=18)
plot!(ts[1]:7:tspan[end], pred, lw=3, color=:red, alpha=0.6,
    lab="model")
# savefig("RSV_fit_to_data.png")

##
logit_β_wk_fit = fit.u[(6+1):(6+n_wk)]
β_wk_fit = 2.0 .* sigmoid.(logit_β_wk_fit)
β_school_fit = 2.0 * sigmoid(fit.u[1])
β_LD_fit = sigmoid(fit.u[2])

function day_to_week_β(t)
    τ = t < ts[1] ? mod(t, T) : t
    β_wk_fit[min(floor(Int64, (τ - ts[1]) / 7) + 1, n_wk)]
end

β₀_fit = map(day_to_week_β, -1500:2000)
# plot(-1500:2000, β₀_fit)

β_sch = [(1 - (t >= T * 3.3 && t <= T * 3.8)) * (mod(t / T, 1) < 0.553 || mod(t / T, 1) > 0.668) * β_school_fit * ct_lkd(t, β_LD_fit) for t = -1500:2000]

plot(-1500:2000, β₀_fit, lab="",
    xticks=(xticks_pos, xticks_labs),
    ylabel="beta(t)",
    xlabel="Year",
    title="Model inferred transmission rates",
    size=(800, 500), dpi=200,
    left_margin=5mm,
    guidefont=14, tickfont=11, titlefont=18)
# savefig("RSV_transmission_ovr_time_no_schools.png")

##
risk_rhino_log, risk_flu_log = fit.u[5:6]

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
plot!(datamart.t, (datamart.RSV_perc), lab="RSV", lw=1, ls=:dot, color=:black)

# savefig("flu_rhino.png")


## Neural network fit to time varying beta
ts_nn_fit = ts[1]:7:ts[end]
β₀_fit_ts = map(day_to_week_β, ts_nn_fit)
# logit_β₀_fit_ts = map(logit_β_wk_fit, )
plot(ts_nn_fit, β₀_fit_ts)

incidence_nn = FastChain((x, p) -> x,
    FastDense(4, 64, x -> x .+ (1 / a) .* cos.(a .* x) .^ 2),
    FastDense(64, 64, x -> x .+ (1 / a) .* cos.(a .* x) .^ 2),
    FastDense(64, 1, x -> exp.(x .+ (1 / a) .* cos.(a .* x) .^ 2)))
snake_p₀ = initial_params(incidence_nn)
# @time incidence_nn([0,0,0],snake_p₀)[1]

function snake_model(x, p, a)
    incidence_nn = FastChain((x, p) -> x,
        FastDense(4, 64, x -> x .+ (1 / a) .* cos.(a .* x) .^ 2),
        FastDense(64, 64, x -> x .+ (1 / a) .* cos.(a .* x) .^ 2),
        FastDense(64, 1, x -> exp.(x .+ (1 / a) .* cos.(a .* x) .^ 2)))
    return incidence_nn(x, p)
end

@time snake_model(rand(4), snake_p₀, log(1.0))

θ₀ = [log(0.56f0); snake_p₀]

function loss_snake(θ)
    a = exp(θ[1])
    p = θ[2:end]
    # idxs = rand(length(ts_nn_fit)) .> 1 - prob
    pred = [snake_model([t / T, max(0.0f0, (t / T) - 3.3), influ_perc(t), rhino_perc(t)], p, a)[1] for t in ts_nn_fit]
    sum(abs2, β₀_fit_ts .- pred) + sum(abs2, a - 1.0f0), ts_nn_fit, pred
end

@time loss_snake(θ₀)
function plot_snake(x, l, ts, pred)
    plt = scatter(ts_nn_fit, β₀_fit_ts, lab="Data", legend=:topleft)
    plot!(plt, ts, pred, lab="pred", lw=3, alpha=0.5)
    display(plt)
    println("loss = $(l)")
    return false
end

fit_snake = DiffEqFlux.sciml_train(θ -> loss_snake(θ), θ₀,
    callback=plot_snake)
