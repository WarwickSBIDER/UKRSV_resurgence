using OrdinaryDiffEq, StatsPlots, Statistics, CSV, DataFrames, Dates, LinearAlgebra, DiffEqFlux
using DataInterpolations, DiffEqSensitivity, Zygote, JLD2, Plots.PlotMeasures,DiffEqCallbacks
using Distributions,Tullio,GalacticOptim

## Load contact matrices, population distribution and RSV hosp data time series

@load("normalised_contact_matrices_and_demography.jld2")
rsv_hosp_df = CSV.File("rsv_hosp_data.csv") |> DataFrame
ts = rsv_hosp_df.ts
rsv_hosp_rate = rsv_hosp_df.rsv_hosp_rate

##Declare baseline risk per infection
#This baseline is from Kinyanjui et al 2015 using averages over age groups

dis_risk₁ = [mean([31.2, 28.6]); mean([20.0, 13.0]); mean([7.6,2.0]); fill(2.0, 3)] / 100 #Disease risk on first infection
dis_risk₂ = [fill(5.0, 4); 2.0;2.0] / 100 #Disease risk on second infection
hosp_risk = [mean([32.76, 33.07, 21.9, 20.74, 18.86, 12, 27]), mean([9.4, 10.76, 9.1, 12.11, 9, 8.7, 6.7]),
    mean([6.7, 7.11, 7.78, 7.43, 4.13, 4.1, 10.52, 20, 14.84, 13.95, 10.18, 2.36, 8.41]),
    mean([3.76, 1.08, 0.19]), 0.0, 0.0] / 100 #Risk of hospitalisation given disease

hosprisk_per_infection₁ = dis_risk₁ .* hosp_risk
hosprisk_per_infection₂ = dis_risk₂ .* hosp_risk

## Interpolation for percentage positive for other respiratory infections --- used inside loss model

datamart = CSV.File("datamart_resp_data.csv") |> DataFrame
influ_perc = ConstantInterpolation(datamart.influ_perc, datamart.t)
rhino_perc = ConstantInterpolation(datamart.rhino_perc, datamart.t)


## Declare fixed parameters

T = 365.25 #Length of year in days (all rates in per day)
γ₁ = 40.6 / T# mean 9 day infectious period
γ₂ = 93.7 / T# mean 3.9 day infectious period
ω = 2 / T #6 months before return to potential susceptibility
σ = 0.65 #Reduction in susceptibility after first episode
ι = 0.25 #Reduction in transmissibility during subsequent episodes
N = 56.1e6#Pop size England and Wales
L = 80 * T #Mean life expectancy
B = 668_000/T # Mean daily live births in England and Wales since 2000
μ = B / (N * proportion_of_population[end]) # Balancing effective mortality rate to births
ϵ = 5e-6 #~250 RSV infectious contacts from outside UK per day when not in measures
N_vec = N .* proportion_of_population 

## Declare transmission model

#Aging rates 
l_ages = [0.5 * T, 0.5 * T, T, 3 * T, 15 * T, Inf]
r_ages = 1.0 ./ l_ages
aging_mat = Bidiagonal(-r_ages, r_ages[1:(end-1)], :L)
aging_rates(A, u) = @tullio C[a, s, n] := A[a, b] * u[b, s, n] # Einsum contraction of dim = 1, i.e. the age groups

#Piece wise linear effect on transmission rate in schools (when schools are open)
function ct_lkd(t, minpoint_y;
    measures_x=1177.0, minpoint_x=1184.0, returnpoint_x=1916.0)
    ct = 1.0 * (t < measures_x)
    ct += (1.0 + ((t - measures_x) * (minpoint_y - 1.0) / (minpoint_x - measures_x))) * (t ≥ measures_x && t < minpoint_x)
    ct += (minpoint_y + ((t - minpoint_x) * (1.0 - minpoint_y) / (returnpoint_x - minpoint_x))) * (t ≥ minpoint_x && t < returnpoint_x)
    ct += 1.0 * (t ≥ returnpoint_x)
end

##

n_wk = Int((ts[end] - ts[1])/7) #Number of weeks from first week of data and last week (including weeks with no reported data)

function f_RSV_wk(du, u, p, t)
    #Gather parameters and states
    β_school, β_LD, γ₁, γ₂, σ, ι, B, μ, ϵ,schoolsopen_or_closed = p[1:10] #Redeclare some fixed parameters in case of sensitivity analysis
    β_wk = p[(10+1):end] #Weekly continuously varying transmission rates

    S₁ = @view u[:, 1, 1]
    S₂ = @view u[:, 1, 2]
    I₁ = @view u[:, 2, 1]
    I₂ = @view u[:, 2, 2]
    R₁ = @view u[:, 3, 1]
    R₂ = @view u[:, 3, 2]

    #Time varying infection rates
    τ = t < ts[1] ? mod(t, T) : t #Use 2017 as the default year if before data starts
    inlockdown = (t >= T * 3.3 && t <= T * 3.8)
    # schoolopen = (mod(t, T) < 0.553 * T || mod(t, T) > 0.668 * T)
    β₀ = β_wk[min(floor(Int64, (τ - ts[1]) / 7) + 1, n_wk)]# Get weekly transmission rate ex. schools
    β_s = β_school * ct_lkd(t, β_LD) * schoolsopen_or_closed#Within schools transmission rate
    λ =  (β₀ * ct_mat_other .+ β_s * ct_mat_schools) * (I₁ .+ ι .* I₂) ./ N_vec 

    #Transmission Dynamics
    du[:, 1, 1] .= -S₁ .* (λ .+ ϵ * (1 - inlockdown))
    du[:, 1, 2] .= -σ .* S₂ .* (λ .+ ϵ * (1 - inlockdown)) .+ ω .* (R₁ .+ R₂)
    du[:, 2, 1] .= S₁ .* (λ .+ ϵ * (1 - inlockdown)) .- γ₁ .* I₁
    du[:, 2, 2] .= σ .* S₂ .* (λ .+ ϵ * (1 - inlockdown)) .- γ₂ .* I₂
    du[:, 3, 1] .= γ₁ .* I₁ .- ω .* R₁
    du[:, 3, 2] .= γ₂ .* I₂ .- ω .* R₂
    #Add Demography
    du[1, 1, 1] += B # Births
    du[end, :, :] .+= -μ .* u[end, :, :] #Deaths in the final age group
    #Add Aging
    du .+= aging_rates(aging_mat, u)
    return nothing
end

## Set school holidays dates for schools mixing to start and stop
date0 = Date(2017,1,1) #Baseline date

function schoolsclose!(integrator)
    integrator.p[10] = 0.0 #Set the soft indicator function for schoolsopen_or_closed to shut
end
function schoolsopen!(integrator)
    integrator.p[10] = 1.0 #Set the soft indicator function for schoolsopen_or_closed to open
end

#Start and finish dates for holidays based on 2018/2019
xmas_hols_st = [(Date(y,12,24) - date0).value for y = 2008:2023] 
xmas_hols_end = [(Date(y,1,4) - date0).value for y = [2008:2020;2022:2023]] 
febhalf_hols_st = [(Date(y,2,18) - date0).value for y = [2008:2020;2022:2023]]
febhalf_hols_end = [(Date(y,3,1) - date0).value for y = [2008:2020;2022:2023]]
easter_hols_st = [(Date(y,4,8) - date0).value for y = [2008:2019;2021:2023]]
easter_hols_end = [(Date(y,4,26) - date0).value for y = [2008:2019;2021:2023]]
mayhalf_hols_st = [(Date(y,5,27) - date0).value for y = [2008:2019;2021:2023]]
mayhalf_hols_end = [(Date(y,5,31) - date0).value for y = [2008:2019;2021:2023]]
summer_hols_st = [(Date(y,7,15) - date0).value for y = 2008:2023]
summer_hols_end = [(Date(y,9,6) - date0).value for y = 2008:2023]
octhalf_hols_st = [(Date(y,10,21) - date0).value for y = 2008:2023]
octhalf_hols_end = [(Date(y,11,1) - date0).value for y = 2008:2023]

#Start and finish dates for lockdown effects on schools 

lkdown1_st = [(Date(2020,3,18) - date0).value]
lkdown1_end = [(Date(2020,6,1) - date0).value]
lkdown2_st = [(Date(2021,1,4) - date0).value]
lkdown2_end = [(Date(2021,3,8) - date0).value]
#Gather the callbacks
hols_start_times = sort(vcat(xmas_hols_st,febhalf_hols_st,easter_hols_st,mayhalf_hols_st,summer_hols_st,octhalf_hols_st,lkdown1_st,lkdown2_st))
hols_end_times = sort(vcat(xmas_hols_end,febhalf_hols_end,easter_hols_end,mayhalf_hols_end,summer_hols_end,octhalf_hols_end,lkdown1_end,lkdown2_end))

hols_start_cb = PresetTimeCallback(hols_start_times,schoolsclose!, save_positions=(false,false))
hols_end_cb = PresetTimeCallback(hols_end_times,schoolsclose!, save_positions=(false,false))

hols_cb = CallbackSet(hols_start_cb,hols_end_cb)

## Declare basic ODEProblem --- this will get modified during parameter fitting

tspan = (-3136.0, ts[end]) #Simulation runs from 1st June 2008 to 15th May 2022
prob_RSV_wk = ODEProblem(f_RSV_wk, zeros(6,3,2), tspan, rand(n_wk+10)) 

## Set up initial condition arrays

u0 = zeros( 6, 3, 2)
u0[1, 1, 1] = proportion_of_population[1] .* N
u0[2, 1, 1] = 0.75 * proportion_of_population[2] .* N
u0[2, 1, 2] = 0.25 * proportion_of_population[2] .* N
u0[3, 1, 1] = 0.5 * proportion_of_population[3] .* N
u0[3, 1, 2] = 0.5 * proportion_of_population[3] .* N
u0[4, 1, 1] = 0.1 * proportion_of_population[4] .* N
u0[4, 1, 2] = 0.9 * proportion_of_population[4] .* N
u0[5:end, 1, 2] .= proportion_of_population[5:end] .* N
u0 = u0 * 0.99
u0[:, 2, 2] .+= 0.01 * proportion_of_population .* N

change_mat = zeros(Float32, 6, 3, 2)
change_mat[:, 3, 2] .= proportion_of_population .* N

## Set up GP regularisation on the logistic transformed weekly β
ts_gp = collect((ts[1]+7):7.0:ts[end])
# kernel = SE(log(7),log(1)) #<--- typical correlation time 7 days, local std dev 1.0 in logit domain
l = 7.0
σ² = 2.0
C = [σ²*exp(-(ts_gp[i] - ts_gp[j])^2/2*l^2) for i in 1:n_wk, j in 1:n_wk] 
function GP_prior(x,pm,C)
    return -logpdf(MvNormal(C),x .- pm)
end


## Define loss function for inital fitting

function loss_func_wk(θ, prob, hosp_data, sensealg)
    #Get transformed parameters
    β_school_logit, β_LD_logit, p_R_logit, obs_scale_log, risk_rhino_log, risk_flu_log = θ[1:6]
    logit_wk = θ[(6+1):end]
    #Define the precision of the relatively weekly change in transmission and hospitalisation observations
    # prec_logdiff_wk = 1/0.1^2
    prec_hosprate= 1 / 0.25
    #Transform parameters into natural scale
    β_school = 3.0 * sigmoid(β_school_logit)
    β_LD = sigmoid(β_LD_logit)
    p_R = sigmoid(p_R_logit)
    wks = 3.0 * sigmoid.(logit_wk)
    obs_scale = [exp(obs_scale_log + risk_rhino_log * rhino_perc(t) + risk_flu_log * influ_perc(t)) for t in ts]
    # Create new parameters and initial conditions
    _p = [[β_school, β_LD, γ₁, γ₂, σ, ι, B, μ, ϵ,1.0]; wks]
    _u0 = u0 .* (1 - p_R) .+ p_R .* change_mat
    #Modify the ODEProblem and solve
    _prob = remake(prob; p=_p, u0=_u0)
    sol = solve(_prob, Tsit5(); saveat=ts, sensealg=sensealg,callback = hols_cb)
    failure = length(sol.u) < length(ts)
    #Return loss
    if failure
            return Inf, zeros(100)
    else
        #Make predictions of hospitalisation rates and 1 year old and 
        pred_hosps = 7.0 * obs_scale .* [sum(γ₁ .* u[:, 2, 1] .* hosprisk_per_infection₁ .+ γ₂ * u[:, 2, 2] .* hosprisk_per_infection₂) for u in sol.u] .* (100_000 / N)
        pred_6_12mnth_seropos = [(sum(u[2, :, 2]) + sum(u[2, 2:3, 1])) / (sum(u[2, :, :])) for u in sol.u[[24, 46, 67]]]
        pred_1_seropos = [(sum(u[3, :, 2]) + sum(u[3, 2:3, 1])) / (sum(u[3, :, :])) for u in sol.u[[24, 46, 67]]]
        #loss due to prediction error
        loss = sum(abs, pred_hosps .- hosp_data) * prec_hosprate
        loss += sum([-logpdf(Beta(15,85),pred) for pred in pred_6_12mnth_seropos])
        loss += sum([-logpdf(Beta(50,50),pred) for pred in pred_1_seropos])
        #loss due to deviation from priors
        reg = GP_prior(logit_wk,-2.0,C) #Log-Normal prior distributed change in transmission rates across weeks
        reg += 0.5 * sum(abs2, [obs_scale_log, risk_rhino_log, risk_flu_log]) #log-normal prior distributed risk factors
        # reg += -logpdf(Gamma(20,10/20),0.5* prec_hosprate^2) #
        # reg += -logpdf(Gamma(20,100/20), prec_logdiff_wk)
        return loss + reg, pred_hosps
    end
end

##
logit_wk₀ =  rand(n_wk)
θ₀ = [rand(6); logit_wk₀] 
@time l, pred = loss_func_wk(θ₀, prob_RSV_wk, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
# @time loss_func_wk(θ₀, prob_RSV_wk, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
plot(pred)
# 
##
Zygote.gradient(θ -> loss_func_wk(θ, prob_RSV_wk, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false)))[1],
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
fit = DiffEqFlux.sciml_train(θ -> loss_func_wk(θ, prob_RSV_wk, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))),
    θ₀,
    callback=plot_hosps)

## Save fit
# @save("fit_wk_mat_cb.jld2", fit)
##
###
### MEAN-Field VI
### 
# @load("fit_wk_mat_cb.jld2")
# θ₀ = copy(fit.u)
n_params = length(θ₀)
n_vi_params = 2*length(θ₀)
ll = log(7)
lσ = 0.0
pm = -2.0
vi_θ₀ = [θ₀;fill(-3.0,n_params)]#;ll;lσ;pm]

function loss_func_wk_vi(θ_vi, prob, hosp_data, sensealg)
    #Generate parameters under mean-field Gaussian assumption
    μs = θ_vi[1:n_params]
    ωs = θ_vi[(n_params+1):(2*n_params)]
    # ll = θ_vi[end-2]
    # lσ = θ_vi[end-1]
    # pm = θ_vi[end]
    # C = [1e-3*(i==j) + exp(2*lσ)*exp(-(ts_gp[i] - ts_gp[j])^2/2*exp(2*l)) for i in 1:n_wk, j in 1:n_wk] 
    
    θ = exp.(ωs).*randn(n_params) .+ μs
    # θ = μs
    #Get transformed parameters
    β_school_logit, β_LD_logit, p_R_logit, obs_scale_log, risk_rhino_log, risk_flu_log = θ[1:6]
    logit_wk = θ[(6+1):end]
    #Define the precision of the relatively weekly change in transmission and hospitalisation observations
    # prec_logdiff_wk = 1/0.1^2
    prec_hosprate= 1 / 0.25
    #Transform parameters into natural scale
    β_school = 3.0 * sigmoid(β_school_logit)
    β_LD = sigmoid(β_LD_logit)
    p_R = sigmoid(p_R_logit)
    wks = 3.0 * sigmoid.(logit_wk)
    obs_scale = [exp(obs_scale_log + risk_rhino_log * rhino_perc(t) + risk_flu_log * influ_perc(t)) for t in ts]
    # Create new parameters and initial conditions
    _p = [[β_school, β_LD, γ₁, γ₂, σ, ι, B, μ, ϵ,1.0]; wks]
    _u0 = u0 .* (1 - p_R) .+ p_R .* change_mat
    #Modify the ODEProblem and solve
    _prob = remake(prob; p=_p, u0=_u0)
    sol = solve(_prob, Tsit5(); saveat=ts, sensealg=sensealg,callback = hols_cb)
    failure = length(sol.u) < length(ts)
    #Return loss
    if failure
            return Inf, zeros(100)
    else
        #Make predictions of hospitalisation rates and 1 year old and 
        pred_hosps = 7.0 * obs_scale .* [sum(γ₁ .* u[:, 2, 1] .* hosprisk_per_infection₁ .+ γ₂ * u[:, 2, 2] .* hosprisk_per_infection₂) for u in sol.u] .* (100_000 / N)
        pred_6_12mnth_seropos = [(sum(u[2, :, 2]) + sum(u[2, 2:3, 1])) / (sum(u[2, :, :])) for u in sol.u[[24, 46, 67]]]
        pred_1_seropos = [(sum(u[3, :, 2]) + sum(u[3, 2:3, 1])) / (sum(u[3, :, :])) for u in sol.u[[24, 46, 67]]]
        #loss due to prediction error
        loss = sum(abs, pred_hosps .- hosp_data) * prec_hosprate
        loss += sum([-logpdf(Beta(15,85),pred) for pred in pred_6_12mnth_seropos])
        loss += sum([-logpdf(Beta(50,50),pred) for pred in pred_1_seropos])
        #loss due to deviation from priors
        reg = GP_prior(logit_wk,-3,C) #Log-Normal prior distributed change in transmission rates across weeks
        
        reg += 0.5 * sum(abs2, [obs_scale_log, risk_rhino_log, risk_flu_log]) #log-normal prior distributed risk factors
        # reg += 0.5 * (1/1.5^2) * sum(abs2,[β_LD_logit, p_R_logit]) 
        reg += -logpdf(Gamma(3,1.5/3),β_school)  

        entropy_mf_gaussian = sum(ω)
        # reg += -logpdf(Gamma(20,10/20),0.5* prec_hosprate^2) #
        # reg += -logpdf(Gamma(20,100/20), prec_logdiff_wk)
        return loss + reg - entropy_mf_gaussian, pred_hosps
    end
end

loss,pred = loss_func_wk_vi(vi_θ₀, prob_RSV_wk, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false)))
plot(pred)

fit_vi = DiffEqFlux.sciml_train(θ -> loss_func_wk_vi(θ, prob_RSV_wk, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))),
    vi_θ₀,
    callback=plot_hosps,
    maxiters = 500)

@save("fit_wk_mat_cb_vi_gp.jld2", fit_vi)

bar(exp.(fit_vi[(n_params+1):end]),lab = "")
plot(3*sigmoid.(fit_vi.u[7:n_params]))

## Play with GPs
ts = collect(1:1.0:445)'
kernel = SE(log(7),log(1))
C = cov(kernel,ts,ts) .+ 1e-10*Diagonal(ones(445))
ch = cholesky(C)
C, ch = GaussianProcesses.make_posdef!(C)
isposdef(C)
function GP_prior(x,pm,C)
    return logpdf(MvNormal(C),x .- pm)
end

x = zeros(445)

Zygote.withgradient(x -> GP_prior(x,0.0,C),x)

heatmap(C)