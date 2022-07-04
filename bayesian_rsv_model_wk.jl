using OrdinaryDiffEq, StatsPlots, Statistics, CSV, DataFrames, Dates, LinearAlgebra, Tullio
using DataInterpolations, DiffEqSensitivity, Zygote, JLD2, Plots.PlotMeasures, DiffEqCallbacks
using Distributions,MCMCChains
using Flux: sigmoid
using Turing
using Turing: Variational

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

##
###
### BAYESIAN FITTING 
### 
@load("fit_wk_mat_cb.jld2")

function log_lklhood(θ, prob, hosp_data, sensealg)
    #Get transformed parameters
    β_school_logit, β_LD_logit, p_R_logit, obs_scale_log, risk_rhino_log, risk_flu_log = θ[1:6]
    logit_wk = θ[(6+1):end]
    #Define the precision of the relatively weekly change in transmission and hospitalisation observations
    prec_logdiff_wk = 1/0.1^2
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
            return -Inf
    else
        #Make predictions of hospitalisation rates and 1 year old and 
        pred_hosps = 7.0 * obs_scale .* [sum(γ₁ .* u[:, 2, 1] .* hosprisk_per_infection₁ .+ γ₂ * u[:, 2, 2] .* hosprisk_per_infection₂) for u in sol.u] .* (100_000 / N)
        pred_6_12mnth_seropos = [(sum(u[2, :, 2]) + sum(u[2, 2:3, 1])) / (sum(u[2, :, :])) for u in sol.u[[24, 46, 67]]]
        pred_1_seropos = [(sum(u[3, :, 2]) + sum(u[3, 2:3, 1])) / (sum(u[3, :, :])) for u in sol.u[[24, 46, 67]]]
        #loss due to prediction error
        loss = -sum(abs, pred_hosps .- hosp_data) * prec_hosprate
        loss += sum([logpdf(Beta(15,85),pred) for pred in pred_6_12mnth_seropos])
        loss += sum([logpdf(Beta(50,50),pred) for pred in pred_1_seropos])
        
        return loss
    end
end

function log_prior(θ)
    β_school_logit, β_LD_logit, p_R_logit, obs_scale_log, risk_rhino_log, risk_flu_log = θ[1:6]
    logit_wk = θ[(6+1):end]
    wks = 3.0 * sigmoid.(logit_wk)
    #Define the precision of the relatively weekly change in transmission and hospitalisation observations
    prec_logdiff_wk = 1/0.1^2
    prec_hosprate= 1 / 0.25
    #loss due to deviation from priors
    reg = -sum(abs2, diff(log.(wks))) * prec_logdiff_wk * 0.5 #Log-Normal prior distributed change in transmission rates across weeks
    reg -= 0.5 * sum(abs2, [obs_scale_log, risk_rhino_log, risk_flu_log]) #log-normal prior distributed risk factors
    return reg
end

θ_fit = copy(fit.u)

log_posterior_density(θ) = log_lklhood(θ, prob_RSV_wk, rsv_hosp_rate, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))) + log_prior(θ)

log_posterior_density(θ_fit)
val,g = Zygote.withgradient(log_posterior_density,θ_fit)

n_params = length(θ_fit)
# function psgld!(posterior_density,θ,V,G,λ,α,dt)
#     g = Zygote.gradient(posterior_density,θ)[1]
#     V .= α.*V .+ (1 - α) .* g .*g
#     G .= 1 ./ ( λ .+ sqrt.(V))
#     θ .+= (0.5 * dt * G .* g) .+ (sqrt(dt) * sqrt.(G) .* randn(n_params))
# end

function mala(lpd::Function,θ,V,G,λ,α,dt)
    acceptance = false
    log_pd,∇ = Zygote.withgradient(lpd,θ)#log post. density and grad
    g = ∇[1]
    V .= α.*V .+ (1 - α) .* g .*g #Update local estimate of the element squared gradient
    G .= 1 ./ ( λ .+ sqrt.(V)) #Update local estimate for prec-conditioning matrix
    θ_prop = θ .+ (0.5 * dt * G .* g) .+ (sqrt(dt) * sqrt.(G) .* randn(n_params)) # Proposed next parameters
    log_pd_prop,∇_prop = Zygote.withgradient(lpd,θ_prop) # log post density and grad at proposal 
    g_prop = ∇_prop[1]
    #Calculate the acceptance probability
    log_prob_step = log_pd_prop - log_pd  
    log_prob_step += 0.5 * (1/dt) * (-sum((λ .+ sqrt.(V)) .* (θ .- (θ_prop .+ 0.5 * dt * G .* g_prop)).^2 ) + sum((λ .+ sqrt.(V)) .* (θ_prop .- (θ .+ 0.5 * dt * G .* g)).^2 ) )
    if log_prob_step >= 0.0 || log(rand()) < log_prob_step
        θ .= θ_prop
        acceptance = true
        log_pd = log_pd_prop
    end

    return acceptance,min(exp(log_prob_step),1.0),log_pd

end
# @time acc,acc_prob = mala(log_posterior_density,θ_fit .+ 0.00001*randn(n_params) ,zeros(n_params),zeros(n_params),1e-5,0.99,0.01)
# @time psgld!(posterior_density,θ₀,zeros(n_params),zeros(n_params),1e-3,0.99,0.01)


# function run_psgld(posterior_density,θ₀,n_samples;λ = 1e-3,α = 0.99,dt = 0.01,sample_times = 0.1)
#     G = zeros(n_params)
#     draws = zeros(n_params,n_samples)
#     θ = copy(θ₀)
#     println("Initial post. density $(posterior_density(θ))")
#     # g₀ = Zygote.gradient(posterior_density,θ₀)[1]
#     V = zeros(n_params)

#     for j = 1:n_samples
#         T = 0.0
#         while T <= sample_times
#             psgld!(posterior_density,θ,V,G,λ,α,dt)
#             T += dt
#         end
#         draws[:,j] .= θ
#         println("Sample $(j), post. density $(posterior_density(θ))")
#     end
#     return draws
# end


function run_mala(lpd::Function,θ₀,n_samples;λ = 1e-3,α = 0.99,dt = 0.01,sample_times = 0.1)
    G = zeros(n_params)
    draws = zeros(n_params,n_samples)
    θ = copy(θ₀)
    log_pd_init = lpd(θ)
    println("Initial log. post. density $(log_pd_init)")
    # g₀ = Zygote.gradient(posterior_density,θ₀)[1]
    V = zeros(n_params)
    recent_acp_prob = 0.574
    for j = 1:n_samples
        T = 0.0
        while T <= sample_times
            acc,acc_prob,log_pd =  mala(lpd,θ,V,G,λ,α,dt)
            recent_acp_prob = 0.95*recent_acp_prob + 0.05*acc_prob
            log_pd_init = log_pd
            # dt *= 1 + 0.1*(recent_acp_prob - 0.574)
            # println("time = $(T), acc prob = $(recent_acp_prob), log. post. den = $(log_pd)")
            if acc
                T += dt
            end
        end
        draws[:,j] .= θ
        println("Sample $(j), log. post. density $(log_pd_init), acc prob = $(recent_acp_prob)")
    end
    return draws
end


θ₀ = copy(θ_fit)

# draws = run_psgld(posterior_density,θ₀,1000;dt = 1e-4,λ=1e-3,sample_times = 1e-3)
draws = run_mala(log_posterior_density,θ₀.*exp.(0.00.*randn(n_params)),1000;dt = 1e-5,λ=1e-3,sample_times = 1e-4)

# setadbackend(:zygote)
# advi = ADVI(10, 1000)

# @model function lpd(prob, hosp_data, sensealg)
#     β_school_logit ~ Normal(0,4)

#     , β_LD_logit, p_R_logit, obs_scale_log, risk_rhino_log, risk_flu_log
# end

# q = vi(log_posterior_density, advi)



# #After inital estimate run a Bayesian inference using EM algorithm to regress onto marginal mode
# #estimate for the weekly infection rates
# Turing.setadbackend(:forwarddiff)
# @model function rsv_fixed_wk(prob,hosp_data,target_6_12mnth_seropos,target_1yr_seropos,β_wk,diff_log_β_wk)
#     #Declare priors
#     β_school ~ truncated(Normal(2.0, 1.0),0,3.0)
#     β_LD ~ truncated(Normal(0.5, 0.25),0,1.5)
#     p_R ~ Beta(2.5,7.5)
#     obs_scale_log ~ Normal(0,1)
#     risk_rhino_log ~ Normal(0,1)
#     risk_flu_log ~ Normal(0,1)
#     β_scale ~ LogNormal(0.0,0.05)
#     prec_logdiff_wk ~ Gamma(5,100/5)
#     prec_hosprate ~ Gamma(5,4/5)
    
#     #Redefine ODEProblem
#     _p = [[β_school, β_LD, γ₁, γ₂, σ, ι, B, μ, ϵ,1.0]; β_scale * β_wk]
#     _u0 = u0 .* (1 - p_R) .+ p_R .* change_mat
#     _prob = remake(prob; p=_p, u0=_u0)
#     sol = solve(_prob, Tsit5(); saveat=ts, 
#                                 callback = hols_cb)    
#     if sol.retcode != :Success
#         Turing.@addlogprob! -Inf
#         # Exit the model evaluation early
#         return
#     end

#     obs_scale = [exp(obs_scale_log + risk_rhino_log * rhino_perc(t) + risk_flu_log * influ_perc(t)) for t in ts]
#     #Make predictions of hospitalisation rates and 1 year old and 
#     pred_hosps = 7.0 * obs_scale .* [sum(γ₁ .* u[:, 2, 1] .* hosprisk_per_infection₁ .+ γ₂ * u[:, 2, 2] .* hosprisk_per_infection₂) for u in sol.u] .* (100_000 / N)
#     pred_6_12mnth_seropos = [(sum(u[2, :, 2]) + sum(u[2, 2:3, 1])) / (sum(u[2, :, :])) for u in sol.u[[24, 46, 67]]]
#     pred_1_seropos = [(sum(u[3, :, 2]) + sum(u[3, 2:3, 1])) / (sum(u[3, :, :])) for u in sol.u[[24, 46, 67]]]

#     #Assess log-likelihood
#     # diff_log_β_wk = diff(log.(β_wk))
#     diff_log_β_wk ~ MvNormal(fill(0.0,length(diff_log_β_wk)),fill(1/prec_logdiff_wk,length(diff_log_β_wk)))
    
#     for i in 1:length(pred_hosps)
#         hosp_data[i] ~ Laplace(pred_hosps[i],sqrt(1/(2*prec_hosprate)))
#     end
#     for k = 1:3
#         target_6_12mnth_seropos[k] ~ Beta(pred_6_12mnth_seropos[k]*100,(1-pred_6_12mnth_seropos[k])*100)
#         target_1yr_seropos[k] ~ Beta(pred_1_seropos[k]*100,(1-pred_1_seropos[k])*100)
#     end        
# end

# β_fit = 3 * sigmoid.(fit.u[7:end])
# rsv_inference = rsv_fixed_wk(prob_RSV_wk, rsv_hosp_rate,fill(0.15,3),fill(0.5,3),β_fit,diff(log.(β_fit)))

# ## Sample
# # map_estimate = optimize(rsv_inference, MAP())
# chain = sample(rsv_inference, NUTS(), 1_000)
# # priorchain = sample(rsv_inference, Prior(), 1_000)


# # chn = sample(rsv_inference,NUTS(),1000)
