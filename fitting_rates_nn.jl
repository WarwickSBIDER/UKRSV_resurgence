cd("./RSV_resurgence/")
using Pkg
Pkg.activate(".")
##


using StatsPlots, Statistics, Dates, LinearAlgebra,CSV, DataFrames
using Flux, ForwardDiff, Zygote, JLD2, Plots.PlotMeasures,DiffEqFlux
using Distributions, RecursiveArrayTools,DataInterpolations,Turing,MCMCChains

##
@load("fit_wk_mat.jld2", fit)
n_wk = 278 + 165

β_wk_fit =  3.0* sigmoid.(fit.u[(6+1):(6+n_wk)])

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

##
influ_perc = ConstantInterpolation((datamart.Influenza_A_perc .+ datamart.Influenza_B_perc) ./ 100 .|> Float32, datamart.t .|> Float32)
rhino_perc = ConstantInterpolation(datamart.Rhinovirus_perc ./ 100 .|> Float32, datamart.t .|> Float32)


##
ts_nn_fit = collect(ts[1]:7:ts[end])[1:(end-1)]
scatter(ts_nn_fit,β_wk_fit)

## Construct input data matrix
T = 365.25f0
y = log.([Float32(β) for i = 1:1,β in β_wk_fit])

X = [[ts_nn_fit[j]/T,max(0f0,(ts_nn_fit[j]/T - 3.3)),influ_perc(ts_nn_fit[j]),rhino_perc(ts_nn_fit[j])][i] 
        for i = 1:4,j = 1:length(ts_nn_fit)]


##
snake_layer(x) = Float32.(x .+ (1/(2*pi))*cospi.(2.0*x) )

# model = Chain(LSTM((n_features + n_rand_elements) => 4), Dense(4, 1), x -> exp.(x)) |> f32
p_drop = 0.05
model = Chain(
        Dense(4,64),snake_layer,
        # Flux.Dropout(p_drop,dims = 1),
        Dense(64,64),snake_layer,
        # Flux.Dropout(p_drop,dims = 1),
        Dense(64,1)
) |> f32

@time model(X)
sqnorm(x) = sum(x -> abs2(x)/(2*0.04), x);

function loss(X,y)
    sum(abs2,model(X) .- y)/(2*0.01) + sum(sqnorm, Flux.params(model))
end
opt = Flux.ADAM(0.001)
ps = Flux.params(model)
data = [(X,y)]


## Learning epochs

# preds = model(X)[:]
# plt = scatter(ts_nn_fit,y[:],lab = "data")
# plot!(plt, ts_nn_fit,preds,
#         lab = "pred", lw = 3, color = :red, alpha = 0.3)

for epoch in 1:30_000
    Flux.train!(loss, ps, data, opt)
    if epoch % 100 == 0
        preds = model(X)[:]
        plt = scatter(ts_nn_fit,y[:],lab = "data")
        plot!(plt, ts_nn_fit,preds,
                lab = "pred", lw = 3, color = :red, alpha = 0.5)
        display(plt)
        println("Epoch: $(epoch), loss per sample = $(loss(X, y)/length(y))")
    end
end

##
testmode!(model,true)
preds = model(X)[:]
plt = scatter(ts_nn_fit,y[:],lab = "data")
plot!(plt, ts_nn_fit,preds,
        lab = "pred", lw = 3, color = :red, alpha = 0.5)
display(plt)

##
p_fit, re = Flux.destructure(model)
n_p = length(p_fit)

rand_ps = [0.2*randn(n_p) for j = 1:100]

prior_preds = vecvec_to_mat([re(rand_ps[j])(X)[:] for j = 1:100])'
plot(prior_preds,color = :grey,lab = "")


## Turing Bayesian NN

# Create a regularization term and a Gaussian prior variance term.
# alpha = 0.09
# sig = sqrt(1.0 / alpha)
sig = 0.5f0
err_sig = 0.05f0

Turing.setadbackend(:zygote)
# Specify the probabilistic model.
@model function bayes_nn(ys, X, n_p,n_data, re)
    # Create the NN parameters.
    ps ~ MvNormal(zeros(n_p), sig .* ones(n_p))
    # Construct NN from parameters
    nn = re(ps)
    # Forward NN to make predictions
    preds = nn(X)[:]
    #Calculate error
    error ~ MvNormal(preds .- ys,err_sig .* ones(Float32,n_data))
end

inference_model = bayes_nn(y[:],X,length(p_fit),length(y),re)
##
chain = sample(inference_model, NUTS(), 100)

theta = MCMCChains.group(chain, :ps).value

preds = vecvec_to_mat([re(theta.data[i,:,1])(X)[:] for i = 1:100])'

scatter(ts_nn_fit,y[:],lab = "data")
plot(ts_nn_fit[1:10],preds[1:10,:],lw = 2 , color = :grey, alpha = 0.3,lab = "")