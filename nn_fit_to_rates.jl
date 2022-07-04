using JLD2,Flux,BSON,Statistics,DiffEqFlux
using CSV,DataFrames,DataInterpolations, Plots
## Interpolation for percentage positive for other respiratory infections --- used inside loss model

datamart = CSV.File("datamart_resp_data.csv") |> DataFrame
influ_perc = ConstantInterpolation(datamart.influ_perc, datamart.t)
rhino_perc = ConstantInterpolation(datamart.rhino_perc, datamart.t)

## hospitalisation data

rsv_hosp_df = CSV.File("rsv_hosp_data.csv") |> DataFrame
ts = rsv_hosp_df.ts
rsv_hosp_rate = rsv_hosp_df.rsv_hosp_rate


## Fitted rates data
@load("fit_wk_mat_cb_vi.jld2")
n_params = Integer(length(fit_vi.u)/2)
μs = fit_vi.u[7:n_params]
ωs = fit_vi.u[(n_params+7):end]

y = 3 * sigmoid.(μs)
σs = 3 * sigmoid.(μs .+ exp.(ωs)) .- y

scatter(y,yerr = 3*σs)
rel_noise = σs./minimum(σs)

plot(μs,ribbon = 10*exp.(ωs))

## Option 1: fit as time varying with snake layers
ts_nn_fit = collect(ts[1]:7.0:(ts[1]+7*(length(μs)-1)))
scatter(ts_nn_fit,μs,yerr = 3* exp.(ωs))

## Construct input data matrix
# Each col = [time (years rel to 1st Jan 2017), time since first lockdown, prop flu (standardised on prepandemic data), prop rhino (standardised on prepandemic data)]
T = 365.25f0
mean_flu = mean(datamart.influ_perc[datamart.t .< 1000])
std_flu = std(datamart.influ_perc[datamart.t .< 1000])
mean_rhino = mean(datamart.rhino_perc[datamart.t .< 1000])
std_rhino = std(datamart.rhino_perc[datamart.t .< 1000])


# X = [[ts_nn_fit[j]/T,max(0f0,(ts_nn_fit[j]/T - 3.3)),(influ_perc(ts_nn_fit[j])-mean_flu)/std_flu,(rhino_perc(ts_nn_fit[j]) - mean_rhino)/std_rhino][i] 
#         for i = 1:4,j = 1:length(ts_nn_fit)]

X = [[ts_nn_fit[j]/T,max(0f0,(ts_nn_fit[j]/T - 3.3))][i] 
        for i = 1:2,j = 1:length(ts_nn_fit)]        




# model = Chain(LSTM((n_features + n_rand_elements) => 4), Dense(4, 1), x -> exp.(x)) |> f32
p_drop = 0.05
snake_layer(x) = Float32.(x .+ (1/(2*pi))*(sinpi.(2.0*x)).^2 )
# model = Chain(
#         Dense(4,64),snake_layer,
#         Flux.Dropout(p_drop,dims = 1),
#         Dense(64,64),snake_layer,
#         Flux.Dropout(p_drop,dims = 1),
#         Dense(64,1)
# ) |> f32
model = Chain(
        Dense(2,64),snake_layer,
        # Flux.Dropout(p_drop,dims = 1),
        Dense(64,64),snake_layer,
        Flux.Dropout(p_drop,dims = 1),
        Dense(64,1)
) |> f32

@time model(X)
sqnorm(x,σ²_w) = sum(x -> abs2(x)/(2*σ²_w), x);
sqnorm(x) = sum(x -> abs2(x)/(2*100), x);


function loss(X,y)
    0.5 * sum(abs2,(model(X) .- y)./rel_noise) + sum(sqnorm, Flux.params(model))
end
opt = Flux.ADAM(0.001)
ps = Flux.params(model)
data = [(X,y')]


## Learning epochs

# preds = model(X)[:]
# plt = scatter(ts_nn_fit,y[:],lab = "data")
# plot!(plt, ts_nn_fit,preds,
#         lab = "pred", lw = 3, color = :red, alpha = 0.3)

for epoch in 1:30_000
    Flux.train!(loss, ps, data, opt)
    if epoch % 100 == 0
        preds = model(X)[:]
        plt = scatter(ts_nn_fit,y[:],yerr = 3 * σs, lab = "data")
        plot!(plt, ts_nn_fit,preds,
                lab = "pred", lw = 3, color = :red, alpha = 0.5)
        display(plt)
        println("Epoch: $(epoch), loss per sample = $(loss(X, y')/length(y))")
    end
end

##Make a prediction

testmode!(model,true)
ts_nn_pred = collect((ts_nn_fit[end] + 7):7.0:(ts_nn_fit[end] + 52*7))

X_future = [[ts_nn_pred[j]/T,max(0f0,(ts_nn_pred[j]/T - 3.3)),(0-mean_flu)/std_flu,(0- mean_rhino)/std_rhino][i] 
for i = 1:4,j = 1:length(ts_nn_pred)]

preds = model([X X_future])[:]
plt = scatter(ts_nn_fit,y[:],lab = "data")
plot!(plt, [ts_nn_fit;ts_nn_pred],preds,
        lab = "pred", lw = 3, color = :red, alpha = 0.5)
display(plt)

##
p_fit, re = Flux.destructure(model)
n_p = length(p_fit)

rand_ps = [0.2*randn(n_p) for j = 1:100]

prior_preds = vecvec_to_mat([re(rand_ps[j])(X)[:] for j = 1:100])'
plot(prior_preds,color = :grey,lab = "")


##
