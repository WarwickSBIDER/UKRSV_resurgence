###
### PRE-PROCESS RSV DATA INTO FORM FOR MODELLING
###

using CSV, FileIO, DataFrames, Dates,JLD2, RCall,LinearAlgebra

## Load contact matrices and popultation size data

R"""
library('socialmixr')
data("polymod")
ct_plymd_info = socialmixr::contact_matrix(polymod, countries = "United Kingdom", age.limits = c(0, 1,2, 4, 19), symmetric = TRUE)
ct_plymd_info2 = socialmixr::contact_matrix(polymod, countries = "United Kingdom", age.limits = seq(0,75,5), symmetric = TRUE)

library('contactdata')
pop_data = contactdata::age_df_countries(c("UK"))
ct_all = contactdata::contact_matrix('UK', location = c("all"))
ct_school = contactdata::contact_matrix('UK', location = c("school"))
"""

ct_all_fromto = @rget ct_all # All setting contact matrix in 5 year age bins
ct_school_fromto = @rget ct_school # school contact matrix in 5 year age bins
ct_plymd_dict = @rget ct_plymd_info # Polymod contact matrix and demographics in (almost) the age structure for model
ct_plymd2_dict = @rget ct_plymd_info2 # Polymod contact matrix and demographics in 5 year age bins, used in conversion

#Spilt the 0-1 years old into 0-6 and 7-12 months old
proportion_of_population = [fill(ct_plymd_dict[:demography].proportion[1] / 2, 2); ct_plymd_dict[:demography].proportion[2:5]]
proportion_of_population_5yrs = ct_plymd2_dict[:demography].proportion

## Conversion matrices for transitioning from 16 age groups to 6
#These encode the idea of Prob( age ∈ a | age ∈ A) where A is 
#the age group among the 6 for the model and a is the age group among the 16 for Prem/Jit contact matrix
P_Aa = zeros(6, 16) 
P_Aa[1, 1] = 1 #e.g. Probability that someone in age group 0-6 months is in age group 0-5 years
P_Aa[2, 1] = 1
P_Aa[3, 1] = 1
P_Aa[4, 1] = 1
P_Aa[5, 2] = 1 / 3
P_Aa[5, 3] = 1 / 3
P_Aa[5, 4] = 1 / 3
P_Aa[6, 5:end] = proportion_of_population_5yrs[5:end] ./ sum(proportion_of_population_5yrs[5:end])

P_bB = zeros(16, 6) #e.g the probability that someone in age group 0-5 years is in age group 0-6 months
P_bB[1, 1] = 0.5 / 5
P_bB[1, 2] = 0.5 / 5
P_bB[1, 3] = 1 / 5
P_bB[1, 4] = 3 / 5
P_bB[2:4, 5] .= 1.0
P_bB[5:end, 6] .= 1.0

## Create normalized mixing matrices with leading eigen value 1 --- so that tranmission rates can be expressed as O(1) size
#also converting so that c_ab = normalised rate of someone in age group b meeting anyone in age group a
ct_all_un = Matrix((P_Aa * ct_all_fromto * P_bB)')
ct_school_un = Matrix((P_Aa * ct_school_fromto * P_bB)')
ct_other_un = ct_all_un .- ct_school_un

max_eigval_ct = maximum(abs.(eigvals(ct_all_un)))
ct_mat_other = ct_other_un ./ max_eigval_ct 
ct_mat_schools = ct_school_un ./ max_eigval_ct

@save("normalised_contact_matrices_and_demography.jld2",ct_mat_other,ct_mat_schools,proportion_of_population)

## Load raw data from DATAMART and SARIWATCH for RSV and other resp. viruses
ref_date = Date(2017,1,1)
datamart = CSV.File("datamart_other_resp_from_wk402013.csv") |> DataFrame
datamart.date = Date.(datamart.date, DateFormat("dd/mm/yyyy"));
sariwatch = CSV.File("SARI_watch_RSV.csv") |> DataFrame;
sariwatch.date = Date.(sariwatch.date, DateFormat("dd/mm/yyyy"));

ref_date = Date(2017, 1, 1)
sariwatch.t = [Float64((d - ref_date).value) for d in sariwatch.date]
datamart.t = [Float64((d - ref_date).value) for d in datamart.date]

## Linear fit of datamart RSV 0-4 year old perc positive against 
#Sari-watch RSV hospitalisation rates to project Sariwatch data into past
#Also projection each week with <= 0.5% datamart RSV as low RSV hosp incidence
#Then gather data with sorted time indices

comparison_times = intersect(sariwatch.t, datamart.t[.~ismissing.(datamart.RSV_0_4_perc)])
comparison_datamart = datamart.RSV_0_4_perc[[t ∈ comparison_times for t in datamart.t]]
comparison_sariwatch = sariwatch.RSV_hosp_ICU_per_100_000[[t ∈ comparison_times for t in sariwatch.t]]
β_fit = comparison_datamart \ comparison_sariwatch

projection_times = setdiff(datamart.t[.~ismissing.(datamart.RSV_0_4_perc)], comparison_times)
projection_idxs = [t ∈ projection_times for t in datamart.t]
projection_times2 = setdiff(setdiff(datamart.t[[p <= 0.5 for p in datamart.RSV_perc]], sariwatch.t), collect(1000.0:1.0:2000.0))
projection_idxs2 = [t ∈ projection_times2 for t in datamart.t]

ts_data_unsrted = [datamart.t[projection_idxs]; datamart.t[projection_idxs2]; sariwatch.t]
projection_unsrted = [datamart.RSV_0_4_perc[projection_idxs] .* β_fit; fill(0.05, length(datamart.t[projection_idxs2])); sariwatch.RSV_hosp_ICU_per_100_000]
idxs = sortperm(ts_data_unsrted)
ts = ts_data_unsrted[idxs]
rsv_hosp_rate = projection_unsrted[idxs]

CSV.write("rsv_hosp_data.csv",DataFrame(ts = ts, rsv_hosp_rate = rsv_hosp_rate))
CSV.write("datamart_resp_data.csv",DataFrame(t = datamart.t,
                                                influ_perc = (datamart.Influenza_A_perc .+ datamart.Influenza_B_perc) ./ 100,
                                                rhino_perc = (datamart.Rhinovirus_perc) ./ 100))
