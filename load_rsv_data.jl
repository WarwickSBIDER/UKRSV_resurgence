###
### LOAD RSV DATA
###

using CSV, FileIO, DataFrames, Dates,JLD2, RCall

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
"""

ct_all_fromto = @rget ct_all
ct_school_fromto = @rget ct_school
ct_plymd_dict = @rget ct_plymd_info
ct_plymd2_dict = @rget ct_plymd_info2

pop_data = @rget pop_data

proportion_of_population = [fill(ct_plymd_dict[:demography].proportion[1] / 2, 2); ct_plymd_dict[:demography].proportion[2:5]] .|> Float32
proportion_of_population_5yrs = ct_plymd2_dict[:demography].proportion
