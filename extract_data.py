### import modules ###
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import Counter
import os
import dhs_f_education
import fnmatch
import pyreadstat
import copy
#for meaningfull error massages turn of parallel_apply/ replace with apply
from pandarallel import pandarallel
import time
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
import plotly.express as px
import scipy
# Initialization
pandarallel.initialize()

### set display options ###
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

### options ###
min_dhs_version = 6

###Paths
dhs_path = r"/mnt/datadisk/data/surveys/DHS_raw_data/"
projects_p = r"/mnt/datadisk/data/Projects/education/"
locations_f = projects_p + '/' + 'locations.csv'

#walk folder for finding .tifs
walk_f = r"/mnt/datadisk/"

#might be split into urban/rural as well cf. below
out_f = projects_p + '/labels_new.csv'
#export on HH level for further analysis (Imputation project Mrs. Pranti)
out_f_hh = projects_p + '/hh_all.csv'

questionaires = ['IR','MR','PR']

codes_to_keep = [
            ##### Source: https://dhsprogram.com/data/Guide-to-DHS-Statistics/Organization_of_DHS_Data.htm
            ### Household Member Recode (PR) ###
            ## Basic characteristics of the household interview ##
            'HV016', #day of interview
            'HV009', #number of household members
            'HV008', #date of interview (cmc)
            'HV007', #year of interview
            'HV006', #month of interview
            'HV002', #household number
            'HV001', #cluster number
            'HV000', #country code and phase
            'HV003', #respondent's line number (answering household questionnaire)
            'HV024', #transmission zones, region, province
            ## Characteristics of household members ##
            # Gender, age #
            'HV104', 'HV105',
            # Education
            'HV106', 'HV107', 'HV108','HV109', 'HV121', 'HV122', 'HV123',
            'HV124', 'HV125', 'HV126', 'HV127', 'HV128', 'HV129',
            ## [...] for women ##
            'HA66', 'HA67', 'HA68',
            ## [...] for men ##
            'HB66', 'HB67', 'HB68',
            ## [...] for children ##
            'HC61', 'HC62', 'HC68',

            ###  Women’s Individual Recode (IR) ###
            ## Basic characteristics of the women’s interview ##
            'V000', # country code and phase
            'V001', # cluster number
            'V002', # household number
            'V003', # respondent's line number
            'V004', # ultimate area unit
            'V006', # month of interview
            'V007', # year of interview
            'V012', # current age - respondent, respondent's current age
            'V013', # age in 5-year groups, age 5-year groups
            'V016', # day of interview
            'V024', # transmission zones, region, province
            'V034', # line number of husband
            ## Woman’s characteristics ##
            # Education #
            'V106', 'V107', 'V133', 'V149', 'V155', 'V156', 'V157', 'V158',
            ## Husband’s characteristics ##
            # Education #
            'V701', 'V702', 'V715', 'V729',

            ### Men’s Recode (MR) ###
            ## Basic characteristics of the men’s interview ##
            'MV000', # country code and phase
            'MV001', # cluster number
            'MV002', # household number
            'MV003', # respondent's line number
            'MV004', # ultimate area unit
            'MV008', # date of interview (cmc)
            'MV012', # current age, current age - respondent
            'MV013', # age in 5-year groups, age 5-year groups
            'MV016', # day of interview
            'MV024', # region, province
            ## Man’s characteristics ##
            # Education #
            'MV106', 'MV107', 'MV133', 'MV149', 'MV155', 'MV156', 'MV157', 'MV158'
            ]

#calculate separately for urban/rural
split_urban_rural = False
#calcualte only for available s2 data (PCA is performed on all HH data)
limit_calc_on_available_s2img = False
#limit by year NOT IMPLEMENTED
#min_year = False
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

###load data, preselect [...] columns and save into pickle for faster reloading
# Note: Important to use meta files since column names and especially value labels are unambiguous
# unify column names and replace values

def extract_survey_data(questionaire):

    dhs_d_all, country_d, data_file_types_d, typ_l, dhs_dirs_d = \
        dhs_f_education.load_dhs_data(dhs_path)
    pathes = {}

    # catch all files
    for i, types in dhs_d_all.items():
        # only extract files where Household Recode data is available
        if 'GE' in types and questionaire in types:
            if int(i[2]) >= min_dhs_version:
                for (dirrpath, dirrnames, filenames) in os.walk(dhs_dirs_d[i][types.index(questionaire)]):
                    for file in filenames:
                        if fnmatch.fnmatch(file, '*.sav') or fnmatch.fnmatch(file, '*.SAV'):
                            # also get GE folder for matching
                            splitted_p = os.path.normpath(dhs_dirs_d[i][types.index('GE')]).split(os.sep)
                            pathes[splitted_p[-1]] = dirrpath + '/' + file
    print(len(pathes), pathes)
    df_l = []
    meta_l = []

    # iterate over files and replace numerical values and cryptic column names with actual values
    for n, (ge_f, path) in enumerate(pathes.items()):
        print('________________________', n, '(', len(pathes), ')', ' __________________________________')
        print(path)
        print(ge_f)
        try:
            df, meta = pyreadstat.read_sav(path, encoding='LATIN1')
        except:
            print("Encoding Error:", path)
            continue

        # add GEID for matching files
        df["GEID"] = ge_f

        # only use relevant columns
        df = df[df.columns[df.columns.isin(codes_to_keep)]]

        df_l.append(df)

    #concatenating
    df = pd.concat(
        df_l,
        axis=0,
        join="outer",
        # ignore_index=True,
        # keys=None,
        # levels=None,
        # names=None,
        verify_integrity=False,
        # copy=True,
    )

    # safe as pickle
    df.to_pickle(projects_p + '/' + f"{questionaire}_pickle_tmp")

    return

for questionaire in questionaires:
    extract_survey_data(questionaire)
