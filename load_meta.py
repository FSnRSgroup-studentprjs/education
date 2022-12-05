def load_questionaires_meta(questionaire):
###load data, preselect water columns and save into pickle for faster reloading
# Note: Important to use meta files since column names and especially value labels are unambiguous
# unify column names and replace values
    column_names_d = {}
    bottled_d = {}
    counter = 0
    counter2 = {}

    dhs_d_all, country_d, data_file_types_d, typ_l, dhs_dirs_d = \
        dhs_f_education.load_dhs_data(dhs_path)

    pathes = {}
    df_l = []
    meta_l = []

    # catch all files
    for i, types in dhs_d_all.items():
        # only extract files where [...] is available
            if 'GE' in types and questionaire in types:
                if int(i[2]) >= min_dhs_version:
                    for (dirrpath, dirrnames, filenames) in os.walk(dhs_dirs_d[i][types.index(questionaire)]):
                        splitted_p = os.path.normpath(dhs_dirs_d[i][types.index('GE')]).split(os.sep)
                        for file in filenames:
                            if fnmatch.fnmatch(file, '*.sav') or fnmatch.fnmatch(file, '*.SAV'):
                                # also get GE folder for matching
                                pathes[splitted_p[-1]] = dirrpath + '/' + file

    column_codes = []
    column_names = []
    questionaire_type = []
    questionaire_version = []
    country_code = []

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

        v_s = set([])

        df["GEID"] = ge_f
        df.columns = df.columns.str.lower()


        new_columns = {}
        v_s = set([])
        for column_code, column_name in meta.column_names_to_labels.items():
            # to show and find all relevant columns to do: export to csv?
            # Note not all column codes and column names are uniquee and written consistently
            if column_name[:2].lower() != 'na' :
                print(column_code, column_name)
                column_codes.append(column_code)
                column_names.append(column_name)
                questionaire_type.append(path.split("/")[-1][2:4])
                questionaire_version.append(ge_f[4])
                country_code.append(ge_f[0:2])

        col_names_df = pd.DataFrame({'Questionaire_type': questionaire_type,
                                    'Questionaire_version': questionaire_version,
                                    'Country_code': country_code,
                                    'Column_codes': column_codes,
                                    'Column_names': column_names
                                    })

        df_l.append(col_names_df)
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

    print(df)
    df.to_csv(projects_p + '/' + f"{questionaire}_vars.csv")

    return
