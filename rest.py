   print('1')
    # replace int values with meaningful names
    variable_values_d = {}
    for col_n in new_columns.keys():
        if col_n in meta.variable_value_labels.keys():
            variable_values_d[col_n] = {}
            for k, v in meta.variable_value_labels[col_n].items():
                if type(v) == str:
                    variable_values_d[col_n][k] = v.lower()
                else:
                    variable_values_d[col_n][k] = v

    df = df.replace(variable_values_d)
    # only use matched columns
    df = df[df.columns.intersection(new_columns.keys())]
    # rename columns to uniform names
    df = df.rename(columns=new_columns)
    # add GEID for matching files
    df["GEID"] = ge_f
    #         df = df.reset_index()
    df_l.append(df)

#print('final bottled counter', counter)
#print('final bottled counter', counter2)
#for k, v in counter2.items():
    #print(k, v)
#for k, v in bottled_d.items():
    #print(k, v)
    #         print(df)

with open(pickle_f, 'wb') as pf:
    pickle.dump(df_l, pf)
