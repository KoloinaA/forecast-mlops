def unistats(df):
    import pandas as pd
    output_df = pd.DataFrame(columns=['Count','Missing','Unique','Dtype','Numeric','Mode','Mean','Min','25%','Median','75%','Max','Std','Skew','Kurtosis'])

    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            output_df.loc[col] = [df[col].count(),df[col].isnull().sum(),df[col].nunique(), df[col].dtype, pd.api.types.is_numeric_dtype(df[col]),
                                  df[col].mode()[0], df[col].mean(), df[col].min(), df[col].quantile(0.25), df[col].median(), df[col].quantile(0.75),
                                  df[col].max(), df[col].std(), df[col].skew(), df[col].kurt()]
        else:
            output_df.loc[col] = [df[col].count(),df[col].isnull().sum(), df[col].nunique(),df[col].dtype, pd.api.types.is_numeric_dtype(df[col]), 
                                   df[col].mode()[0],'-','-','-','-','-','-','-','-','-' ]

    return output_df.sort_values(by=['Numeric', 'Skew' ,'Unique'], ascending=False)

def anova(df, feature, label):
    import pandas as pd
    import numpy as np
    from scipy import stats

    groups = df[feature].unique()
    #print(groups)
    df_grouped = df.groupby(feature)
    #print(df_grouped)
    group_labels = []

    for g in groups:
        g_list = df_grouped.get_group(g)
        #print(g_list)
        group_labels.append(g_list[label])
        #print(group_labels)

    return stats.f_oneway(*group_labels)


def bivstats(df, label):
    from scipy import stats
    import pandas as pd
    import numpy as np
    #Create an empty dataframe to store the output
    output_df = pd.DataFrame(columns= ['r', 'F', 'X2', 'p-value'])
    
    for col in df:
        if not col == label:
            if df[col].isnull().sum() == 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    r, p = stats.pearsonr(df[label] , df[col])
                    output_df.loc[col] = [round(r, 3),np.nan, np.nan,  round(p, 5)]
                else:
                    F, p = anova(df[[col, label]], col, label)
                    output_df.loc[col] = [np.nan, round(F, 3),np.nan,  round(p, 5)]
            else:
                output_df.loc[col] = [np.nan,np.nan,np.nan,'nulls']
    #output_df.sort_values(by=['r'], ascending= False)          
    return output_df.reindex(output_df.r.abs().sort_values(ascending=False).index)