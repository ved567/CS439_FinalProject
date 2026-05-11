import pandas as pd
import numpy as np

df = pd.read_csv('startup_data_clean.csv')
df = df.drop(columns=['name', 'city', 'state_code', 'zip_code',
                       'latitude', 'longitude', 'category_code'])

def group_impute(df, col, group_col, method='median'):
    def impute_group(group):
        if method == 'median':
            fill = group[col].median()
        else:
            fill = group[col].mean()
        # fallback to global if entire group is null
        if pd.isna(fill):
            fill = df[col].median() if method == 'median' else df[col].mean()
        return group[col].fillna(fill)
    return df.groupby(group_col, group_keys=False).apply(impute_group)

category_cols = ['is_software','is_web','is_mobile','is_enterprise',
                 'is_advertising','is_gamesvideo','is_ecommerce',
                 'is_biotech','is_consulting','is_othercategory']

def get_category(row):
    for col in category_cols:
        if row[col] == 1:
            return col
    return 'other'

df['_category'] = df.apply(get_category, axis=1)

print("Nulls before imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])

df['age_first_milestone_year'] = group_impute(df, 'age_first_milestone_year',
                                               '_category', method='median')
df['age_last_milestone_year']  = group_impute(df, 'age_last_milestone_year',
                                               '_category', method='median')

df['sector_trend'] = group_impute(df, 'sector_trend',
                                   '_category', method='mean')

df = df.drop(columns=['_category'])

# Verify
assert df.isnull().sum().sum() == 0, "Still has nulls!"
print("\nClean. Shape:", df.shape)
print("\nFinal columns:", df.columns.tolist())
print("\nSample stats:")
print(df[['age_first_milestone_year', 'age_last_milestone_year', 'sector_trend']].describe().round(3))

df.to_csv('startup_data_final.csv', index=False)
print("\n  Saved → startup_data_final.csv")
