# %%
# Load each year separately, keeping track of which year it came from
import pandas as pd

df_2023 = pd.read_csv('../data/raw/dvf_montpellier_2023.csv')
df_2023['year'] = 2023

df_2024 = pd.read_csv('../data/raw/dvf_montpellier_2024.csv')
df_2024['year'] = 2024

df_2025 = pd.read_csv('../data/raw/dvf_montpellier_2025.csv')
df_2025['year'] = 2025

# Stack all three into one dataframe
df = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)

print('Total rows:', len(df))
print(df['year'].value_counts())

# %%
# How many mutations (sales) involve more than one row?
rows_per_sale = df.groupby('id_mutation').size()
print(rows_per_sale.value_counts().sort_index())

# %%
# What types of property are actually in this data, and how common is each?
print(df['type_local'].value_counts(dropna=False))

# %%
# Among apartment rows only, how many share the same sale (id_mutation)?
apartment_rows = df[df['type_local'] == 'Appartement']
apartments_per_sale = apartment_rows.groupby('id_mutation').size()
print(apartments_per_sale.value_counts().sort_index())

# %%
# Keep only sales where exactly one apartment was involved (excludes bulk/developer sales)
single_apartment_sales = apartments_per_sale[apartments_per_sale == 1].index
clean_df = apartment_rows[apartment_rows['id_mutation'].isin(single_apartment_sales)].copy()

print('Clean apartment sales:', len(clean_df))
print(clean_df[['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales']].describe())

# %%
# Drop rows with no price or no surface — unusable for training
clean_df = clean_df.dropna(subset=['valeur_fonciere', 'surface_reelle_bati'])

# Price per square meter — the standard real estate sanity-check metric
clean_df['price_per_sqm'] = clean_df['valeur_fonciere'] / clean_df['surface_reelle_bati']
print(clean_df['price_per_sqm'].describe())

# %%
# Domain floor from market knowledge — below this, not a genuine owner-occupier sale
clean_df = clean_df[clean_df['price_per_sqm'] >= 1000]

# Statistical trim as a secondary safety net — catches remaining oddities on the high end
lower = clean_df['price_per_sqm'].quantile(0.01)
upper = clean_df['price_per_sqm'].quantile(0.99)
clean_df = clean_df[(clean_df['price_per_sqm'] >= lower) & (clean_df['price_per_sqm'] <= upper)]

print('Remaining rows:', len(clean_df))
print(clean_df['price_per_sqm'].describe())

# %%
# How many distinct postal code zones do we have within Montpellier, and how are sales distributed?
print(clean_df['code_postal'].value_counts(dropna=False))

# Also check remaining missing values across the columns we actually care about
key_columns = ['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales',
                'code_postal', 'longitude', 'latitude', 'date_mutation']
print(clean_df[key_columns].isna().sum())

# %%
# Drop the remaining rows missing location data
clean_df = clean_df.dropna(subset=['code_postal', 'longitude', 'latitude'])

# Keep only the columns we'll actually use going forward
final_columns = ['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales',
                  'code_postal', 'longitude', 'latitude', 'year', 'date_mutation']
clean_df = clean_df[final_columns]

print('Final dataset:', len(clean_df), 'rows')
print(clean_df.dtypes)
print(clean_df.isna().sum())

# %%
# Fix column types, then save the final cleaned dataset to disk
import os

clean_df['nombre_pieces_principales'] = clean_df['nombre_pieces_principales'].astype(int)
clean_df['code_postal'] = clean_df['code_postal'].astype(int)

os.makedirs('../data/processed', exist_ok=True)
clean_df.to_csv('../data/processed/montpellier_apartments_clean.csv', index=False)
print('Saved:', len(clean_df), 'rows')