# %%
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/processed/montpellier_apartments_clean.csv')

# Features the model will learn from
feature_columns = ['surface_reelle_bati', 'nombre_pieces_principales',
                    'code_postal', 'longitude', 'latitude', 'year']
X = df[feature_columns]
y = df['valeur_fonciere']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Train:', len(X_train), 'Test:', len(X_test))
# %%
# %%
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print(f'MAE:  {mae:,.0f} EUR')
print(f'RMSE: {rmse:,.0f} EUR')
print(f'R2:   {r2:.3f}')
# %%
# %%
import pandas as pd

# Which features does the model actually rely on?
importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
print(importance)

# %%
# The worst individual predictions — where does the model struggle most?
results = X_test.copy()
results['actual'] = y_test
results['predicted'] = predictions
results['error'] = (results['predicted'] - results['actual']).abs()
print(results.sort_values('error', ascending=False).head(10))
# %%
# %%
# Check the two suspicious rows against the original cleaned dataset
suspicious = df.loc[[3329, 4993]]
print(suspicious)
# %%
# %%
raw_2023 = pd.read_csv('../data/raw/dvf_montpellier_2023.csv')
raw_2024 = pd.read_csv('../data/raw/dvf_montpellier_2024.csv')

match_2023 = raw_2023[(raw_2023['valeur_fonciere'] == 160000.0) & (raw_2023['surface_reelle_bati'] == 120.0)]
match_2024 = raw_2024[(raw_2024['valeur_fonciere'] == 425870.0) & (raw_2024['surface_reelle_bati'] == 109.0)]

print(match_2023[['adresse_numero', 'adresse_nom_voie', 'code_postal', 'date_mutation']])
print(match_2024[['adresse_numero', 'adresse_nom_voie', 'code_postal', 'date_mutation']])
# %%
# %%
import joblib
import os

os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/xgboost_baseline.pkl')
print('Model saved')

# %%
# %%
import joblib

model = joblib.load('../models/xgboost_baseline.pkl')

def predict_price(surface, rooms, postal_code, longitude, latitude, year=2025):
    input_data = pd.DataFrame([{
        'surface_reelle_bati': surface,
        'nombre_pieces_principales': rooms,
        'code_postal': postal_code,
        'longitude': longitude,
        'latitude': latitude,
        'year': year
    }])
    return model.predict(input_data)[0]

# %%
# %%
your_apartment = predict_price(
    surface=45,
    rooms=2,
    postal_code=34000,
    longitude=3.874682321038496,
    latitude=43.60442425169727,
    year=2025
)
print(f'Estimated price: {your_apartment:,.0f} EUR')
print(f'Price per sqm: {your_apartment/45:,.0f} EUR/m2')
# %%
# %%
import requests

def geocode_address(address):
    response = requests.get(
        'https://data.geopf.fr/geocodage/search',
        params={'q': address, 'limit': 1}
    )
    data = response.json()

    if not data['features']:
        raise ValueError(f'Address not found: {address}')

    feature = data['features'][0]
    longitude, latitude = feature['geometry']['coordinates']

    return {
        'longitude': longitude,
        'latitude': latitude,
        'postal_code': int(feature['properties']['postcode']),
        'matched_address': feature['properties']['label'],
        'confidence': feature['properties']['score']
    }

# Quick test
test_location = geocode_address("Place de la Comedie, Montpellier")
print(test_location)
# %%
def estimate_price_by_address(address, surface, rooms, year=2025):
    location = geocode_address(address)

    if location['confidence'] < 0.5:
        print(f"Warning: low confidence match ({location['confidence']:.2f}) for '{address}'")

    price = predict_price(
        surface=surface,
        rooms=rooms,
        postal_code=location['postal_code'],
        longitude=location['longitude'],
        latitude=location['latitude'],
        year=year
    )

    print(f"Address matched: {location['matched_address']}")
    print(f"Estimated price: {price:,.0f} EUR")
    print(f"Price per sqm: {price/surface:,.0f} EUR/m2")

    return price
# %%
# %%
estimate_price_by_address("02 Rue de Syracuse, Montpellier", surface=45, rooms=2)
# %%
