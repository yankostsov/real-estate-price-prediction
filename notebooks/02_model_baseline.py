# %%
# Load the cleaned dataset and split it into training and test sets
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/processed/montpellier_apartments_clean.csv')

feature_columns = ['surface_reelle_bati', 'nombre_pieces_principales',
                    'code_postal', 'longitude', 'latitude', 'year']
X = df[feature_columns]
y = df['valeur_fonciere']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Train:', len(X_train), 'Test:', len(X_test))

# %%
# Train an XGBoost regressor and evaluate it on the held-out test set
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
# Check which features the model relies on most
importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
print(importance)

# %%
# Find the 10 individual test-set predictions with the largest errors
results = X_test.copy()
results['actual'] = y_test
results['predicted'] = predictions
results['error'] = (results['predicted'] - results['actual']).abs()
print(results.sort_values('error', ascending=False).head(10))

# %%
# Pull the full rows (including sale date) for the two most suspicious large-error sales
suspicious = df.loc[[3329, 4993]]
print(suspicious)

# %%
# Look up the exact street address of those two sales in the original raw data
raw_2023 = pd.read_csv('../data/raw/dvf_montpellier_2023.csv')
raw_2024 = pd.read_csv('../data/raw/dvf_montpellier_2024.csv')

match_2023 = raw_2023[(raw_2023['valeur_fonciere'] == 160000.0) & (raw_2023['surface_reelle_bati'] == 120.0)]
match_2024 = raw_2024[(raw_2024['valeur_fonciere'] == 425870.0) & (raw_2024['surface_reelle_bati'] == 109.0)]

print(match_2023[['adresse_numero', 'adresse_nom_voie', 'code_postal', 'date_mutation']])
print(match_2024[['adresse_numero', 'adresse_nom_voie', 'code_postal', 'date_mutation']])

# %%
# Save the trained model to disk so it can be reloaded without retraining
import joblib
import os

os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/xgboost_baseline.pkl')
print('Model saved')

# %%
# Reload the saved model and wrap it in a function that prices one apartment at a time
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
# Sanity-check predict_price on a real, known apartment (Yan's own)
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
# Convert a real street address into coordinates + postal code, using France's free geocoding API
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

# Quick test to confirm the geocoder is working
test_location = geocode_address("Place de la Comedie, Montpellier")
print(test_location)

# %%
# Combine geocoding + prediction: estimate a price from a street address alone
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
# Final end-to-end test: address in, price out
estimate_price_by_address("52 Rue de Syracuse, Montpellier", surface=45, rooms=2)