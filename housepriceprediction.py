import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data harga rumah di Jawa Timur dari BPS
data = {
    'Year': np.array([2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]),
    'Price': np.array([65000000, 77000000, 64000000, 50000000, 77000000, 128160000, 147390000, 
                       154640000, 114120000, 304400000, 170200000])
}

# Membuat DataFrame
df = pd.DataFrame(data)

# Membuat model regresi linier
model = LinearRegression()

# Mengubah Year menjadi array 2D
X = df['Year'].values.reshape(-1, 1)
y = df['Price'].values

# Melatih model
model.fit(X, y)

# Memprediksi harga rumah untuk 10 tahun mendatang (2019-2028)
future_years = np.arange(2019, 2029).reshape(-1, 1)
predictions = model.predict(future_years)

# Menyimpan prediksi ke DataFrame
future_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted Price': predictions})

# Plot hasil prediksi
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Price'], label='Historical Prices', marker='o')
plt.plot(future_df['Year'], future_df['Predicted Price'], label='Predicted Prices', marker='x', linestyle='--')
plt.xlabel('Year')
plt.ylabel('House Price (in IDR)')
plt.title('House Price Prediction in East Java')
plt.legend()
plt.grid(True)
plt.show()

# Menyimpan data prediksi ke file CSV
future_df.to_csv('house_price_predictions_east_java.csv', index=False)
