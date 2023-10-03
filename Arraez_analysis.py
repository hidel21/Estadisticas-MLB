import pandas as pd
import matplotlib.pyplot as plt


# Cargar datos
data = pd.read_csv("data/luis_arraez.csv")
data = data[data['Year'].apply(lambda x: str(x).isdigit())]
data['Year'] = data['Year'].astype(int)
data_recent = data[data['Year'].isin([2022, 2023])]

# Calcular los hits anuales promedio de 2022 y 2023
avg_hits_recent = data_recent['H'].mean()

# Asumir una carrera de 20 años en total
career_years = 20
remaining_years = career_years - len(data)

# Estimar hits totales en los próximos años
predicted_hits_next_years = avg_hits_recent * remaining_years
total_hits_end_of_career = data['H'].sum() + predicted_hits_next_years

# Mostrar datos
print(f"Promedio de hits en 2022 y 2023: {avg_hits_recent}")
print(f"Estimación de hits totales al final de su carrera: {total_hits_end_of_career}")

# Graficar hits anuales y proyección
plt.figure(figsize=(12, 6))
plt.plot(data['Year'], data['H'], marker='o', label='Hits reales por año')
plt.axhline(y=avg_hits_recent, color='r', linestyle='--', label='Promedio de hits basado en 2022 y 2023')
plt.xlabel('Año')
plt.ylabel('Hits')
plt.title('Hits anuales de Luis Arraez y promedio basado en 2022 y 2023')
plt.legend()
plt.grid(True)
plt.show()