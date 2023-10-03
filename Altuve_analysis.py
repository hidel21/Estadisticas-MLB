import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import lxml
except ImportError:
    print("lxml is not installed. Please install it before running this script.")
    exit()


def calculate_stats_by_year(data, column_name):
    """Calcula estadísticas por año.

    Args:
        data: Un DataFrame de Pandas.
        column_name: El nombre de la columna para calcular las estadísticas.

    Returns:
        Una serie de Pandas con las estadísticas por año.
    """

    stats = data.groupby("Year")[column_name].sum()
    return stats


# Cargar datos
data = pd.DataFrame({
    "Year": ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"],
    "Hits": [61, 167, 177, 225, 200, 216, 204, 169, 149, 42, 167, 158, 112],
    "HR": [2, 7, 5, 7, 15, 24, 24, 13, 31, 5, 31, 28, 17]
})

# Convierte la columna 'Year' a números enteros
data["Year"] = pd.to_numeric(data["Year"])

# Filtra los datos para comenzar desde 2011
data = data[data["Year"] >= 2011]

# Calcular estadísticas por año
hits_year = calculate_stats_by_year(data, "Hits")
hr_year = calculate_stats_by_year(data, "HR")

# Crear gráficos
fig, axs = plt.subplots(2, 1, sharex=True)

# Graficar hits
axs[0].plot(hits_year, label="Hits")

# Graficar HR
axs[1].plot(hr_year, label="HR")

# Agregar títulos y etiquetas
axs[0].set_title("Hits por año")
axs[1].set_title("HR por año")
axs[1].set_xlabel("Año")

# Agregar leyendas
axs[0].legend()
axs[1].legend()

# Mostrar gráficos
plt.show()

# Análisis predictivo

# Pronosticar hits
hits_pred = np.polyfit(data["Year"], data["Hits"], 1)
hits_pred_year = np.polyval(hits_pred, data["Year"])

# Pronosticar HR
hr_pred = np.polyfit(data["Year"], data["HR"], 1)
hr_pred_year = np.polyval(hr_pred, data["Year"])

# Calcular probabilidad de alcanzar 3000 hits
prob_3000_hits = (hits_pred_year[-1] - hits_year[-1]) / (3000 - hits_year[-1])
print("Probabilidad de alcanzar 3000 hits:", prob_3000_hits)

# Calcular probabilidad de alcanzar 500 HR
prob_500_hr = (hr_pred_year[-1] - hr_year[-1]) / (500 - hr_year[-1])
print("Probabilidad de alcanzar 500 HR:", prob_500_hr)
