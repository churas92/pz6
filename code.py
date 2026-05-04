import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    data = pd.read_csv('insurance.csv')
    print("Датасет успешно загружен.")
except FileNotFoundError:
    print("Файл 'insurance.csv' не найден. Проверьте путь к файлу.")

print("\nПервые 5 строк датасета:")
print(data.head())

print("\nИнформация о датасете:")
data.info()

print("\nСтатистическое описание числовых признаков:")
print(data.describe())

print("\nПропущенные значения в каждом столбце:")
print(data.isnull().sum())

categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']

target = 'charges'

X = data.drop(target, axis=1)
y = data[target]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nРазмер обучающей выборки: {X_train.shape[0]} записей")
print(f"Размер тестовой выборки: {X_test.shape[0]} записей")

linear_model_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

linear_model_pipe.fit(X_train, y_train)

y_pred_linear = linear_model_pipe.predict(X_test)

print("\n--- Линейная регрессия ---")
print(f"MAE (Средняя абсолютная ошибка): {mean_absolute_error(y_test, y_pred_linear):.2f}")
print(f"MSE (Среднеквадратичная ошибка): {mean_squared_error(y_test, y_pred_linear):.2f}")
print(f"RMSE (Корень из среднеквадратичной ошибки): {np.sqrt(mean_squared_error(y_test, y_pred_linear)):.2f}")
print(f"R² (Коэффициент детерминации): {r2_score(y_test, y_pred_linear):.4f}")

poly_model_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', LinearRegression())
])

poly_model_pipe.fit(X_train, y_train)

y_pred_poly = poly_model_pipe.predict(X_test)

print("\n--- Полиномиальная регрессия (степень 2) ---")
print(f"MAE (Средняя абсолютная ошибка): {mean_absolute_error(y_test, y_pred_poly):.2f}")
print(f"MSE (Среднеквадратичная ошибка): {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"RMSE (Корень из среднеквадратичной ошибки): {np.sqrt(mean_squared_error(y_test, y_pred_poly)):.2f}")
print(f"R² (Коэффициент детерминации): {r2_score(y_test, y_pred_poly):.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Фактические расходы (charges)')
plt.ylabel('Предсказанные расходы')
plt.title('Линейная регрессия')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_poly, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Фактические расходы (charges)')
plt.ylabel('Предсказанные расходы')
plt.title('Полиномиальная регрессия (степень 2)')
plt.grid(True)

plt.tight_layout()
plt.show()