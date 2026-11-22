# EXPERIMENT NO.4 : MULTIPLE REGRESION ANALYSIS (FISH DATASET)

## AIM

To perform Multiple Linear Regression on a dataset imported from a CSV file to analyze how multiple features together predict a dependent variable.

## DATASET

The dataset used for this experiment is `fish.csv` which contains real-world physical measurements of various fish species:
* **Independent Variables ($X$):**
  * `L`: Mean length derived from `Length1`, `Length2`, and `Length3`
  * `Height`: Vertical height of the fish
  * `Width`: Diagonal width of the fish
* **Dependent Variable ($Y$):**
  * `Weight`: Weight of the fish (in grams)

## ALGORITHM/STEPS FOLLOWED

1. Import the required Python libraries (`numpy`, `pandas`, `matplotlib`, `scikit-learn`).
2. Load the fish dataset into a Pandas DataFrame using `pd.read_csv()`.
3. Display the dataset and inspect its columns.
4. Define the dependent ($Y$) and independent ($X$) variables.
5. Train the Multiple Linear Regression model using `LinearRegression()`.
6. Display the model coefficients (`model.coef_`), intercept (`model.intercept_`), and regression equation.
7. Predict target variable (`Weight`) values using `model.predict()`.
8. Evaluate model accuracy using the $R^2$ score (`r2_score()`).
9. Plot actual vs. predicted values for visual comparison.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Creation / Handling** | `pd.read_csv()` | Load the CSV dataset into a Pandas DataFrame |
| | `np.array()` / `pd.DataFrame()` | Create and manage numerical arrays and tabular data for regression |
| **Model Building (Scikit-learn)** | `LinearRegression()` | Initialize the Multiple Linear Regression model |
| | `model.fit()` | Train the regression model using multiple independent variables ($X$) and target ($Y$) |
| | `model.predict()` | Predict dependent variable values ($Y$) using the trained model |
| | `r2_score()` | Evaluate the accuracy of the regression model using the $R^2$ score |
| **Model Attributes** | `model.intercept_` | Retrieve the intercept (constant term) of the regression equation |
| | `model.coef_` | Retrieve the coefficients (slopes) for each independent variable |
| **Visualization (Matplotlib)** | `plt.plot()` | Plot actual and predicted weight values for comparison |
| | `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.legend()`, `plt.show()` | Configure titles, axis labels, legend, and display the plotted figure |

## PROGRAM & OUTPUT

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import r2_score  
from google.colab import drive  
drive.mount('/content/drive')  
file_path = '/content/drive/My Drive/Fish_length.csv'  
df = pd.read_csv(file_path)  
print("--- 1. Dataset Loaded Successfully ---")  
print(df.head())  
print(f"\nDataset Shape (Rows, Columns): {df.shape}")  
df['L'] = (df['Length1'] + df['Length2'] + df['Length3']) / 3  
X = df[['L', 'Height', 'Width']]  
y = df['Weight']  
model = LinearRegression()  
model.fit(X, y)  
intercept = model.intercept_  
coefficients = model.coef_  
print("\n--- 2. Model Parameters ---")  
print(f"Intercept (a): {intercept:.4f}")  
print(f"Coefficients (b1, b2, b3):")  
for col, coef in zip(X.columns, coefficients):  
    print(f"  {col}: {coef:.4f}")  
print(f"\nRegression Equation:")  
print(f"Weight = ({coefficients[0]:.4f} * L) + ({coefficients[1]:.4f} * Height) + ({coefficients[2]:.4f} * Width) + ({intercept:.4f})")  
y_pred = model.predict(X)  
r2 = r2_score(y, y_pred)  
print(f"\n--- 3. Model Performance ---")  
print(f"R-squared (R^2) Score: {r2:.4f}")  
plt.figure(figsize=(8, 5))  
plt.plot(y.values[:30], label='Actual Weight', color='blue', marker='o', linestyle='-')  
plt.plot(y_pred[:30], label='Predicted Weight', color='red', marker='x', linestyle='--')  
plt.title('Multiple Linear Regression: Actual vs Predicted Fish Weight')  
plt.xlabel('Sample Index (First 30 Fish)')  
plt.ylabel('Weight (grams)')  
plt.legend()  
plt.grid(True, linestyle='--', alpha=0.6)  
plt.tight_layout()  
plt.show()  

<img width="460" height="261" alt="image" src="https://github.com/user-attachments/assets/6884a821-3a9f-408c-bba3-e6b1074256e0" />
<img width="581" height="45" alt="image" src="https://github.com/user-attachments/assets/e84488ca-5bef-4c2d-8c87-92f60bdcb992" />
<img width="752" height="497" alt="image" src="https://github.com/user-attachments/assets/d4bb2e70-a76b-43c1-91ef-97c294b933ef" />


## RESULT

The model predicts fish weight with high accuracy ($R^2 = 0.9140$$R^2 = 0.9140$) using its average length, height, and width, 
Both length and height contribute positively to the predicted weight, while width has a negative coefficient in this combination.
