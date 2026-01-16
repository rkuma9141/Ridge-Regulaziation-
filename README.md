# Ridge Regularization (Ridge Regression)

Ridge Regularization, also known as **Ridge Regression** or **L2 Regularization**, is a technique used in machine learning to reduce overfitting in linear models by penalizing large coefficients.

---

## 📌 Why Ridge Regularization?

In linear regression, models can overfit when:
- Features are highly correlated (multicollinearity)
- The number of features is large
- The model learns noise from the training data

Ridge regularization addresses this by **shrinking coefficients**, making the model more stable and generalizable.

---

## 🧮 Mathematical Formulation

Standard Linear Regression minimizes:

\[
\text{Loss} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

Ridge Regression adds an **L2 penalty**:

\[
\text{Loss} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} w_j^2
\]

Where:
- \( y_i \) = actual value  
- \( \hat{y}_i \) = predicted value  
- \( w_j \) = model coefficients  
- \( \lambda \) = regularization strength  

---

## ⚙️ Key Characteristics

- Uses **L2 norm** (squared coefficients)
- Shrinks coefficients toward zero (but **never exactly zero**)
- Helps with multicollinearity
- Improves model generalization

---

## 🔧 Hyperparameter: Lambda (λ)

- **λ = 0** → Equivalent to Linear Regression  
- **Small λ** → Slight regularization  
- **Large λ** → Strong regularization (more shrinkage)

Choosing λ is usually done using **cross-validation**.

---

## 🆚 Ridge vs Lasso

| Feature | Ridge | Lasso |
|------|------|------|
| Regularization | L2 | L1 |
| Coefficients | Shrinks | Can become zero |
| Feature Selection | ❌ No | ✅ Yes |
| Multicollinearity | ✅ Handles well | ⚠️ Less stable |

---

## 🧪 Example (Python – scikit-learn)

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=10, noise=10)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

print("Coefficients:", model.coef_)
print("Score:", model.score(X_test, y_test))
