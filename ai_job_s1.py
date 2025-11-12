import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Implementare Naïve Bayes (BernoulliNB) pentru Clasificarea Salariului ---")

try:
    df = pd.read_csv('ai_job_market.csv')
except FileNotFoundError:
    print("Eroare: Fișierul 'ai_job_market.csv' nu a fost găsit. Asigurați-vă că este în același director.")
    exit()

# Crearea coloanei 'avg_salary_usd'
df[['min_salary_usd', 'max_salary_usd']] = df['salary_range_usd'].str.split('-', expand=True).astype(float)
df['avg_salary_usd'] = (df['min_salary_usd'] + df['max_salary_usd']) / 2

# Crearea Variabilei Țintă (Clasificare Salariu High/Low)

median_salary = df['avg_salary_usd'].median()
df['salary_class'] = np.where(df['avg_salary_usd'] > median_salary, 'High', 'Low')
print(f"\nSalariul Median folosit pentru clasificare: ${median_salary:,.2f}")

# Selectarea Caracteristicilor și Codificarea Variabilelor Categorice

features = ['job_title', 'industry', 'experience_level', 'employment_type', 'company_size']
X = df[features]
y = df['salary_class']

# Codificare One-Hot pentru variabilele categorice
X_encoded = pd.get_dummies(X, columns=features, drop_first=True)
print(f"Numărul de caracteristici după codificarea One-Hot: {X_encoded.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

# Antrenarea Modelului Naïve Bayes
# BernoulliNB este mai potrivit pentru date binare (one-hot encoded)
# GaussianNB presupune distribuție normală, ceea ce nu se aplică la date binare

nb_classifier = BernoulliNB()
nb_classifier.fit(X_train, y_train)

# 6. Evaluarea Modelului

y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Rezultate Evaluare Model Naïve Bayes ---")
print(f"Acuratețe: {accuracy:.4f}")
print("\nRaport de Clasificare (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred))

# Generarea Diagramei (Matricea de Confuzie)

cm = confusion_matrix(y_test, y_pred, labels=['Low', 'High'])
cm_df = pd.DataFrame(cm, 
                     index=['Actual Low', 'Actual High'], 
                     columns=['Predicted Low', 'Predicted High'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Matricea de Confuzie (Naïve Bayes)')
plt.ylabel('Valori Adevărate')
plt.xlabel('Valori Prezise')
plt.tight_layout()
plt.show()
print("Diagrama Matricei de Confuzie a fost generată.")
print("\nExecuție 'ai_job_s1.py' finalizată.")
