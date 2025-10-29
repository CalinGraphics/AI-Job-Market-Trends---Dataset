import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurare stil vizualizare
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)

df = pd.read_csv('ai_job_market.csv')

# Crearea coloanei 'avg_salary_usd'
df[['min_salary_usd', 'max_salary_usd']] = df['salary_range_usd'].str.split('-', expand=True).astype(float)
df['avg_salary_usd'] = (df['min_salary_usd'] + df['max_salary_usd']) / 2

df['state'] = df['location'].apply(lambda x: x.split(', ')[-1] if ', ' in x else 'Unknown')

# Funcție pentru a extrage și număra toate elementele (skill-uri/tools)
def count_items(df, column_name):
    all_items = df[column_name].dropna().str.split(', ').explode().str.strip() 
    return all_items.value_counts()

print("Preprocesare finalizată: coloanele 'avg_salary_usd' și 'state' au fost create.\n")
print("## ANALIZA CELOR 8 ASPECTE ESENȚIALE (EDA)\n")

# 1. Distribuția Salariului Mediu
print("### 1. Distribuția Salariului Mediu (Statistici)")
print(df['avg_salary_usd'].describe().to_frame().to_string())

plt.figure(figsize=(10, 6))
sns.histplot(df['avg_salary_usd'], bins=30, kde=True, color='skyblue')
plt.title('1. Distribuția Salariului Mediu (USD)')
plt.xlabel('Salariu Mediu (USD)')
plt.ylabel('Frecvență')
plt.ticklabel_format(style='plain', axis='x')
plt.show()

# 2. Salariu Mediu în funcție de Nivelul de Experiență
experience_order = ['Internship', 'Entry', 'Mid', 'Senior']
print("\n### 2. Salariul Median vs. Nivel de Experiență")
salary_exp = df.groupby('experience_level')['avg_salary_usd'].median().reindex(experience_order)
print(salary_exp.to_frame(name='Salariu Median').to_string())

plt.figure(figsize=(9, 6))
sns.boxplot(x='experience_level', y='avg_salary_usd', data=df, order=experience_order, palette='viridis')
plt.title('2. Salariul Mediu în funcție de Nivelul de Experiență')
plt.xlabel('Nivel de Experiență')
plt.ylabel('Salariu Mediu (USD)')
plt.ticklabel_format(style='plain', axis='y')
plt.show()

# 3. Top 10 Competențe Solicitate (Skills)
top_10_skills = count_items(df, 'skills_required').nlargest(10)
print("\n### 3. Top 10 Competențe Solicitate (Frecvență)")
print(top_10_skills.to_frame(name='Frecvență').to_string())

plt.figure(figsize=(12, 7))
sns.barplot(x=top_10_skills.index, y=top_10_skills.values, palette='Greens_r')
plt.title('3. Top 10 Competențe Solicitate')
plt.xlabel('Competență')
plt.ylabel('Număr de Posturi care o Solicită')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 4. Corelația Salariu vs. Industrie vs. Mărime Companie (Heatmap)
print("\n### 4. Matricea Salariului Median (Industrie vs. Mărime Companie)")
pivot_industry_size = df.pivot_table(
    values='avg_salary_usd',
    index='industry',
    columns='company_size',
    aggfunc='median',
    fill_value=np.nan
)
print(pivot_industry_size.head().to_string())

plt.figure(figsize=(12, 10))
sns.heatmap(pivot_industry_size, annot=True, fmt=".0f", cmap="magma", linewidths=.5, cbar_kws={'label': 'Salariu Median (USD)'})
plt.title('4. Salariul Median în funcție de Industrie și Mărimea Companiei')
plt.xlabel('Mărimea Companiei')
plt.ylabel('Industrie')
plt.show()

# 5. Distribuția Joburilor pe Mărimea Companiei
print("\n### 5. Distribuția Posturilor pe Mărimea Companiei (Frecvență)")
company_counts = df['company_size'].value_counts()
print(company_counts.to_frame(name='Frecvență').to_string())

plt.figure(figsize=(7, 7))
plt.pie(company_counts, labels=company_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Purples_r'))
plt.title('5. Distribuția Posturilor pe Mărimea Companiei')
plt.tight_layout()
plt.show()

# 6. Analiza Top 5 Titluri de Post și Salariul Median
top_5_titles = df['job_title'].value_counts().nlargest(5).index
df_top_titles = df[df['job_title'].isin(top_5_titles)].groupby('job_title')['avg_salary_usd'].median().reset_index(name='Salariu Median')
freq_series = df['job_title'].value_counts()
df_top_titles['Frecvență'] = df_top_titles['job_title'].map(freq_series)
df_top_titles = df_top_titles.sort_values(by='Salariu Median', ascending=False)

print("\n### 6. Salariu Median și Frecvență pentru Top 5 Titluri:")
print(df_top_titles.to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(x='job_title', y='Salariu Median', data=df_top_titles, palette='rocket')
plt.title('6. Salariul Median vs. Top 5 Titluri de Post')
plt.xlabel('Titlu de Post')
plt.ylabel('Salariu Median (USD)')
plt.xticks(rotation=45, ha='right')
plt.ticklabel_format(style='plain', axis='y')
plt.show()

# 7. Salariul Median în funcție de Tipul de Angajare
print("\n### 7. Salariul Median vs. Tipul de Angajare")
salary_type = df.groupby('employment_type')['avg_salary_usd'].median().sort_values(ascending=False)
print(salary_type.to_frame(name='Salariu Median').to_string())

plt.figure(figsize=(8, 5))
sns.barplot(x=salary_type.index, y=salary_type.values, palette='RdYlBu')
plt.title('7. Salariul Median în funcție de Tipul de Angajare')
plt.xlabel('Tip de Angajare')
plt.ylabel('Salariu Median (USD)')
plt.ticklabel_format(style='plain', axis='y')
plt.show()

# 8. Analiza Geografică (Top 5 State/Regiuni)
top_5_states = df['state'].value_counts().nlargest(5).index
df_top_states_filtered = df[df['state'].isin(top_5_states)]

state_analysis = df_top_states_filtered.groupby('state')['avg_salary_usd'].median().reset_index(name='Median_Salary_USD')
state_counts = df['state'].value_counts()
state_analysis['Frecvență'] = state_analysis['state'].map(state_counts)
state_analysis = state_analysis.sort_values(by='Median_Salary_USD', ascending=False)

print("\n### 8. Salariu Median și Frecvență pentru Top 5 State/Regiuni:")
print(state_analysis.to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(x='Median_Salary_USD', y='state', data=state_analysis, palette='tab10')
plt.title('8. Salariul Median vs. Top 5 State/Regiuni')
plt.xlabel('Salariu Median (USD)')
plt.ylabel('Stat/Regiune')
plt.ticklabel_format(style='plain', axis='x')
plt.show()