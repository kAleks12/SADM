import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
import statsmodels.api as sm
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- Konfiguracja Wyświetlania ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.style.use('seaborn-v0_8-darkgrid')  # Użyj dostępnego stylu seaborn


# --- Funkcje Pomocnicze ---
def run_analysis_section(title):
    print("\n" + "=" * 80)
    print(f"--- {title} ---")
    print("=" * 80 + "\n")


def interpret_p_value(p_val, alpha=0.05):
    if p_val < alpha:
        print(f"p = {p_val:.4f} < {alpha}. Odrzucamy hipotezę zerową (H0). Wynik istotny statystycznie.")
    else:
        print(
            f"p = {p_val:.4f} >= {alpha}. Brak podstaw do odrzucenia hipotezy zerowej (H0). Wynik nieistotny statystycznie.")


# === ZBIÓR DANYCH 1: Heart Failure Prediction ===

run_analysis_section("Analiza Zbioru Danych: Heart Failure Prediction")

# --- Ładowanie Danych ---
try:
    df_heart = pd.read_csv(os.path.join('data', 'heart_failure_clinical_records_dataset.csv'))
    print("Załadowano dane Heart Failure Prediction.")
    print(df_heart.head())
except FileNotFoundError:
    print(
        "BŁĄD: Plik 'heart_failure_clinical_data_dataset.csv' nie znaleziony. Pobierz go i umieść w folderze skryptu.")
    exit()  # Zakończ jeśli nie ma pliku

# --- 1. Statystyka opisowa ---
run_analysis_section("1. Statystyka Opisowa (Heart Failure)")
print("Podstawowe statystyki opisowe dla zmiennych numerycznych:")
print(df_heart.describe())
print("\nLiczności dla zmiennych kategorycznych/binarnych:")
for col in ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']:
    print(f"\n{col}:")
    print(df_heart[col].value_counts(normalize=True))

# Wizualizacje (przykłady)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_heart['age'], kde=True)
plt.title('Rozkład Wieku')
plt.subplot(1, 2, 2)
sns.boxplot(x='DEATH_EVENT', y='ejection_fraction', data=df_heart)
plt.title('Frakcja wyrzutowa vs Zgon')
plt.xticks([0, 1], ['Przeżył', 'Zmarł'])
plt.tight_layout()
plt.show()

# --- 2. Testy parametryczne (min 2) ---
run_analysis_section("2. Testy Parametryczne (Heart Failure)")

# Test 1: t-test dla prób niezależnych (age vs DEATH_EVENT)
print("\nTest t: Średni wiek vs Zdarzenie śmierci (DEATH_EVENT)")
group_survived = df_heart[df_heart['DEATH_EVENT'] == 0]['age']
group_died = df_heart[df_heart['DEATH_EVENT'] == 1]['age']
# Sprawdzenie wariancji (opcjonalne, wpływa na parametr equal_var)
levene_stat, levene_p = stats.levene(group_survived, group_died)
print(f"Test Levene'a dla wariancji wieku: stat={levene_stat:.4f}, p={levene_p:.4f}")
equal_var_flag = levene_p >= 0.05

t_stat, p_val = stats.ttest_ind(group_survived, group_died, equal_var=equal_var_flag, nan_policy='omit')
print(f"H0: Średni wiek jest taki sam w obu grupach.")
print(f"H1: Średni wiek różni się między grupami.")
print(f"Statystyka T = {t_stat:.4f}")
interpret_p_value(p_val)

# Test 2: t-test dla prób niezależnych (ejection_fraction vs smoking)
print("\nTest t: Średnia frakcja wyrzutowa vs Palenie (smoking)")
group_nonsmoker = df_heart[df_heart['smoking'] == 0]['ejection_fraction']
group_smoker = df_heart[df_heart['smoking'] == 1]['ejection_fraction']
levene_stat_ef, levene_p_ef = stats.levene(group_nonsmoker.dropna(), group_smoker.dropna())
print(f"Test Levene'a dla wariancji frakcji wyrzutowej: stat={levene_stat_ef:.4f}, p={levene_p_ef:.4f}")
equal_var_flag_ef = levene_p_ef >= 0.05

t_stat_ef, p_val_ef = stats.ttest_ind(group_nonsmoker, group_smoker, equal_var=equal_var_flag_ef, nan_policy='omit')
print(f"H0: Średnia frakcja wyrzutowa jest taka sama dla palących i niepalących.")
print(f"H1: Średnia frakcja wyrzutowa różni się między palącymi a niepalącymi.")
print(f"Statystyka T = {t_stat_ef:.4f}")
interpret_p_value(p_val_ef)

# --- 3. Testy nieparametryczne (min 2) ---
run_analysis_section("3. Testy Nieparametryczne (Heart Failure)")

# Test 1: U Manna-Whitneya (serum_creatinine vs DEATH_EVENT)
print("\nTest U Manna-Whitneya: Stężenie kreatyniny vs Zdarzenie śmierci")
group_survived_sc = df_heart[df_heart['DEATH_EVENT'] == 0]['serum_creatinine']
group_died_sc = df_heart[df_heart['DEATH_EVENT'] == 1]['serum_creatinine']
u_stat, p_val_mw = stats.mannwhitneyu(group_survived_sc, group_died_sc, alternative='two-sided', nan_policy='omit')
print(f"H0: Rozkłady stężenia kreatyniny są takie same w obu grupach.")
print(f"H1: Rozkłady stężenia kreatyniny różnią się między grupami.")
print(f"Statystyka U = {u_stat:.4f}")
interpret_p_value(p_val_mw)

# Test 2: Chi-kwadrat (diabetes vs high_blood_pressure)
print("\nTest Chi-kwadrat: Zależność Cukrzyca vs Nadciśnienie")
contingency_table = pd.crosstab(df_heart['diabetes'], df_heart['high_blood_pressure'])
print("Tabela kontyngencji:")
print(contingency_table)
chi2_stat, p_val_chi, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nH0: Występowanie cukrzycy jest niezależne od występowania nadciśnienia.")
print(f"H1: Istnieje zależność między cukrzycą a nadciśnieniem.")
print(f"Statystyka Chi^2 = {chi2_stat:.4f}, df = {dof}")
interpret_p_value(p_val_chi)

# --- 4. Test Friedmana rank + analiza post-hoc + wizualizacja ---
run_analysis_section("4. Test Friedmana (Heart Failure)")
print("Test Friedmana nie ma zastosowania do tego zbioru danych - brak pomiarów powtarzanych.")

# --- 5. Krzywe przeżycia – wykres Kaplana Mayera ---
run_analysis_section("5. Analiza Przeżycia - Kaplan-Meier (Heart Failure)")

kmf = KaplanMeierFitter()

# Ogólna krzywa przeżycia
T = df_heart['time']  # Czas do zdarzenia lub cenzurowania
E = df_heart['DEATH_EVENT']  # Status zdarzenia (1=zgon, 0=cenzurowane)

kmf.fit(T, event_observed=E, label='Ogół Pacjentów')
ax = kmf.plot(figsize=(10, 6))
ax.set_title('Ogólna Krzywa Przeżycia Pacjentów z Niewydolnością Serca')
ax.set_xlabel('Czas (dni)')
ax.set_ylabel('Prawdopodobieństwo Przeżycia')
plt.show()

# Porównanie grup (np. anemia vs brak anemii)
ax = plt.subplot(111)
plt.title('Krzywe Przeżycia wg Statusu Anemii')

anaemia_status = df_heart['anaemia'] == 1

kmf.fit(T[anaemia_status], event_observed=E[anaemia_status], label='Anemia (1)')
kmf.plot(ax=ax)

kmf.fit(T[~anaemia_status], event_observed=E[~anaemia_status], label='Brak Anemii (0)')
kmf.plot(ax=ax)

plt.xlabel('Czas (dni)')
plt.ylabel('Prawdopodobieństwo Przeżycia')
plt.show()

# Test log-rank do porównania grup (anemia vs brak anemii)
results_logrank = logrank_test(T[anaemia_status], T[~anaemia_status], E[anaemia_status], E[~anaemia_status])
print("\nTest Log-Rank porównujący grupy wg statusu anemii:")
print(f"H0: Funkcje przeżycia są takie same dla pacjentów z anemią i bez anemii.")
print(f"H1: Funkcje przeżycia różnią się istotnie między grupami.")
results_logrank.print_summary()
interpret_p_value(results_logrank.p_value)

# --- 6. Regresja logistyczna ---
run_analysis_section("6. Regresja Logistyczna (Heart Failure)")

# Przygotowanie danych
X = df_heart.drop(['time', 'DEATH_EVENT'],
                  axis=1)
y = df_heart['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
numerical_features = X.select_dtypes(include=np.number).columns

preprocessor = Pipeline(steps=[('scaler', StandardScaler())])
log_reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', LogisticRegression(solver='liblinear', random_state=42))])
log_reg_pipeline.fit(X_train, y_train)

print("\nOcena modelu regresji logistycznej na zbiorze testowym:")
y_pred = log_reg_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

model_coefs = log_reg_pipeline.named_steps['classifier'].coef_[0]
model_intercept = log_reg_pipeline.named_steps['classifier'].intercept_[0]
feature_names = numerical_features  # Wszystkie są numeryczne
coef_df = pd.DataFrame({'Cecha': feature_names, 'Współczynnik': model_coefs})

print("\nWspółczynniki modelu regresji logistycznej:")
print(coef_df.sort_values(by='Współczynnik', key=abs, ascending=False))
print(f"Wyraz wolny (Intercept): {model_intercept:.4f}")


# --- 7. Wizualizacja jakości klasyfikacji – ROC ---
run_analysis_section("7. Krzywa ROC (Heart Failure)")

y_pred_proba = log_reg_pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"\nPole pod krzywą ROC (AUC): {roc_auc:.4f}")

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Krzywa ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Losowy klasyfikator')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Odsetek fałszywie pozytywnych (False Positive Rate)')
plt.ylabel('Odsetek prawdziwie pozytywnych (True Positive Rate)')
plt.title('Krzywa ROC dla modelu przewidującego DEATH_EVENT')
plt.legend(loc="lower right")
plt.show()

# ===== KONIEC ANALIZY ZBIORU 1 =====


# === ZBIÓR DANYCH 2: Student Performance ===

run_analysis_section("Analiza Zbioru Danych: Student Performance (j. portugalski)")

# --- Ładowanie Danych ---
try:
    df_student = pd.read_csv(os.path.join('data', 'student-por.csv'))
    print("Załadowano dane Student Performance (student-por.csv).")

    print("\nKonwersja kolumn G1, G2, G3 na typ numeryczny (błędy zamieniane na NaN)...")
    cols_to_convert = ['G1', 'G2', 'G3']
    for col in cols_to_convert:
        if col in df_student.columns:
            original_dtype = df_student[col].dtype
            df_student[col] = pd.to_numeric(df_student[col], errors='coerce')
            if original_dtype != df_student[col].dtype:
                print(f"Kolumna '{col}' została przekonwertowana na typ {df_student[col].dtype}.")
            # Sprawdź, czy pojawiły się NaN w wyniku konwersji
            if df_student[col].isna().any():
                # Możesz zliczyć ile NaN powstało: df_student[col].isna().sum()
                print(f"UWAGA: Kolumna '{col}' zawiera teraz wartości NaN po konwersji.")
        else:
            print(f"Ostrzeżenie: Kolumna '{col}' nie znaleziona w DataFrame.")
    print("Sprawdzenie typów po konwersji:")
    print(df_student[cols_to_convert].dtypes)

    print(df_student.head())
except FileNotFoundError:
    print("BŁĄD: Plik 'student-por.csv' nie znaleziony. Pobierz go i umieść w folderze skryptu.")
    exit()
except Exception as e:
    print(f"BŁĄD podczas ładowania lub konwersji pliku student-por.csv: {e}")
    exit()

for col in ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']:
    if col in df_student.columns:
        df_student[col] = df_student[col].map({'yes': 1, 'no': 0})

# --- 1. Statystyka opisowa ---
run_analysis_section("1. Statystyka Opisowa (Student Performance)")
print("Podstawowe statystyki opisowe dla zmiennych numerycznych:")
print(df_student.describe())
print("\nLiczności dla wybranych zmiennych kategorycznych:")
for col in ['sex', 'Mjob', 'Fjob', 'reason', 'guardian', 'paid', 'internet']:
    if col in df_student.columns:
        print(f"\n{col}:")
        print(df_student[col].value_counts(normalize=True).head())  # .head() dla zwięzłości


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_student['G3'], bins=20, kde=False)
plt.title('Rozkład Oceny Końcowej (G3)')
plt.subplot(1, 2, 2)
sns.boxplot(x='studytime', y='G3', data=df_student)
plt.title('Ocena Końcowa (G3) vs Czas Nauki')
plt.xlabel("Czas nauki (1:<2h, 2:2-5h, 3:5-10h, 4:>10h)")
plt.tight_layout()
plt.show()

# --- 2. Testy parametryczne (min 2) ---
run_analysis_section("2. Testy Parametryczne (Student Performance)")

# Test 1: t-test dla prób niezależnych (G3 vs sex)
print("\nTest t: Średnia ocena końcowa (G3) vs Płeć (sex)")
group_F = df_student[df_student['sex'] == 'F']['G3']
group_M = df_student[df_student['sex'] == 'M']['G3']
levene_stat_g3s, levene_p_g3s = stats.levene(group_F.dropna(), group_M.dropna())
print(f"Test Levene'a dla wariancji G3 vs sex: stat={levene_stat_g3s:.4f}, p={levene_p_g3s:.4f}")
equal_var_flag_g3s = levene_p_g3s >= 0.05

t_stat_g3s, p_val_g3s = stats.ttest_ind(group_F, group_M, equal_var=equal_var_flag_g3s, nan_policy='omit')
print(f"H0: Średnia ocena G3 jest taka sama dla obu płci.")
print(f"H1: Średnia ocena G3 różni się między płciami.")
print(f"Statystyka T = {t_stat_g3s:.4f}")
interpret_p_value(p_val_g3s)

# Test 2: Test t dla prób zależnych (G1 vs G2)
print("\nTest t: Porównanie średnich ocen G1 vs G2")
g1 = df_student['G1']
g2 = df_student['G2']

valid_indices = g1.notna() & g2.notna()
t_stat_g1g2, p_val_g1g2 = stats.ttest_rel(g1[valid_indices], g2[valid_indices])
print(f"H0: Średnia ocena G1 jest równa średniej ocenie G2.")
print(f"H1: Średnie oceny G1 i G2 różnią się istotnie.")
print(f"Statystyka T = {t_stat_g1g2:.4f}")
interpret_p_value(p_val_g1g2)

# Opcjonalnie Test 3: ANOVA (G3 vs studytime)
print("\nANOVA: Ocena końcowa (G3) vs Czas nauki (studytime)")
model_anova = ols('G3 ~ C(studytime)', data=df_student).fit()
anova_table = sm.stats.anova_lm(model_anova, typ=2)
print(anova_table)
p_val_anova = anova_table['PR(>F)'][0]
print(f"\nH0: Średnia ocena G3 jest taka sama dla wszystkich grup czasu nauki.")
print(f"H1: Co najmniej jedna grupa różni się średnią oceną G3.")
interpret_p_value(p_val_anova)

# Jeśli ANOVA istotna, można dodać test post-hoc Tukeya
if p_val_anova < 0.05:
    print("\nTest post-hoc Tukeya HSD (dla ANOVA):")
    tukey_results = pairwise_tukeyhsd(df_student['G3'], df_student['studytime'], alpha=0.05)
    print(tukey_results)

# --- 3. Testy nieparametryczne (min 2) ---
run_analysis_section("3. Testy Nieparametryczne (Student Performance)")

# Test 1: U Manna-Whitneya (absences vs paid)
print("\nTest U Manna-Whitneya: Liczba nieobecności vs Płatne zajęcia")
group_paid_no = df_student[df_student['paid'] == 0]['absences']
group_paid_yes = df_student[df_student['paid'] == 1]['absences']
u_stat_abs, p_val_abs = stats.mannwhitneyu(group_paid_no, group_paid_yes, alternative='two-sided', nan_policy='omit')
print(f"H0: Rozkłady liczby nieobecności są takie same dla obu grup (paid yes/no).")
print(f"H1: Rozkłady liczby nieobecności różnią się między grupami.")
print(f"Statystyka U = {u_stat_abs:.4f}")
interpret_p_value(p_val_abs)

# Test 2: Kruskala-Wallisa (G3 vs studytime) - nieparametryczny odpowiednik ANOVA
print("\nTest Kruskala-Wallisa: Ocena końcowa (G3) vs Czas nauki (studytime)")
groups_kw = [df_student['G3'][df_student['studytime'] == i] for i in sorted(df_student['studytime'].unique())]
groups_kw_clean = [group.dropna() for group in groups_kw]


if len(groups_kw_clean) < 2 or any(len(g) == 0 for g in groups_kw_clean):
    print("Nie można przeprowadzić testu Kruskala-Wallisa - niewystarczająca liczba grup lub dane.")
else:
    kw_stat, p_val_kw = stats.kruskal(*groups_kw_clean)  # * rozpakowuje listę grup
    print(f"H0: Rozkład oceny G3 jest taki sam dla wszystkich grup czasu nauki.")
    print(f"H1: Co najmniej jedna grupa różni się rozkładem oceny G3.")
    print(f"Statystyka H = {kw_stat:.4f}")
    interpret_p_value(p_val_kw)

    # Opcjonalnie Test post-hoc po Kruskalu-Wallisie (np. Dunn's test) jeśli istotny
    if p_val_kw < 0.05:
        print("\nTest post-hoc Dunna (dla Kruskala-Wallisa):")
        dunn_results = sp.posthoc_dunn(df_student, val_col='G3', group_col='studytime', p_adjust='bonferroni')
        print(dunn_results)

# --- 4. Test Friedmana rank + analiza post-hoc + wizualizacja ---
run_analysis_section("4. Test Friedmana (G1, G2, G3) (Student Performance)")

g1 = df_student['G1']
g2 = df_student['G2']
g3 = df_student['G3']

if g1.isna().any() or g2.isna().any() or g3.isna().any():
    print(
        "Uwaga: Istnieją braki danych w G1, G2 lub G3. Test Friedmana zostanie przeprowadzony na kompletnych przypadkach.")

friedman_stat, p_val_friedman = stats.friedmanchisquare(g1, g2, g3)

print(f"H0: Rozkłady ocen G1, G2, G3 są takie same.")
print(f"H1: Co najmniej jeden okres różni się rozkładem ocen od pozostałych.")
print(f"Statystyka Chi^2 Friedmana = {friedman_stat:.4f}")
interpret_p_value(p_val_friedman)

if p_val_friedman < 0.05:
    print("\nTest post-hoc Nemenyi (dla Friedmana):")
    df_grades_long = pd.melt(df_student.reset_index(), id_vars=['index'], value_vars=['G1', 'G2', 'G3'],
                             var_name='Okres', value_name='Ocena')

    try:
        if 'Ocena' not in df_grades_long.columns or 'Okres' not in df_grades_long.columns:
            print("BŁĄD KRYTYCZNY: Brak kolumn 'Ocena' lub 'Okres' w df_grades_long.")

        rows_before_drop = len(df_grades_long)
        df_grades_long_for_wilcoxon = df_grades_long.dropna(subset=['Ocena'])
        rows_after_drop = len(df_grades_long_for_wilcoxon)
        if rows_before_drop != rows_after_drop:
            print(f"Usunięto {rows_before_drop - rows_after_drop} wierszy z NaN w 'Ocena' przed Wilcoxonem.")

        if df_grades_long_for_wilcoxon.empty:
            print("Brak danych w df_grades_long po usunięciu NaN z 'Ocena'. Nie można wykonać Wilcoxona.")
        else:
            posthoc_wilcox = sp.posthoc_wilcoxon(df_grades_long_for_wilcoxon, val_col='Ocena', group_col='Okres',
                                                 p_adjust='bonferroni')

            print("\nWynik testu post-hoc Wilcoxona z korektą Bonferroniego:")
            print(posthoc_wilcox)
            print("\n(Jeśli to zadziałało, potwierdza to problem specyficzny dla Nemenyi lub jego użycia block_col)")

    except Exception as e_wilcox:
        import traceback

        print(f"\nPost-hoc Wilcoxon (na danych długich, bez block_col) również NIE zadziałał:")
        print(f"Błąd: {e_wilcox}")

columns_to_compare = ['G1', 'G2', 'G3']
median_grades = df_student[columns_to_compare].median()

print("Medianowe wartości ocen dla poszczególnych okresów:")
print(median_grades)

print("\nSzczegółowe mediany:")
for col in columns_to_compare:
    print(f"Mediana dla {col}: {median_grades[col]:.2f}")

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_student[['G1', 'G2', 'G3']])
plt.title('Rozkład ocen w poszczególnych okresach (G1, G2, G3)')
plt.ylabel('Ocena')
plt.show()

# ===== KONIEC ANALIZY ZBIORU 2 =====

print("\n" + "=" * 80)
print("--- Zakończono Analizę ---")
print("=" * 80)
