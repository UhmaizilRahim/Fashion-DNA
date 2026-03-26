# =========================================================
#  STYLEPREDICT — Mini IA per Moda Sostenibile
#  Modello: Random Forest Regressor
#  Librerie: numpy, pandas, matplotlib, scikit-learn
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(99)

# ---------------------------------------------------------
#  1. DATASET — 300 valutazioni utente-capo
# ---------------------------------------------------------

n = 300

casual_utente  = np.random.uniform(0, 1, n)   # stile preferito: 0=formale, 1=casual
eco_utente     = np.random.uniform(0, 1, n)   # sensibilita alla sostenibilita
budget         = np.random.uniform(30, 300, n) # budget in euro

casual_capo    = np.random.uniform(0, 1, n)
sostenibilita  = np.random.uniform(30, 100, n)
prezzo         = np.random.uniform(20, 300, n)

# Punteggio gradimento 1–10
gradimento = (
    3.0 +
    2.5 * (1 - np.abs(casual_utente - casual_capo)) +
    1.5 * eco_utente * (sostenibilita / 100) +
    1.5 * (1 - np.abs(budget - prezzo) / 300) +
    np.random.normal(0, 0.5, n)
).clip(1, 10)

df = pd.DataFrame({'casual_utente': casual_utente, 'eco_utente': eco_utente,
                   'budget': budget, 'casual_capo': casual_capo,
                   'sostenibilita': sostenibilita, 'prezzo': prezzo,
                   'gradimento': gradimento})

print("STYLEPREDICT — Dataset:", len(df), "valutazioni")
print(f"  Gradimento medio: {gradimento.mean():.1f} / 10")
print(df.head(4).round(2).to_string(index=False))

# ---------------------------------------------------------
#  2. GRAFICO EDA — Gradimento medio per fascia
# ---------------------------------------------------------

# Raggruppiamo in 4 fasce di "allineamento stile" (quanto l'utente e il capo si somigliano)
df['allineamento'] = 1 - np.abs(df['casual_utente'] - df['casual_capo'])
df['fascia'] = pd.cut(df['allineamento'], bins=4,
                      labels=['Molto diverso', 'Diverso', 'Simile', 'Identico'])

medie_fascia = df.groupby('fascia', observed=True)['gradimento'].mean()

fig, ax = plt.subplots(figsize=(7, 4))
colori_fascia = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
barre = ax.bar(medie_fascia.index, medie_fascia.values,
               color=colori_fascia, edgecolor='black', linewidth=0.8)
ax.set_title('STYLEPREDICT — Gradimento per Allineamento di Stile', fontweight='bold')
ax.set_xlabel('Quanto lo stile del capo corrisponde al tuo stile')
ax.set_ylabel('Gradimento medio (1-10)')
ax.set_ylim(0, 10)
for b, v in zip(barre, medie_fascia.values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.1,
            f'{v:.1f}', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('stylepredict_grafico.png', dpi=120, bbox_inches='tight')
plt.show()
print("Grafico salvato: stylepredict_grafico.png")

# ---------------------------------------------------------
#  3. MODELLO ML — Regressione
# ---------------------------------------------------------

X = df[['casual_utente', 'eco_utente', 'budget', 'casual_capo', 'sostenibilita', 'prezzo']]
y = df['gradimento']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modello = RandomForestRegressor(n_estimators=50, random_state=42)
modello.fit(X_train, y_train)
y_pred = modello.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"\nErrore medio (MAE): {mae:.2f} punti su 10")
print(f"R2: {r2:.2f}  ({r2*100:.0f}% varianza spiegata)")

# ---------------------------------------------------------
#  4. GRAFICO RISULTATI — Reale vs Predetto
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(y_test, y_pred, alpha=0.4, s=25, color='#8e44ad')
ax.plot([1, 10], [1, 10], 'r--', linewidth=1.5, label='Predizione perfetta')
ax.set_xlabel('Gradimento Reale'); ax.set_ylabel('Gradimento Predetto')
ax.set_title('STYLEPREDICT — Reale vs Predetto', fontweight='bold')
ax.legend()
ax.text(1.3, 9.0, f'MAE={mae:.2f}  R2={r2:.2f}', fontsize=10,
        color='darkblue', fontweight='bold')
plt.tight_layout()
plt.savefig('stylepredict_risultati.png', dpi=120, bbox_inches='tight')
plt.show()
print("Risultati salvati: stylepredict_risultati.png")

# ---------------------------------------------------------
#  5. SIMULATORE — Quanto piacera questo capo?
# ---------------------------------------------------------

print("\nSIMULATORE:")
capi_da_valutare = pd.DataFrame({
    'casual_utente': [0.2, 0.9, 0.5],
    'eco_utente':    [0.9, 0.4, 0.6],
    'budget':        [100, 60,  150],
    'casual_capo':   [0.3, 0.8, 0.5],
    'sostenibilita': [90,  50,  75],
    'prezzo':        [85,  55,  140],
})
nomi_capi = ['Giacca Lino Bio', 'Sneakers Hemp', 'Vestito Tencel']
pred = modello.predict(capi_da_valutare)
for nome, voto in zip(nomi_capi, pred):
    print(f"  {nome:20s}  →  gradimento previsto: {voto:.1f}/10")
