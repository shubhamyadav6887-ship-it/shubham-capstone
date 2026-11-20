"""VolleyVision AI - Attack performance analysis (VNL 2024, Men)
Capstone project, AML002 - Introduction to Data Science
Author: Shubham Yadav (2601840018), Section A

Usage: put VNL2024Men_Attackers.csv in the same folder and run:  python volleyvision_analysis.py
Requires: pandas numpy matplotlib seaborn scipy scikit-learn
"""


# # VolleyVision AI – Attack Performance Analysis (VNL 2024, Men)
# Capstone notebook for AML002. Upload `VNL2024Men_Attackers.csv` to Colab (or keep it next to this notebook) and run all cells.


# ## Phase 1–2: Load and understand the data

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
sns.set_theme(style='whitegrid')

# the file is not UTF-8 (accented player names), so latin-1 is needed
raw = pd.read_csv('VNL2024Men_Attackers.csv', encoding='latin-1')
raw.columns = ['name','team','points','errors','other','pts_per_match','kill_pct_src','total']
print(raw.shape); print(raw.head()); print(raw.dtypes)

print('Missing values:\n', raw.isna().sum())
print('Duplicate rows:', raw.duplicated().sum())
print('Rows where points+errors+other != total:', ((raw.points+raw.errors+raw.other)!=raw.total).sum())
print('Players with 0 attacks:', (raw.total==0).sum(), '| Teams:', raw.team.nunique())


# ## Phase 3: Prepare and explore
# Decisions: remove players with 0 attacks (rates impossible; the missing values come from them) and keep players with at least 30 attacks for the main analysis.

df = raw[raw.total > 0].copy()
df['kill_pct']   = df.points / df.total * 100
df['error_pct']  = df.errors / df.total * 100
df['efficiency'] = (df.points - df.errors) / df.total * 100

MIN = 30
q = df[df.total >= MIN].copy()
print('With >=1 attack:', len(df), '| Qualified (>=%d attacks):' % MIN, len(q))
print(q[['total','points','kill_pct','error_pct','efficiency']].describe().round(2))

fig, ax = plt.subplots(1, 2, figsize=(9,3.6))
sns.histplot(df.total, bins=25, ax=ax[0]); ax[0].axvline(MIN, color='crimson', ls='--'); ax[0].set_title('Total attacks per player')
sns.histplot(q.kill_pct, bins=20, ax=ax[1]); ax[1].set_title('Attack success % (qualified)')
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(11,4))
sns.scatterplot(data=q, x='total', y='kill_pct', ax=ax[0]); ax[0].set_title('Volume vs success %')
sns.regplot(data=q, x='error_pct', y='kill_pct', ax=ax[1]); ax[1].set_title('Error % vs success %')
plt.tight_layout(); plt.show()


# ## Phase 4: Analysis

for a, b in [('error_pct','kill_pct'), ('total','kill_pct'), ('total','error_pct')]:
    r, p = stats.spearmanr(q[a], q[b]); print(f'Spearman {a} vs {b}: rho={r:.3f}, p={p:.4g}')

med = q.total.median()
hi, lo = q[q.total >= med], q[q.total < med]
t = stats.ttest_ind(hi.kill_pct, lo.kill_pct, equal_var=False)
u = stats.mannwhitneyu(hi.kill_pct, lo.kill_pct)
d = (hi.kill_pct.mean()-lo.kill_pct.mean()) / np.sqrt((hi.kill_pct.var()+lo.kill_pct.var())/2)
print(f'High vs low volume: diff={hi.kill_pct.mean()-lo.kill_pct.mean():.2f}, Welch p={t.pvalue:.3f}, MWU p={u.pvalue:.3f}, d={d:.2f}')

kw = stats.kruskal(*[g.kill_pct.values for _, g in q.groupby('team')])
print(f'Kruskal-Wallis across teams: H={kw.statistic:.2f}, p={kw.pvalue:.3f}')
print(q.groupby('team').agg(players=('name','count'), mean_success=('kill_pct','mean'), mean_error=('error_pct','mean')).round(2).sort_values('mean_success', ascending=False))

X = q[['error_pct']].assign(log_total=np.log(q.total)); y = q.kill_pct
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42)
m = LinearRegression().fit(Xtr, ytr); pred = m.predict(Xte)
print('Coefficients:', dict(zip(X.columns, m.coef_.round(3))), '| intercept', round(m.intercept_,2))
print('Test R2 = %.3f, MAE = %.2f' % (r2_score(yte, pred), mean_absolute_error(yte, pred)))

F = StandardScaler().fit_transform(np.c_[q.kill_pct, q.error_pct, np.log(q.total)])
for k in [2,3,4]:
    print(k, round(silhouette_score(F, KMeans(k, n_init=10, random_state=42).fit_predict(F)), 3))
q['cluster'] = KMeans(3, n_init=10, random_state=42).fit_predict(F) + 1
print(q.groupby('cluster').agg(players=('name','count'), attacks=('total','mean'), success=('kill_pct','mean'),
                                 error=('error_pct','mean'), efficiency=('efficiency','mean')).round(1))


# ### VolleyVision Performance Index
# Index = 50 + 10 × (0.5·z(efficiency) + 0.3·z(points per match) + 0.2·z(log attacks)). The weights are a design choice.

z = lambda s: (s - s.mean()) / s.std()
q['index'] = 50 + 10*(0.5*z(q.efficiency) + 0.3*z(q.pts_per_match) + 0.2*z(np.log(q.total)))
top10 = q.sort_values('index', ascending=False).head(10)[['name','team','total','kill_pct','efficiency','pts_per_match','index']].round(1)
print(top10)

def analyse_player(name):
    # mini 'VolleyVision' lookup for one player
    r = q[q.name.str.lower() == name.lower()]
    return r[['name','team','total','kill_pct','error_pct','efficiency','cluster','index']].round(1) if len(r) else 'Not in qualified group'
print(analyse_player('Leon'))


# ### Robustness: other minimum-attack thresholds

for th in [20, 30, 50]:
    s = df[df.total >= th]
    r1,_ = stats.spearmanr(s.error_pct, s.kill_pct); r2,p2 = stats.spearmanr(s.total, s.kill_pct)
    print(f'>= {th}: n={len(s)}, rho(error,success)={r1:.2f}, rho(volume,success)={r2:.2f} (p={p2:.3f})')


# ## Phase 5: Communicate
# See the Word report for findings, limitations and conclusions. Key point: error rate is the strongest sign of attack success; attack volume is only weakly (negatively) related; teams do not differ significantly.
