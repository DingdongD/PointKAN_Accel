
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
json_path = base / 'baseline_pointkan_elite_breakdown_by_points_fast.json'
if not json_path.exists():
    raise SystemExit('missing json')
rows = json.loads(json_path.read_text())
df = pd.DataFrame(rows).sort_values('input_points')
lat_cols = [c for c in df.columns if c.endswith('__latency_ns')]
eng_cols = [c for c in df.columns if c.endswith('__energy_nj')]
lat = pd.DataFrame({'input_points': df['input_points']})
eng = pd.DataFrame({'input_points': df['input_points']})
for c in lat_cols:
    lat[c[:-12]] = df[c] / 1e6
for c in eng_cols:
    eng[c[:-11]] = df[c]
lat = lat.set_index('input_points')
eng = eng.set_index('input_points')
lat_pct = lat.div(lat.sum(axis=1), axis=0) * 100.0
eng_pct = eng.div(eng.sum(axis=1), axis=0) * 100.0
for name, table, ylabel in [
    ('latency_ms', lat, 'Latency (ms)'),
    ('latency_pct', lat_pct, 'Latency Share (%)'),
    ('energy_nj', eng, 'Energy (nJ)'),
    ('energy_pct', eng_pct, 'Energy Share (%)'),
]:
    ax = table.plot(kind='bar', stacked=True, figsize=(12,6), colormap='tab20')
    ax.set_xlabel('Input points')
    ax.set_ylabel(ylabel)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(base / f'baseline_pointkan_elite_breakdown_{name}.png', dpi=200)
    plt.close()
lat.to_csv(base / 'baseline_pointkan_elite_breakdown_latency_ms.csv')
lat_pct.to_csv(base / 'baseline_pointkan_elite_breakdown_latency_pct.csv')
eng.to_csv(base / 'baseline_pointkan_elite_breakdown_energy_nj.csv')
eng_pct.to_csv(base / 'baseline_pointkan_elite_breakdown_energy_pct.csv')
print(base)
