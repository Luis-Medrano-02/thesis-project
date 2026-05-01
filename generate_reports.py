import pandas as pd
import ast
import json
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DRIVE_BASE   = Path('/media/datos/lmedrano/drive')
METADATA_DIR = DRIVE_BASE / 'metadata'
OUTPUT_DIR   = Path('/media/datos/lmedrano')

breast_df  = pd.read_csv(METADATA_DIR / 'breast-level_annotations.csv')
finding_df = pd.read_csv(METADATA_DIR / 'finding_annotations.csv')

# ── MAPPINGS ──────────────────────────────────────────────────────────────────
DENSITY_MAP = {
    'DENSITY A': 'ACR A: almost entirely fatty',
    'DENSITY B': 'ACR B: scattered fibroglandular densities',
    'DENSITY C': 'ACR C: heterogeneously dense',
    'DENSITY D': 'ACR D: extremely dense',
}

# FIX 1: BI-RADS 4 y 5 ahora tienen etiquetas distintas.
# Antes ambos eran 'suspicious', lo que impedía al modelo diferenciarlos.
BIRADS_SUSPICION = {
    'BI-RADS 0': 'incomplete — additional imaging needed',
    'BI-RADS 1': 'negative — no abnormality',
    'BI-RADS 2': 'benign — no malignancy',
    'BI-RADS 3': 'probably benign — short-interval follow-up',
    'BI-RADS 4': 'suspicious — tissue sampling advised',
    'BI-RADS 5': 'highly suggestive of malignancy — biopsy recommended',
}

FINDING_EN = {
    'No Finding'               : 'no significant findings',
    'Mass'                     : 'mass',
    'Suspicious Calcification' : 'suspicious calcifications',
    'Focal Asymmetry'          : 'focal asymmetry',
    'Architectural Distortion' : 'architectural distortion',
    'Asymmetry'                : 'asymmetry',
    'Suspicious Lymph Node'    : 'suspicious lymph node',
    'Skin Thickening'          : 'skin thickening',
    'Global Asymmetry'         : 'global asymmetry',
    'Nipple Retraction'        : 'nipple retraction',
    'Skin Retraction'          : 'skin retraction',
}

# ── FUNCIONES ─────────────────────────────────────────────────────────────────
def parse_findings(row):
    try:
        cats = ast.literal_eval(row['finding_categories'])
    except:
        cats = ['No Finding']
    return [FINDING_EN.get(c, c.lower()) for c in cats]

def get_findings_for(find_rows, lat):
    rows = find_rows[find_rows['laterality'] == lat]
    if rows.empty:
        return 'no significant findings'
    all_findings = []
    for _, r in rows.iterrows():
        all_findings.extend(parse_findings(r))
    unique = list(dict.fromkeys(all_findings))
    findings = [f for f in unique if f != 'no significant findings']
    return ', '.join(findings) if findings else 'no significant findings'

def build_report(study_id):
    breast_rows = breast_df[breast_df['study_id'] == study_id]
    if breast_rows.empty:
        return None

    birads    = breast_rows['breast_birads'].iloc[0]
    density   = breast_rows['breast_density'].iloc[0]
    find_rows = finding_df[finding_df['study_id'] == study_id]

    left_findings  = get_findings_for(find_rows, 'L')
    right_findings = get_findings_for(find_rows, 'R')

    # Findings summary
    all_findings = set()
    for lat in ['L', 'R']:
        rows = find_rows[find_rows['laterality'] == lat]
        for _, r in rows.iterrows():
            all_findings.update(parse_findings(r))
    all_findings.discard('no significant findings')
    findings_summary = ', '.join(all_findings) if all_findings else 'no significant findings'

    # FIX 2: birads_num como campo separado (solo el dígito),
    # más fácil de parsear durante evaluación sin regex sobre strings compuestos.
    birads_num  = birads.replace('BI-RADS ', '')
    suspicion   = BIRADS_SUSPICION.get(birads, 'benign')
    density_text = DENSITY_MAP.get(density, density)

    report = json.dumps({
        'breast_density'   : density_text,
        'findings'         : findings_summary,
        'left_findings'    : left_findings,
        'right_findings'   : right_findings,
        # FIX 2: campo numérico limpio para parseo directo en evaluación
        'birads_category'  : birads_num,
        # campo legible para el modelo durante generación de texto
        'birads_assessment': f'BI-RADS {birads_num}: {suspicion}',
        'suspicion'        : suspicion,
    }, ensure_ascii=False)

    return report

# ── GENERAR DATASET ───────────────────────────────────────────────────────────
study_ids = breast_df['study_id'].unique().tolist()
print(f'Total studies: {len(study_ids)}')

records = []
fail = 0

for study_id in study_ids:
    breast_rows = breast_df[breast_df['study_id'] == study_id]
    birads      = breast_rows['breast_birads'].iloc[0]
    density     = breast_rows['breast_density'].iloc[0]
    split       = breast_rows['split'].iloc[0]
    report      = build_report(study_id)

    if report is None:
        fail += 1
        continue

    png_path = DRIVE_BASE / 'processed' / f'{study_id}.png'

    records.append({
        'study_id'      : study_id,
        'image_path'    : str(png_path),
        'breast_birads' : birads,
        'breast_density': density,
        'split'         : split,
        'report'        : report,
    })

df_out = pd.DataFrame(records)
print(f'Generated: {len(df_out)} | Failed: {fail}')
print(f'\nBI-RADS distribution:')
print(df_out['breast_birads'].value_counts())

# ── REBALANCEO ────────────────────────────────────────────────────────────────
print('\nRebalancing training set...')
TARGET = {
    'BI-RADS 1': 500,
    'BI-RADS 2': 500,
    'BI-RADS 3': 500,
    'BI-RADS 4': 500,
    'BI-RADS 5': 200,
}

train_df = df_out[df_out['split'] == 'training']
test_df  = df_out[df_out['split'] == 'test']

balanced_parts = []
for birads, target in TARGET.items():
    subset = train_df[train_df['breast_birads'] == birads]
    if len(subset) >= target:
        balanced_parts.append(subset.sample(n=target, random_state=42))
    else:
        # oversampling con reemplazo para clases minoritarias
        balanced_parts.append(subset.sample(n=target, replace=True, random_state=42))

train_balanced = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)
df_final = pd.concat([train_balanced, test_df]).reset_index(drop=True)

print(f'\nPost-rebalancing BI-RADS (train):')
print(train_balanced['breast_birads'].value_counts())
print(f'\nTrain: {len(train_balanced)} | Test: {len(test_df)} | Total: {len(df_final)}')

# ── VERIFICACIÓN DE SANIDAD ───────────────────────────────────────────────────
print('\n── Sanity check: etiquetas únicas de suspicion ──')
suspicion_check = {}
for _, row in df_final.drop_duplicates('breast_birads').iterrows():
    report_dict = json.loads(row['report'])
    suspicion_check[row['breast_birads']] = report_dict['suspicion']
for k, v in sorted(suspicion_check.items()):
    print(f'  {k} → {v}')

# ── GUARDAR (sobreescribe el CSV anterior) ────────────────────────────────────
out_path = OUTPUT_DIR / 'vindr_reports.csv'
df_final.to_csv(out_path, index=False)
print(f'\n✅ Saved to {out_path}  (sobreescribió versión anterior)')

print('\n─── Ejemplo BI-RADS 4 ───')
ex4 = df_final[df_final['breast_birads'] == 'BI-RADS 4']['report'].iloc[0]
print(json.dumps(json.loads(ex4), indent=2, ensure_ascii=False))

print('\n─── Ejemplo BI-RADS 5 ───')
ex5 = df_final[df_final['breast_birads'] == 'BI-RADS 5']['report'].iloc[0]
print(json.dumps(json.loads(ex5), indent=2, ensure_ascii=False))
