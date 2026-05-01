import pandas as pd
import json
import ast
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DRIVE_BASE   = Path('/media/datos/lmedrano/drive')
METADATA_DIR = DRIVE_BASE / 'metadata'
OUTPUT_DIR   = Path('/media/datos/lmedrano')

breast_df  = pd.read_csv(METADATA_DIR / 'breast-level_annotations.csv')
finding_df = pd.read_csv(METADATA_DIR / 'finding_annotations.csv')

# ── MAPPINGS ──────────────────────────────────────────────────────────────────
BIRADS_REC = {
    'BI-RADS 0': 'Se requiere evaluación adicional con imágenes complementarias.',
    'BI-RADS 1': 'Control mamográfico anual de rutina.',
    'BI-RADS 2': 'Control mamográfico anual de rutina.',
    'BI-RADS 3': 'Control mamográfico en 6 meses.',
    'BI-RADS 4': 'Se recomienda biopsia para evaluación histológica.',
    'BI-RADS 5': 'Biopsia indicada. Alta sospecha de malignidad.',
}

DENSITY_MAP = {
    'DENSITY A': 'Tipo A: mamas predominantemente grasas.',
    'DENSITY B': 'Tipo B: densidad fibroglandular dispersa.',
    'DENSITY C': 'Tipo C: densidad heterogénea, puede ocultar pequeñas lesiones.',
    'DENSITY D': 'Tipo D: extremadamente densa, reduce la sensibilidad del estudio.',
}

FINDING_ES = {
    'No Finding'                   : 'sin hallazgos significativos',
    'Mass'                         : 'masa',
    'Suspicious Calcification'     : 'calcificaciones sospechosas',
    'Focal Asymmetry'              : 'asimetría focal',
    'Architectural Distortion'     : 'distorsión arquitectural',
    'Asymmetry'                    : 'asimetría',
    'Suspicious Lymph Node'        : 'ganglio linfático sospechoso',
    'Skin Thickening'              : 'engrosamiento cutáneo',
    'Global Asymmetry'             : 'asimetría global',
    'Nipple Retraction'            : 'retracción del pezón',
    'Skin Retraction'              : 'retracción cutánea',
}

# ── FUNCIONES ─────────────────────────────────────────────────────────────────
def parse_findings(row):
    try:
        cats = ast.literal_eval(row['finding_categories'])
    except:
        cats = ['No Finding']
    return [FINDING_ES.get(c, c.lower()) for c in cats]

def build_report(study_id):
    # Datos breast-level (tomar primera fila por estudio para birads y density)
    breast_rows = breast_df[breast_df['study_id'] == study_id]
    if breast_rows.empty:
        return None

    birads  = breast_rows['breast_birads'].iloc[0]
    density = breast_rows['breast_density'].iloc[0]

    # Hallazgos por lateralidad
    find_rows = finding_df[finding_df['study_id'] == study_id]

    def get_findings_for(lat):
        rows = find_rows[find_rows['laterality'] == lat]
        if rows.empty:
            return 'sin hallazgos significativos'
        all_findings = []
        for _, r in rows.iterrows():
            all_findings.extend(parse_findings(r))
        unique = list(dict.fromkeys(all_findings))
        if unique == ['sin hallazgos significativos']:
            return 'sin hallazgos significativos'
        findings = [f for f in unique if f != 'sin hallazgos significativos']
        return ', '.join(findings) if findings else 'sin hallazgos significativos'

    left_findings  = get_findings_for('L')
    right_findings = get_findings_for('R')

    density_text = DENSITY_MAP.get(density, density)
    rec_text     = BIRADS_REC.get(birads, 'Evaluación clínica recomendada.')
    birads_num   = birads.replace('BI-RADS ', '')

    report = (
        f"Estudio mamográfico bilateral.\n"
        f"Densidad mamaria: {density_text}\n"
        f"Mama izquierda: {left_findings}.\n"
        f"Mama derecha: {right_findings}.\n"
        f"Categoría BI-RADS: {birads_num}.\n"
        f"Recomendación: {rec_text}"
    )
    return report

# ── GENERAR DATASET ───────────────────────────────────────────────────────────
study_ids = breast_df['study_id'].unique().tolist()
print(f'Total estudios: {len(study_ids)}')

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
print(f'Generados: {len(df_out)} | Fallidos: {fail}')
print(f'\nDistribución BI-RADS:')
print(df_out['breast_birads'].value_counts())
print(f'\nDistribución split:')
print(df_out['split'].value_counts())

# Guardar
out_path = OUTPUT_DIR / 'vindr_reports.csv'
df_out.to_csv(out_path, index=False)
print(f'\n✅ Guardado en {out_path}')

# Muestra de 2 reportes
print('\n─── Ejemplo de reporte ───')
print(df_out['report'].iloc[0])
print('\n─── Ejemplo de reporte ───')
print(df_out['report'].iloc[100])
