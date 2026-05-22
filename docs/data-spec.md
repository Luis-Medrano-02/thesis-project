# Especificación de Datos: VinDr-Mammo

## Composición de la imagen 2x2
El modelo recibe una única imagen compuesta por 4 cuadrantes:

| Posición | Mama | Vista |
|----------|------|-------|
| Arriba-Izquierda | Derecha | CC (RIGHT_CC) |
| Arriba-Derecha | Izquierda | CC (LEFT_CC) |
| Abajo-Izquierda | Derecha | MLO (RIGHT_MLO) |
| Abajo-Derecha | Izquierda | MLO (LEFT_MLO) |

## Etiquetas objetivo
1. **Densidad mamaria (ACR):** A, B, C, D
2. **Categoría BI-RADS:** 1, 2, 3, 4, 5

## Formato del dataset
- **Fuente:** VinDr-Mammo (5000 estudios, split oficial training/test)
- **Anotaciones:** `breast-level_annotations.csv` (BIRADS + density por vista) + `finding_annotations.csv` (hallazgos por imagen)
- **Split:** 3828 estudios con PNG procesado → 3053 train / 775 test
- **BIRADS original:** muy desbalanceado (BIRADS 1: 1829, 2: 1082, 3: 436, 4: 368, 5: 113)
- **BIRADS usado (breast_birads):** máximo entre las 4 vistas del estudio

## Rutas en Drive
- `vindr_1000/processed_dcm/` — PNGs preprocesados por study_id
- `vindr_1000/processed_aug_2/BIRADS_{3,4,5}/` — augmentadas (flip + rotación)
- `vindr_1000/metadata/` — CSVs de anotaciones
- `vindr_1000/checkpoints/multitask_vf/run/` — checkpoints del fine-tuning

## Target del modelo (JSON)

**Run 1 (orden original):**
```json
{"density": "B", "findings": "mass found in left cranio-caudal (CC)", "birads": "3"}
```

**Run 2 en adelante (orden corregido — birads primero):**
```json
{"birads": "3", "density": "B", "findings": "mass found in left cranio-caudal (CC)"}
```
*Razón: eliminar coupling autoregresivo findings→birads que era inútil para discriminar BIRADS 3/4/5 (mismo texto de findings) y dañino para BIRADS 1 vs 2 ("Healthy Breast" compartido).*
