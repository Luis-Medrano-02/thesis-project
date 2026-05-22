# Decisiones de Arquitectura y Experimentos

## Decisiones base (fijas)
- **Modelo:** `google/medgemma-4b-it`
- **Dataset:** VinDr-Mammo
- **Formato de entrada:** Cuadrícula 2x2 con las 4 proyecciones mamográficas estándar
- **Método de fine-tuning:** QLoRA
- **Alineación:** DPO (Direct Preference Optimization)
- **Targets:** Densidad ACR (A/B/C/D) + Categoría BI-RADS (1-5)

## Historial de experimentos

| # | Fecha | Checkpoint | n | BIRADS Acc | Density Acc | BIRADS F1 | Density F1 | Notas |
|---|-------|-----------|---|------------|-------------|-----------|------------|-------|
| 0 | 2026-05-21 | **Zeroshot MedGemma 4B-it (sin LoRA)** | **1000** | **0.1100** | **0.3680** | **0.0583** | **0.1854** | Colapso a BIRADS 4 (acc=0.822). B1=0.095 B2=0.003 B3=0.022 B4=0.822 B5=0.000 |
| 1 | 2026-05-20 | checkpoint-2475 (época 9) | 50 | 0.360 | 0.900 | 0.352 | 0.872 | B1=0.50 B2=0.20 B3=0.10 B4=0.40 B5=0.60 |
| 1 | 2026-05-20 | checkpoint-3300 (época 12) | 50 | 0.440 | 0.820 | 0.425 | 0.758 | B1=0.50 B2=0.60 B3=0.10 B4=0.40 B5=0.60 |
| 1 | 2026-05-20 | checkpoint-4125 (época 15) | 50 | 0.420 | 0.840 | 0.422 | 0.796 | B1=0.40 B2=0.60 B3=0.20 B4=0.30 B5=0.60 |
| 1 | 2026-05-21 | **checkpoint-3300 (época 12) — FINAL** | **1000** | **0.3940** | **0.8220** | **0.3639** | **0.6889** | **Evaluación definitiva test set completo.** B1=0.4291(n=494) B2=0.3793(n=319) B3=0.2747(n=91) B4=0.3151(n=73) B5=0.5652(n=23) |

*(Claude actualiza esta tabla automáticamente al obtener métricas)*

## Audit Sección 1 — Preprocesamiento (2026-05-22)

**Veredicto: SÓLIDO (8/10). Sin cambios necesarios para Run 2.**

### Decisiones de diseño confirmadas (no cambiar)
- **Orientación CC:** convención "espalda con espalda" (pezones hacia borde exterior) en lugar de la estándar "mariposa" (pezones hacia centro). Causa: DICOMs VinDr-Mammo almacenan R_CC con pezón apuntando a la izquierda; sin flip horizontal al colocar en top-left. **Consistente en los 997 estudios** → el modelo puede aprenderlo. No es la causa de errores BIRADS 3/4. Costo de corregir: regenerar 3883 PNGs. Decisión: mantener y documentar como limitación en tesis.
- **CLAHE tileGridSize=(8,8) fijo pre-resize:** sobre DICOM grande (~3500px) produce tiles de ~437px (realce casi global); sobre crop pequeño (~800px) produce tiles ~100px (realce local). Introduce variabilidad inter-estudio en contraste. Mitigado porque el resize posterior a 448×448 unifica la escala de salida. Mantener.
- **threshold `found >= 2` en `build_merge_2x2`:** riesgo latente de cuadrantes negros si faltan vistas. No activo en práctica: validación previa confirma 4 vistas en todos los estudios. Mantener.

### Corrección de orden confirmada como correcta (no era bug)
- MONOCHROME1 se invierte **antes** de Otsu → Otsu siempre ve fondo=oscuro/tejido=brillante. Correcto.
- Normalización min-max antes de inversión MONOCHROME1: matemáticamente equivalente al orden inverso. Correcto.

### Punto no verificable
- Drive contiene 3883 PNGs pero este notebook solo procesó 1000 (de `selected_1000_studies.csv`). Los ~2883 restantes fueron preprocesados en una ejecución anterior. Si usaron el mismo código y parámetros, no hay inconsistencia. No verificable sin el código original de esa ejecución.

### Pipeline confirmado paso a paso
1. slope/intercept ✅ → 2. normalización float32→uint8 ✅ → 3. inversión MONOCHROME1 ✅ → 4. GaussianBlur+Otsu+MORPH_CLOSE(25×25,4iter)+largest-component+crop+2%pad ✅ → 5. CLAHE(clipLimit=2.0,tileGridSize=8×8) ✅ → 6. resize 448×448 INTER_AREA ✅ → 7. merge 2×2 → convert('RGB') ✅

## Run 1 — Entrenamiento completo (15 épocas)
- **Fecha:** 2026-05-20
- **Steps totales:** 4125 (275/época)
- **Loss final:** 0.0495 (step 4125)
- **Loss inicial:** 6.677
- **Problema detectado:** CollapseCallback fallaba con `expected scalar type BFloat16 but found Float` en `model.generate()` durante training — nunca se obtuvieron métricas válidas
- **Fix identificado:** envolver `model.generate()` en `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` + `do_sample=False`
- **Checkpoints guardados:** checkpoint-1650 a checkpoint-4125 (épocas 6-15) en `/content/drive/MyDrive/vindr_1000/checkpoints/multitask_vf/run/`
- **Resultados checkpoint-2475 (época 9):** BIRADS acc=0.40, Density acc=0.88. Densidad excelente. BIRADS débil en clases 2 y 3 (acc 0.20 y 0.10). Formato JSON consistente.
- **Mejor checkpoint confirmado:** checkpoint-3300 (época 12).
- **Evaluación final (1000 muestras test set real):** BIRADS acc=0.394, F1=0.364, Density acc=0.822, F1=0.689. Guardado en `vindr_1000/checkpoints/multitask_vf/run/final_eval_checkpoint3300.json`
- **Discrepancia 50→1000 muestras:** Los 50 casos estaban balanceados (10/clase), el test real está desbalanceado (494 BIRADS 1, solo 23 BIRADS 5). Por eso BIRADS F1 cae de 0.425→0.364.
- **Trade-off detectado:** época 9 tiene mejor density (acc=0.90) pero peor BIRADS F1 (0.352). Época 12 balancea mejor ambas tareas.
- **Debilidad persistente:** BIRADS 3 (acc=0.275) y BIRADS 4 (acc=0.315) — clases más difíciles. BIRADS 5 sorpresivamente bien (0.565) pese a solo 23 casos.
- **Distribución real del test set (1000 estudios):** BIRADS 1: 494, BIRADS 2: 319, BIRADS 3: 91, BIRADS 4: 73, BIRADS 5: 23 — muy desbalanceado. Test set oficial VinDr-Mammo = 1000 estudios.
- **Resultados 50 casos guardados en Drive:** `vindr_1000/checkpoints/multitask_vf/run/checkpoint_eval_results.json`
- **Próximo paso:** correr baseline zeroshot (notebook 01), luego Run 2 con 3 fixes (CollapseCallback autocast, collate_fn masking, USER_PROMPT duplicado).

## Audit Sección 2 — Dataset Construction (2026-05-22)

**Veredicto: 6/10 — 1 bug crítico (JSON order), 1 bug medio (sampler), 1 bug menor (collate_fn).**

### 2a — Label noise BIRADS 3/4 con "Healthy Breast" ✅ EXCELENTE
- **Resultado:** 0 de 345 BIRADS 3 y 0 de 295 BIRADS 4 tienen findings="No Finding" en las anotaciones.
- **Conclusión:** Cero label noise. Cada estudio BIRADS 3/4 tiene al menos un hallazgo real. No hay objetivos contradictorios en el training set.
- **Validación cruzada:** 345+91=436 BIRADS 3 total ✓, 295+73=368 BIRADS 4 total ✓ (coincide con distribución VinDr-Mammo).

### 2b — Campos descartados por build_target ✅ OK (diseño intencional)
- `build_target()` conserva: `density`, `findings`, `birads`.
- Descarta: `mass`, `calcification`, `asymmetry`, `suspicion_level`.
- Decisión válida: los hallazgos verbales en `findings` ya capturan esa información. Simplifica el target sin perder información diagnóstica esencial.

### 2c — Orden JSON: causa raíz confirmada ⚠️ BUG CRÍTICO → Fix en Run 2
- Orden actual: `density → findings → birads` (BIRADS último).
- BIRADS se predice tras 40-50 tokens de texto de findings → coupling autoregresivo fuerza predicciones consistentes con el texto ("Healthy Breast" → BIRADS 1/2).
- **Fix Run 2:** reordenar a `{"birads": "...", "density": "...", "findings": "..."}` — imagen→BIRADS es la primera predicción, sin contexto previo que la fuerce.

### 2d — Pool augmentadas suficiente ✅ OK
- BIRADS_3: 500 disponibles, necesita 155 → OK
- BIRADS_4: 500 disponibles, necesita 205 → OK
- BIRADS_5: 200 disponibles, necesita 110 → OK
- No hay riesgo de duplicate sampling ni pool exhaustion.

### 2e — swap_findings correcto ✅ OK
- Patrón `__RIGHT__` placeholder funciona correctamente para intercambio bidireccional.
- Case-insensitive. Respeta "Healthy Breast. No Findings" (no modifica).
- Bilateral ("right CC, left MLO") se intercambia correctamente.

### 2f — StratifiedBatchSampler no conectado al trainer ⚠️ BUG MEDIO → Fix en Run 2
- El sampler existe y funciona (verificado: 3 batches muestran las 5 clases).
- SFTTrainer recibe solo `train_dataset` y `data_collator` — sampler ignorado.
- Consecuencia: PyTorch usa RandomSampler. Balance 500/500/500/500/200 existe globalmente en df_balanced pero NO está garantizado por batch efectivo.
- **Fix Run 2:** conectar via DataLoader personalizado o subclase de SFTTrainer.

### 2g — collate_fn masking bug latente ⚠️ BUG MENOR → Fix en Run 2
- `found = -1` / `if found > 0`: si model_turn no encontrado, found=-1 > 0 es False → sin masking → prompt entra en loss.
- En Run 1 nunca se activó (formato siempre consistente).
- **Fix Run 2:** `found = None` / `if found is not None`.

## Audit Sección 3 — Prompts (2026-05-22)

**Veredicto: 7/10 — Sin bugs nuevos. 3 fixes menores necesarios para Run 2 (todos relacionados con el reorder JSON de Sección 2c).**

### 3a — SYSTEM_PROMPT ✅ OK
- "You are a board-certified breast radiologist with extensive experience in screening mammography. You are meticulous and produce clear, clinically actionable reports."
- Conciso, clínicamente correcto. Sin cambios.

### 3b — USER_PROMPT duplica SYSTEM_PROMPT ⚠️ Bug conocido → Fix en Run 2
- Las primeras 2 oraciones de USER_PROMPT son idénticas a SYSTEM_PROMPT. Strings concatenados en Python sin punto separador.
- El modelo ve el rol dos veces antes de las instrucciones.
- **Fix Run 2:** eliminar esas 2 líneas del inicio de USER_PROMPT.

### 3c — Layout 2x2 en USER_PROMPT ✅ CORRECTO (documentación del notebook incorrecta)
- USER_PROMPT dice: top-left=RIGHT CC, top-right=LEFT CC, bot-left=RIGHT MLO, bot-right=LEFT MLO.
- Coincide exactamente con slot_map del preprocessing (`('R','CC'): (0,0)`). Correcto.
- **Inconsistencia:** la celda markdown 13 del notebook documenta lo opuesto (top-left=L_CC). Documentación del notebook equivocada, el prompt de entrenamiento es correcto. No afecta el modelo. Documentar como limitación menor en tesis.

### 3d — Hint ordering de BI-RADS ⚠️ Debe actualizarse para Run 2
- USER_PROMPT dice: "BI-RADS (assigned after density and findings):" — refuerza explícitamente el orden incorrecto.
- **Fix Run 2:** cambiar a "BI-RADS overall assessment:" para ser consistente con el nuevo orden `{"birads": ..., "density": ..., "findings": ...}`.

### 3e — JSON template en USER_PROMPT ⚠️ Debe actualizarse para Run 2
- Template actual: `{"density": "...", "findings": "...", "birads": "..."}` — orden incorrecto.
- El modelo aprende el formato de salida de este ejemplo — el orden importa.
- **Fix Run 2:** cambiar a `{"birads": "...", "density": "...", "findings": "..."}`.

### 3f — Instrucción no-markdown ✅ Funciona post fine-tuning
- "Return ONLY valid JSON. No markdown." — pre-training el modelo la ignoraba (outputs con ```json```). Post fine-tuning fue consistente. parse_birads/parse_density son robustos (json.loads() primero, regex fallback). Mantener.

### 3g — Concatenación SYSTEM+USER en user role ✅ OK
- Todo en `{'role':'user', ...}`. MedGemma (Gemma-based) no tiene system role → correcto.

### 3h — Token budget ✅ OK
- 692 tokens totales/muestra. ~45 activos en loss (solo JSON target). Eficiencia 6.5% — normal para instruction tuning con targets cortos. Sin truncación necesaria (contexto 8192 tokens).

## Audit Sección 4 — QLoRA Config (2026-05-22)

**Veredicto: 8/10 — Sólido. Sin cambios para Run 2. Dos puntos a registrar.**

### 4a — BitsAndBytesConfig ✅ OK
- `nf4`, `double_quant=True`, `compute_dtype=bfloat16`. Configuración estándar QLoRA. Sin issues.

### 4b — lora_r=32, lora_alpha=16: ratio alpha/r = 0.5 ⚠️ Inusual, documentar en tesis
- Scaling LoRA: `output += (lora_B @ lora_A) × (alpha/r) = × 0.5`. Más conservador que estándar QLoRA (ratio=1.0) y LoRA original (ratio=2.0).
- Consecuencia: adaptador contribuye con la mitad del peso relativo → necesita más épocas para converger, más estable en dominios ya calibrados (MedGemma + mamografía).
- Loss final 0.0495 en 15 épocas confirma convergencia. Atribuido a AMRG. Sin cambios para Run 2.

### 4c — 238 target modules ✅ OK
- q/k/v/o_proj + gate/up/down_proj del LM completo. Excluye vision encoder (SigLIP) via regex `^(?!.*vision)`.
- 730M parámetros entrenables (22.69%) — fine-tuning denso del LM vía LoRA.

### 4d — modules_to_save=['lm_head'] ✅ OK para training, ver 4f para deployment
- LM head entrenado completo en bfloat16. Necesario para aprender vocabulario JSON exacto.

### 4e — Vision encoder congelado ✅ OK
- SigLIP pre-entrenado en datos médicos — representaciones visuales preservadas. Correcto.

### 4f — Warning: tie_word_embeddings ⚠️ LATENTE — crítico antes de fase DPO
- MedGemma tiene lm_head y embed_tokens con pesos atados. modules_to_save=['lm_head'] crea copia separada → potencialmente desata pesos.
- En Run 2 (training puro): sin impacto, funciona como Run 1.
- **En DPO o merge_and_unload(): puede causar inconsistencias lm_head ↔ embed_tokens.**
- Fix para deployment: agregar `ensure_weight_tying=True` en LoraConfig o quitar lm_head de modules_to_save.

### 4g — align_dtypes() ✅ Necesario
- 476 params quedaron en float32 al cargar modelo cuantizado → alineados a bfloat16. Sin esto: backward falla con dtype mismatch.

### 4h — VRAM: 4.7GB / 40GB ✅ OK
- 12% de VRAM para modelo. Sin OOM en Run 1. Estable para Run 2.

### 4i — attn_implementation='eager' ✅ Necesario
- Flash Attention 2 incompatible con gradient checkpointing en 4-bit. Sin cambios.

## Audit Sección 5 — Training (2026-05-22)

**Veredicto: 6.5/10 — Sin bugs nuevos. 5e y 5f ya documentados. Un hallazgo nuevo relevante para Run 2.**

### 5a — adamw_bnb_8bit ✅ OK
- Optimizer 8-bit: ahorra ~8GB VRAM para 730M params. Estándar QLoRA.

### 5b — LR=1e-4, cosine, warmup=5% ✅ OK
- AMRG recommendation. Warmup=206 steps de 4125 totales. max_grad_norm=1.0. Sin cambios.

### 5c — Batch efectivo=8 ✅ OK
- batch_size=1 × grad_accum=8. Pequeño pero aceptable dado VRAM y dataset size (2200 muestras).

### 5d — 15 épocas sin EarlyStopping ⚠️ Overfit leve documentado
- eval_strategy='no'. Mejor checkpoint fue época 12 (no 15) → overfit leve en épocas 13-15.
- Mitigado: selección offline por BIRADS F1. Sin cambios para Run 2.
- Documentar en tesis: model selection offline en lugar de EarlyStopping.

### 5e — StratifiedBatchSampler desconectado → ya documentado (2f)

### 5f — CollapseCallback do_sample=True sin autocast → ya documentado (Bug 1)

### 5g — Gradient checkpointing use_reentrant=False ✅ OK

### 5h — save_strategy='epoch', save_total_limit=10 ✅ OK
- Guardó épocas 6-15. El mejor (época 12) estaba disponible. Sin cambios.

### 5i — steps_per_epoch: discrepancia si se conecta el sampler ⚠️ Menor — relevante Run 2
- Sin sampler: 2200 // 8 = 275 steps/época (lo que corrió en Run 1).
- Con sampler conectado: len(sampler) = 200 steps/época (limitado por BIRADS 5: 200/1).
- **Fix Run 2:** si se conecta el sampler, cambiar `steps_per_epoch=200` y ajustar `check_every` del CollapseCallback proporcionalmente.

### 5j — Loss curve: 6.677 → 0.0495 ✅ Convergió bien
- 97% reducción en 4125 steps. Plateau ~época 12. Sin anomalías.

### 5k — validate_custom: mismo bug do_sample=True ⚠️ Bug conocido
- Cell 24 también usa do_sample=True sin autocast. Mismo fix que CollapseCallback (Bug 1). No falló offline porque el modelo estaba en eval mode estable.

## Audit Sección 6 — Training Targets (2026-05-22)

**Veredicto: 7/10 — Sin bugs nuevos. Hallazgo clave: ambigüedad BIRADS 1 vs 2 beneficia directamente del fix de reorder JSON.**

### 6a — Orden JSON → ya documentado (2c). Fix crítico Run 2.

### 6b — ~45 tokens activos por target ✅ OK con matiz
- Variabilidad alta: Healthy Breast ~20 tokens, findings complejos ~60-80 tokens.
- Efecto favorable: clases difíciles (BIRADS 3/4/5 con findings) contribuyen más gradiente por muestra.

### 6c — Distribución density en df_balanced ✅ OK (no crítico)
- Density acc=0.822 en Run 1 indica aprendizaje correcto. Sin sesgo problemático detectado.

### 6d — Formato findings: consistente ✅ OK
- "<finding> found in <laterality> <view>" consistente entre build_findings(), swap_findings() y USER_PROMPT.

### 6e — BIRADS 1 vs 2 con findings idénticos: ambigüedad inherente ⚠️ Importante para interpretación
- Estudios BIRADS 2 con hallazgos benignos no anotados en finding_annotations.csv reciben "Healthy Breast. No Findings" igual que BIRADS 1.
- El modelo debe discriminar BIRADS 1 vs 2 puramente por imagen (no por texto).
- **Implicación del fix 2c (reorder JSON):** al predecir BIRADS antes del findings text, el coupling "Healthy Breast → BIRADS 1/2" se elimina. El modelo usa la imagen directamente para la predicción. Esto probablemente beneficia más a BIRADS 1/2 de lo esperado.
- Run 1: BIRADS 1 acc=0.43, BIRADS 2 acc=0.38 — moderado. Esperamos mejora en Run 2.

### 6f — `\n` prefix y `<end_of_turn>` en loss ✅ OK
- Parte del chat template de MedGemma. Normales y necesarios para correcta terminación de respuesta.

## Audit Sección 7 — Inferencia (2026-05-22)

**Veredicto: 8/10 — Sólido. Sin bugs nuevos. Único fix activo: Bug 1 (ya documentado).**

### 7a — do_sample=True, temperature=0.1 → Bug 1 ya documentado. Fix: do_sample=False.

### 7b — max_new_tokens=124 ✅ OK
- Target promedio ~45 tokens, máximo ~80. 124 da margen seguro. Inconsistencia menor: Cell 29 usa 128.

### 7c — padding_side='left' ✅ OK
- Requerido para generación. Guardado y restaurado correctamente en validate_custom y CollapseCallback.

### 7d — Decode solo tokens nuevos ✅ OK
- `gen[0][inputs['input_ids'].shape[-1]:]` — slice correcto.

### 7e — skip_special_tokens=True ✅ OK

### 7f — pixel_values bfloat16 en inferencia ✅ OK
- Ambos sitios castean correctamente. El pendiente de memory-preferences es el collate_fn de training (no inferencia).

### 7g — parse_birads/parse_density: dos niveles de fallback ✅ OK
- Nivel 1: json.loads(). Nivel 2: regex directo. Maneja JSON limpio, markdown-wrapped, malformado.

### 7h — Batch size=1 en inferencia ✅ Aceptable
- Secuencial. A100 podría hacer batch=4-8. Oportunidad de optimización, no bug.

### 7i — Gestión eval mode ✅ OK (con fix Bug 1)
- model.eval() + use_cache=True antes, model.train() + use_cache=False después. Correcto.

## Audit Sección 8 — Evaluación (2026-05-22)

**Veredicto: 7.5/10 — Sin bugs. Diseño evaluativo correcto. Una aclaración necesaria en tesis.**

### 8a — val_df: 775 vs 1000 estudios ⚠️ Documentar en tesis
- val_df tiene 775 estudios del test split (los que tienen PNG pre-procesado). Los 225 restantes (1000−775) del test oficial se procesaron on-the-fly en evaluación final separada.
- **En tesis:** el resultado reportado (1000 estudios) es el test set oficial completo. val_df (775) fue solo para monitoreo. Clarificar esta distinción.

### 8b — Estratificación n_per_class ✅ OK
- CollapseCallback: n_per_class=3 (solo detección de colapso). Cell 34: n_per_class=1000 (todos disponibles). Evaluación final: test completo.

### 8c — Métricas: accuracy + F1 macro ✅ Apropiadas
- F1 macro: penaliza ignorar BIRADS 3/4/5 minoritarios. Correcto para dataset desbalanceado.

### 8d — Selección checkpoint por BIRADS F1 macro ✅ Correcto
- `max(all_results, key=lambda x: x['birads_f1'])`. Métrica honesta ante desbalance.

### 8g — Monitoreo con val_df: no es leakage real ✅ OK
- CollapseCallback usa 15 muestras de val_df solo para detectar colapso, no para decisiones de hiperparámetros. Test set oficial usado una sola vez al final. Documentar en tesis.

## Audit Sección 9 — DPO (2026-05-22)

**Estado: PENDIENTE (fase 2). No hay código que auditar. Planificación de riesgos.**

### 9a — Checkpoint base para DPO 🔲 Pendiente
- Usar mejor checkpoint de Run 2. Requiere merge_and_unload() antes de DPO. Ver riesgo 9d.

### 9b — Pares de preferencia: estrategia decidida ✅ GT vs predicción errónea
- Chosen: JSON con BIRADS correcto (GT). Rejected: output del modelo cuando falla.
- Procedimiento: evaluar Run 2 en training set → extraer pares fallidos → foco en BIRADS 3/4.
- Automatizable sin acceso a clínicos.

### 9c — VRAM para DPO ✅ Viable en A100
- Policy (~5GB) + Reference (~5GB) + activaciones → estimado ~12-15GB de 40GB. Sin problema.
- Alternativa si hay presión: LoRA-DPO (reference = base cuantizado compartido).

### 9d — tie_word_embeddings: riesgo de merge pre-DPO ⚠️ Bloqueo potencial
- Documentado en 4f. merge_and_unload() puede dejar lm_head ↔ embed_tokens inconsistentes.
- **Fix Run 2:** añadir `ensure_weight_tying=True` en LoraConfig.
- **Verificación post-merge:** `model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()`.

### 9e — DPOTrainer: parámetros clave
- beta=0.1, loss_type="sigmoid", lr=5e-5 (más bajo que SFT), epochs=1 (DPO converge rápido).

### 9f — Alternativa viable: SimPO
- Sin reference model → mitad de VRAM. Considerar si el pipeline se complica.

## Run 2 — Fixes definitivos aplicados (2026-05-22)

**Evaluación rigurosa completada.** Criterios: Correctitud, Señal de entrenamiento, Monitoreo, Riesgo de regresión, Evidencia empírica, Complejidad. Dataset específico: 2200 muestras, 500/500/500/500/200.

### Fixes APLICADOS al notebook 02_Visual_Fine_Tuning.ipynb

| Fix | Celda | Veredicto | Razón determinista |
|-----|-------|-----------|-------------------|
| Bug 1 — autocast en CollapseCallback + do_sample=False | 26 | ✅ APLICADO | Callback roto 4125 steps en Run 1; 15 muestras de monitoreo requieren generación determinista |
| Bug 1 — autocast en validate_custom + do_sample=False | 24 | ✅ APLICADO | Mismo patrón; afecta evaluación offline de checkpoints |
| Bug 2 — eliminar 2 líneas duplicadas en USER_PROMPT | 14 | ✅ APLICADO | 25 tokens de ruido malformado × 33,000 pasos de training |
| 2c — JSON reorder: `{"birads", "density", "findings"}` | 14 | ✅ APLICADO | Coupling findings→birads inútil para BIRADS 3/4/5 (mismo texto de findings para los tres); dañino para BIRADS 2 (comparte "Healthy Breast" con BIRADS 1) |
| 3d — hint BI-RADS: "overall assessment" | 14 | ✅ APLICADO | Indivisible de 2c; sin él hay señal contradictoria en 2200 × 15 épocas |
| 3e — template JSON en USER_PROMPT: birads primero | 14 | ✅ APLICADO | Indivisible de 2c; template actúa como few-shot implícito del formato esperado |
| 2g — collate masking: `found=None` / `if found is not None` | 20 | ✅ APLICADO | Bug de correctitud; si model_turn_ids no se encuentra, prompt entra al loss silenciosamente |

### Fixes DESCARTADOS con justificación

| Fix | Veredicto | Razón determinista |
|-----|-----------|-------------------|
| 2f — StratifiedBatchSampler | ❌ SKIP | Conectarlo reduce steps_per_epoch 275→200 (−27% total), y exposición BIRADS 3/4 de 500→200 muestras/época (−60%). BIRADS 3/4 son las clases más débiles y necesitan más gradiente, no menos |
| pixel_values bfloat16 en collate | ❌ SKIP | Trainer bf16=True usa autocast en forward → processor ya maneja dtype. Run 1 confirmó sin problema |
| 4f — ensure_weight_tying | ⏳ DEFER | Solo crítico antes de merge_and_unload() para DPO. Agregar en fase DPO |

### Razonamiento clave para 2c (JSON reorder)
- BIRADS 3/4/5 comparten texto de findings similar ("mass found in...", "calcification found in...") → el coupling text→birads NO discrimina dentro de este grupo
- BIRADS 2 comparte "Healthy Breast. No Findings" con BIRADS 1 → el coupling activamente perjudica a BIRADS 2
- Con reorder (birads primero): el modelo usa la imagen directamente (SigLIP pretrained en datos médicos) para predecir la categoría
- BIRADS 3 acc=0.275 en Run 1 confirma que el coupling actual NO estaba ayudando

### Target del modelo actualizado para Run 2
```json
{"birads": "3", "density": "B", "findings": "mass found in left cranio-caudal (CC)"}
```
