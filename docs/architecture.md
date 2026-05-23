# Arquitectura del Sistema

## Modelo base
- **Nombre:** `google/medgemma-4b-it`
- **Tipo:** Vision Language Model (VLM)
- **Razón de elección:** Pre-entrenamiento en dominio médico

## Fine-tuning: QLoRA
- Cuantización 4-bit NF4 con `BitsAndBytesConfig` (`bnb_4bit_double_quant=True`)
- `lora_r=32`, `lora_alpha=32`, `lora_dropout=0.05` — Run 3: scaling=α/r=1.0 (estándar LoRA), ΔW_eff=1e-4
- Target modules: todos los linear del LM (q/k/v/o_proj + gate/up/down_proj), excluyendo vision encoder
- 238 capas LoRA — 730M parámetros entrenables (22.69% del total)
- `modules_to_save=['lm_head']` — LM head entrenado completo
- Optimizer: `adamw_bnb_8bit`, LR: `1e-4`, scheduler: cosine, warmup: 5%
- Batch efectivo: 8 (batch_size=1 × grad_accum=8), epochs: 10 — Run 3: reducido de 15; overfitting confirmado post-ep10
- `attn_implementation='eager'` requerido para gradient checkpointing con 4-bit

## Dataset y rebalanceo
- Train: 2200 muestras balanceadas (500 por BIRADS 1-4, 200 para BIRADS 5)
- Val: 775 muestras del test split de VinDr-Mammo
- Rebalanceo: downsample para BIRADS 1/2, upsample con augmentadas (flip + rot) para BIRADS 3/4/5
- `StratifiedBatchSampler`: garantiza todas las clases en cada batch efectivo
- Imágenes: 896×896 PNG preprocesadas (Otsu ROI crop + CLAHE + resize 448×448 → merge 2×2)

## Alineación: DPO
- *(Pendiente — fase 2 del proyecto)*

## Pipeline de inferencia
1. Construir cuadrícula 2x2 con las 4 proyecciones
2. Tokenizar imagen + prompt con el procesador de MedGemma
3. Generar reporte con `model.generate()`
4. Parsear salida: extraer Densidad ACR + BI-RADS + justificación
