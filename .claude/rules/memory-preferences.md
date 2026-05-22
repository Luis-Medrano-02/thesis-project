# Preferencias de Desarrollo

## Hardware (Colab A100)
- Usar siempre `torch.bfloat16`
- Cargar modelos con `low_cpu_mem_usage=True` y `device_map="auto"`
- Limpiar VRAM tras evaluar checkpoints: `del model; torch.cuda.empty_cache(); gc.collect()`

## Estándares de código
- Framework: HuggingFace Transformers + PEFT
- Fine-tuning: QLoRA con `BitsAndBytesConfig`
- Para inferencia/evaluación: configurar `padding_side="left"` en el procesador y restaurarlo después
- No usar `.cuda()` directo — dejar que `device_map="auto"` maneje la ubicación

## Rigor clínico
- Los prompts y salidas deben emular estructura de reporte radiológico
- Evitar alucinaciones — las salidas deben estar ancladas en hallazgos visibles
- Formato de salida esperado: Densidad ACR (A/B/C/D) + BI-RADS (1-5) + justificación

## Bugs resueltos

### Bug 1 — dtype mismatch en model.generate() durante CollapseCallback
- **Error:** `expected scalar type BFloat16 but found Float`
- **Causa:** `model.generate()` crea tensores internos (KV cache, rotary embeddings) en float32 cuando el modelo alterna entre train/eval mode con gradient checkpointing activo
- **Fix:** envolver `model.generate()` en `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`
- **También:** cambiar `do_sample=True, temperature=0.1` → `do_sample=False` en validación para reproducibilidad

### Bug 2 — USER_PROMPT duplica SYSTEM_PROMPT
- **Error:** las primeras dos líneas de USER_PROMPT repiten el texto del SYSTEM_PROMPT
- **Fix:** eliminar esas dos líneas del inicio de USER_PROMPT

### Bug 3 — collate_fn masking (resuelto en Run 2)
- `found = -1` / `if found > 0` → corregido a `found = None` / `if found is not None`
- Cast de pixel_values a bfloat16 en collate_fn: **NO necesario** — Trainer con bf16=True usa autocast automáticamente en el forward pass
