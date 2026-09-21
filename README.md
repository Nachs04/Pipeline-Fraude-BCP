# Pipeline CI/CD - Prototipo de detección de transacciones sospechosas

Pipeline académico alineado con el proyecto:
**Desarrollo de un prototipo basado en machine learning para la prevención del fraude mediante la detección de transacciones sospechosas y la gestión de alertas: propuesta para el BCP.**

## Flujo

1. Integración: rama / Pull Request
2. Calidad: sintaxis, dependencias y secretos
3. Pruebas: API, datos y estados
4. Evaluación ML: división temporal 60/20/20 y comparación con reglas
5. Evaluación IA: validación de fidelidad de salidas generativas (simulada)
6. Empaquetado: artefactos y manifiesto de versiones
7. Staging: prueba de un caso completo
8. Promoción: aprobación antes de demostración

El proyecto usa datos sintéticos o de simulación. No utiliza datos transaccionales reales del BCP.

## Ejecución local

```bash
python -m pip install -r requirements.txt
python -m src.pipeline --data data/transactions.csv
```

El pipeline genera:
- `artifacts/metrics.json`
- `artifacts/manifest.json`
- `artifacts/model.joblib`
- `artifacts/staging_result.json`

## Columnas esperadas

CSV mínimo:

- `timestamp`
- `customer_id`
- `terminal_id`
- `amount`
- `fraud`

`fraud` debe ser 0/1.
