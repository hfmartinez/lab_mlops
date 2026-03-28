# ML Regression API (PyCaret + Flask)

## Estudiantes

- Heberth Martinez
- Diana Mazuera
- Natalia Santamaria



Servicio REST para predicción de costos médicos (`charges`) usando un modelo de regresión entrenado con PyCaret.

## Stack

- Python 3.11  
- PyCaret 3.x  
- Flask  
- Gunicorn (producción)

## Estructura

```
.
├── app
│   ├── app.py
│   ├── dockerfile
│   ├── model.pkl
│   ├── requirements.txt
│   ├── static
│   │   └── style.css
│   └── templates
│       └── home.html
├── README.md
└── scripts
    ├── requirements.txt
    └── train.py
```

## Setup local

```bash
python3.11 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Entrenar modelo

```bash
python scripts/train.py
```

Esto genera:

```
model.pkl # Copiar al folder /app
```

## Ejecutar aplicación

```bash
python app/app.py
```

Por defecto:

```
http://localhost:5000
```

## Ejecutar con Gunicorn (prod)

```bash
export FLASK_APP=app.py
export FLASK_RUN_HOST=0.0.0.0
flask run
```

## Endpoint

### POST /predict

Formulario HTML.

Campos esperados:

```json
{
  "age": 19,
  "sex": "female",
  "bmi": 27.9,
  "children": 0,
  "smoker": "yes",
  "region": "southwest"
}
```

Respuesta: render HTML con predicción.

---

### POST /predict_api

JSON input.

```bash
curl -X POST http://localhost:5000/predict_api   -H "Content-Type: application/json"   -d '{
        "age": 19,
        "sex": "female",
        "bmi": 27.9,
        "children": 0,
        "smoker": "yes",
        "region": "southwest"
      }'
```

Respuesta:

```json
{
  "prediction": 16884.92
}
```

## Notas

- El modelo usa PyCaret (`predict_model`) y retorna `prediction_label`.
- Asegúrate de usar las mismas versiones de dependencias para entrenar y servir.
- En Docker (Debian/Ubuntu), instalar:

```bash
apt-get install -y libgomp1
```

para evitar errores con LightGBM.

