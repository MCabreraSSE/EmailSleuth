# Análisis Forense de Correos Electrónicos

Esta aplicación web permite analizar archivos EML (correos electrónicos) para realizar análisis forense.

## Características

- Carga múltiple de archivos EML
- Análisis automático de correos electrónicos
- Generación de reportes en diferentes formatos
- Interfaz web intuitiva

## Requisitos

- Python 3.x
- Flask
- Dependencias listadas en requirements.txt

## Instalación

1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecutar la aplicación:
```bash
python app.py
```

## Uso

1. Abrir el navegador y acceder a `http://localhost:5000`
2. Seleccionar uno o más archivos EML
3. Esperar el análisis
4. Descargar los reportes generados

## Estructura del Proyecto

```
Forensic/
├── app.py              # Aplicación principal Flask
├── base.py            # Funciones de análisis
├── templates/         # Plantillas HTML
├── uploads/          # Directorio temporal para archivos
└── reportes/         # Directorio para reportes generados
```

## Licencia

[Especificar la licencia del proyecto] 