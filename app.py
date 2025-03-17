from flask import Flask, render_template, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
from base import analyze_eml_files
import datetime

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'eml'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Crear directorio de uploads si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No se han seleccionado archivos'}), 400
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No se han seleccionado archivos'}), 400
    
    # Crear directorio temporal para los archivos
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], timestamp)
    os.makedirs(temp_dir)
    
    # Guardar archivos
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(temp_dir, filename))
    
    try:
        # Analizar los archivos
        result = analyze_eml_files(
            directory_path=temp_dir,
            start_date=None,
            end_date=None,
            extract_attachments=True
        )
        
        # Buscar los archivos de reporte generados
        reportes_dir = os.path.join(temp_dir, "reportes")
        reportes = []
        if os.path.exists(reportes_dir):
            for file in os.listdir(reportes_dir):
                if file.endswith(('.txt', '.json', '.csv', '.xlsx')):
                    reportes.append({
                        'nombre': file,
                        'ruta': os.path.join(reportes_dir, file)
                    })
        
        return jsonify({
            'mensaje': 'Análisis completado exitosamente',
            'reportes': reportes
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 