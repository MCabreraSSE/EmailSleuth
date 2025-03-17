from smolagents import CodeAgent, tool, LiteLLMModel
from huggingface_hub import list_models
import email
from email import policy
from datetime import datetime
import os
import glob
import json
import csv
import pandas as pd
import dns.resolver
import re
from typing import List, Dict, Optional
from email.utils import parsedate_to_datetime
import sys
import pkg_resources

def check_dependencies():
    """
    Verifica que todas las dependencias necesarias estén instaladas.
    """
    required_packages = {
        'pandas': 'pandas',
        'openpyxl': 'openpyxl',
        'dnspython': 'dnspython',
        'smolagents': 'smolagents',
        'huggingface_hub': 'huggingface_hub'
    }
    
    missing_packages = []
    for package, import_name in required_packages.items():
        try:
            pkg_resources.require(package)
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            missing_packages.append(package)
    
    if missing_packages:
        print("Faltan las siguientes dependencias:")
        for package in missing_packages:
            print(f"- {package}")
        print("\nPor favor, instálalas usando:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)

def check_ollama_server():
    """
    Verifica que el servidor Ollama esté ejecutándose.
    """
    import requests
    try:
        response = requests.get("http://172.17.0.2:11434/api/version")
        if response.status_code != 200:
            raise Exception("El servidor Ollama no está respondiendo correctamente")
        return True
    except Exception as e:
        print(f"Error al conectar con el servidor Ollama: {str(e)}")
        print("\nAsegúrate de que:")
        print("1. El servidor Ollama está instalado")
        print("2. El servidor está ejecutándose")
        print("3. La URL http://172.17.0.2:11434 es accesible")
        return False

def setup_environment():
    """
    Configura el entorno para la ejecución del programa.
    """
    # Verificar dependencias
    check_dependencies()
    
    # Verificar servidor Ollama
    if not check_ollama_server():
        sys.exit(1)
    
    # Verificar directorio de trabajo
    current_dir = os.getcwd()
    print(f"Directorio actual: {current_dir}")
    
    if not os.path.exists(current_dir):
        print(f"Error: El directorio actual {current_dir} no existe")
        sys.exit(1)
    
    if not os.access(current_dir, os.R_OK | os.W_OK):
        print(f"Error: No hay permisos de lectura/escritura en el directorio {current_dir}")
        sys.exit(1)
    
    # Crear directorios necesarios si no existen
    directories = ['adjuntos', 'reportes']
    for directory in directories:
        dir_path = os.path.join(current_dir, directory)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                print(f"Directorio creado: {dir_path}")
            except Exception as e:
                print(f"Error al crear el directorio {directory}: {str(e)}")
                sys.exit(1)
        else:
            print(f"Directorio existente: {dir_path}")

# Configurar el entorno antes de continuar
setup_environment()

# Configuración del modelo LLM
try:
    model = LiteLLMModel(
        model_id="ollama_chat/mistral",
        api_base="http://172.17.0.2:11434",  # Cambia esto si usas un servidor remoto
        api_key=None,  # No se necesita API key para Ollama local
        num_ctx=8192,  # Ajusta según la capacidad de tu hardware
        temperature=0.7,  # Ajusta la creatividad del modelo
        max_tokens=4096  # Límite de tokens por respuesta
    )
except Exception as e:
    print(f"Error al inicializar el modelo LLM: {str(e)}")
    raise

@tool
def model_download_tool(task: str) -> str:
    """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint.

    Args:
        task: The task for which to get the download count.
    """
    most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
    
    return most_downloaded_model.id

@tool
def find_eml_files(directory_path: str) -> list:
    """
    Busca todos los archivos .eml en un directorio.
    
    Args:
        directory_path: Ruta al directorio a buscar
        
    Returns:
        list: Lista de rutas de archivos .eml encontrados
    """
    if not os.path.exists(directory_path):
        raise ValueError(f"El directorio {directory_path} no existe")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"{directory_path} no es un directorio")
        
    eml_files = glob.glob(os.path.join(directory_path, "*.eml"))
    return eml_files

@tool
def analyze_email_authentication(msg: email.message.Message) -> dict:
    """
    Analiza los encabezados de autenticación de un correo electrónico.
    
    Args:
        msg: Objeto de mensaje de correo electrónico
        
    Returns:
        dict: Diccionario con el estado de autenticación
    """
    if not isinstance(msg, email.message.Message):
        raise ValueError("El objeto proporcionado no es un mensaje de correo electrónico válido")
        
    dkim_status = "No encontrado"
    spf_status = "No encontrado"
    arc_status = "No encontrado"
    
    try:
        for header, value in msg.items():
            if header.lower() == 'dkim-signature':
                dkim_status = "Presente"
            elif header.lower() == 'received-spf':
                spf_status = value
            elif header.lower() == 'arc-authentication-results':
                arc_status = value
        
        auth_checks = []
        if dkim_status == "Presente":
            auth_checks.append("DKIM")
        if "pass" in spf_status.lower():
            auth_checks.append("SPF")
        if arc_status != "No encontrado":
            auth_checks.append("ARC")
            
        if len(auth_checks) == 3:
            authentication_status = "Fully verificado"
        elif len(auth_checks) >= 1:
            authentication_status = f"Parcialmente verificado ({', '.join(auth_checks)})"
        else:
            authentication_status = "No verificado"
        
        return {
            "dkim": dkim_status,
            "spf": spf_status,
            "arc": arc_status,
            "authentication_status": authentication_status,
            "auth_checks": auth_checks
        }
    except Exception as e:
        raise ValueError(f"Error al analizar la autenticación: {str(e)}")

@tool
def extract_email_content(msg: email.message.Message) -> str:
    """
    Extrae el contenido de texto plano de un correo electrónico.
    
    Args:
        msg: Objeto de mensaje de correo electrónico
        
    Returns:
        str: Contenido del correo en texto plano
    """
    if not isinstance(msg, email.message.Message):
        raise ValueError("El objeto proporcionado no es un mensaje de correo electrónico válido")
        
    try:
        content = ""
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                content = part.get_content()
                break
        return content or "No se encontró contenido de texto plano"
    except Exception as e:
        raise ValueError(f"Error al extraer el contenido: {str(e)}")

@tool
def save_analysis_report(analysis_data: dict, directory_path: str) -> str:
    """
    Guarda el reporte de análisis en un archivo.
    
    Args:
        analysis_data: Diccionario con los datos del análisis
        directory_path: Directorio donde guardar el reporte
        
    Returns:
        str: Ruta del archivo guardado
    """
    if not os.path.exists(directory_path):
        raise ValueError(f"El directorio {directory_path} no existe")
    
    if not isinstance(analysis_data, dict):
        raise ValueError("analysis_data debe ser un diccionario")
        
    if 'text_report' not in analysis_data or 'json_data' not in analysis_data:
        raise ValueError("analysis_data debe contener 'text_report' y 'json_data'")
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar en formato texto
        txt_file = os.path.join(directory_path, f"analisis_correos_{timestamp}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(analysis_data['text_report'])
        
        # Guardar en formato JSON
        json_file = os.path.join(directory_path, f"analisis_correos_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data['json_data'], f, indent=2, ensure_ascii=False)
        
        return f"Reportes guardados en:\n- {txt_file}\n- {json_file}"
    except Exception as e:
        raise ValueError(f"Error al guardar los reportes: {str(e)}")

@tool
def extract_attachments(msg: email.message.Message, save_path: str) -> List[Dict]:
    """
    Extrae y guarda los archivos adjuntos de un correo electrónico.
    
    Args:
        msg: Objeto de mensaje de correo electrónico
        save_path: Directorio donde guardar los archivos adjuntos
        
    Returns:
        List[Dict]: Lista de diccionarios con información de los archivos adjuntos
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    attachments = []
    for part in msg.walk():
        if part.get_content_maintype() == 'multipart':
            continue
            
        filename = part.get_filename()
        if filename:
            try:
                filepath = os.path.join(save_path, filename)
                with open(filepath, 'wb') as f:
                    f.write(part.get_payload(decode=True))
                attachments.append({
                    'filename': filename,
                    'type': part.get_content_type(),
                    'size': os.path.getsize(filepath),
                    'path': filepath
                })
                print(f"Archivo adjunto guardado: {filename}")
            except Exception as e:
                print(f"Error al guardar el archivo adjunto {filename}: {str(e)}")
                
    return attachments

@tool
def verify_domain(domain: str) -> Dict:
    """
    Verifica los registros DNS de un dominio para autenticación de correo.
    
    Args:
        domain: Dominio a verificar
        
    Returns:
        Dict: Diccionario con los resultados de la verificación
    """
    results = {
        'domain': domain,
        'mx_records': [],
        'spf_record': None,
        'dkim_record': None,
        'dmarc_record': None
    }
    
    try:
        # Verificar registros MX
        mx_records = dns.resolver.resolve(domain, 'MX')
        results['mx_records'] = [str(x.exchange).rstrip('.') for x in mx_records]
        
        # Verificar registro SPF
        try:
            spf_records = dns.resolver.resolve(domain, 'TXT')
            for record in spf_records:
                if 'v=spf1' in str(record):
                    results['spf_record'] = str(record)
                    break
        except:
            pass
            
        # Verificar registro DKIM
        try:
            dkim_records = dns.resolver.resolve(f'default._domainkey.{domain}', 'TXT')
            results['dkim_record'] = str(dkim_records[0])
        except:
            pass
            
        # Verificar registro DMARC
        try:
            dmarc_records = dns.resolver.resolve(f'_dmarc.{domain}', 'TXT')
            results['dmarc_record'] = str(dmarc_records[0])
        except:
            pass
            
    except Exception as e:
        print(f"Error al verificar el dominio {domain}: {str(e)}")
        
    return results

@tool
def filter_emails_by_date(emails_data: List[Dict], start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> List[Dict]:
    """
    Filtra correos electrónicos por rango de fechas.
    
    Args:
        emails_data: Lista de diccionarios con datos de correos
        start_date: Fecha inicial (formato: YYYY-MM-DD)
        end_date: Fecha final (formato: YYYY-MM-DD)
        
    Returns:
        List[Dict]: Lista filtrada de correos
    """
    filtered_emails = []
    
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
        end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
        
        for email_data in emails_data:
            try:
                email_date = parsedate_to_datetime(email_data['fecha'])
                if start and email_date < start:
                    continue
                if end and email_date > end:
                    continue
                filtered_emails.append(email_data)
            except:
                continue
                
    except Exception as e:
        print(f"Error al filtrar correos por fecha: {str(e)}")
        
    return filtered_emails

@tool
def export_to_csv(data: List[Dict], output_file: str) -> str:
    """
    Exporta los datos de análisis a formato CSV.
    
    Args:
        data: Lista de diccionarios con datos de correos
        output_file: Ruta del archivo CSV de salida
        
    Returns:
        str: Mensaje de confirmación
    """
    try:
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        return f"Datos exportados exitosamente a {output_file}"
    except Exception as e:
        raise ValueError(f"Error al exportar a CSV: {str(e)}")

@tool
def export_to_excel(data: List[Dict], output_file: str) -> str:
    """
    Exporta los datos de análisis a formato Excel.
    
    Args:
        data: Lista de diccionarios con datos de correos
        output_file: Ruta del archivo Excel de salida
        
    Returns:
        str: Mensaje de confirmación
    """
    try:
        df = pd.DataFrame(data)
        df.to_excel(output_file, index=False, engine='openpyxl')
        return f"Datos exportados exitosamente a {output_file}"
    except Exception as e:
        raise ValueError(f"Error al exportar a Excel: {str(e)}")

@tool
def generate_detailed_spreadsheet(emails_data: List[Dict], output_file: str) -> str:
    """
    Genera una planilla detallada con la información de cada correo analizado.
    
    Args:
        emails_data: Lista de diccionarios con datos de correos
        output_file: Ruta del archivo Excel de salida
        
    Returns:
        str: Mensaje de confirmación
    """
    try:
        print("\nGenerando planilla detallada...")
        print(f"Datos recibidos: {len(emails_data)} correos")
        
        # Crear un DataFrame con los datos principales
        df = pd.DataFrame(emails_data)
        
        # Verificar qué columnas están disponibles
        available_columns = df.columns.tolist()
        print(f"Columnas disponibles: {available_columns}")
        
        # Crear un archivo Excel con múltiples hojas
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Hoja principal con todos los datos
            print("Creando hoja de datos completos...")
            df.to_excel(writer, sheet_name='Datos Completos', index=False)
            
            # Hoja de autenticación
            print("Creando hoja de autenticación...")
            auth_columns = ['archivo', 'dkim', 'spf', 'arc', 'authentication_status']
            if all(col in df.columns for col in auth_columns):
                auth_data = df[auth_columns]
                auth_data.to_excel(writer, sheet_name='Autenticación', index=False)
            else:
                print(f"Columnas de autenticación faltantes: {[col for col in auth_columns if col not in df.columns]}")
            
            # Hoja de dominios
            print("Creando hoja de dominios...")
            if 'dominio' in df.columns:
                domain_data = df[['archivo', 'dominio']].drop_duplicates()
                domain_data.to_excel(writer, sheet_name='Dominios', index=False)
            else:
                print("Columna 'dominio' no encontrada")
            
            # Hoja de estadísticas
            print("Creando hoja de estadísticas...")
            stats = {
                'Métrica': [
                    'Total de correos',
                    'Correos procesados',
                    'Correos con error',
                    'Correos con DKIM',
                    'Correos con SPF exitoso',
                    'Correos con ARC',
                    'Correos totalmente verificados',
                    'Correos parcialmente verificados',
                    'Correos no verificados'
                ],
                'Cantidad': [
                    len(df),
                    len(df[df['archivo'].notna()]),
                    len(df) - len(df[df['archivo'].notna()]),
                    len(df[df['dkim'] == 'Presente']) if 'dkim' in df.columns else 0,
                    len(df[df['spf'].str.contains('pass', case=False, na=False)]) if 'spf' in df.columns else 0,
                    len(df[df['arc'] != 'No encontrado']) if 'arc' in df.columns else 0,
                    len(df[df['authentication_status'] == 'Fully verificado']) if 'authentication_status' in df.columns else 0,
                    len(df[df['authentication_status'].str.contains('Parcialmente', na=False)]) if 'authentication_status' in df.columns else 0,
                    len(df[df['authentication_status'] == 'No verificado']) if 'authentication_status' in df.columns else 0
                ]
            }
            pd.DataFrame(stats).to_excel(writer, sheet_name='Estadísticas', index=False)
        
        print(f"Planilla detallada guardada en: {output_file}")
        return f"Planilla detallada guardada en: {output_file}"
    except Exception as e:
        print(f"Error al generar la planilla detallada: {str(e)}")
        print(f"Columnas disponibles: {df.columns.tolist() if 'df' in locals() else 'No se pudo crear el DataFrame'}")
        raise ValueError(f"Error al generar la planilla detallada: {str(e)}")

@tool
def analyze_eml_files(directory_path: str, start_date: Optional[str] = None, 
                     end_date: Optional[str] = None, extract_attachments: bool = False) -> str:
    """
    Analiza archivos de correo electrónico .eml en un directorio.
    
    Args:
        directory_path: Ruta al directorio que contiene los archivos .eml a analizar
        start_date: Fecha inicial para filtrar correos (formato: YYYY-MM-DD)
        end_date: Fecha final para filtrar correos (formato: YYYY-MM-DD)
        extract_attachments: Si se deben extraer los archivos adjuntos de los correos
        
    Returns:
        str: Reporte detallado del análisis de los correos
    """
    try:
        print(f"\nIniciando análisis en el directorio: {directory_path}")
        
        # Verificar directorio
        if not os.path.exists(directory_path):
            print(f"Error: El directorio {directory_path} no existe")
            return f"Error: El directorio {directory_path} no existe"
        
        if not os.access(directory_path, os.R_OK | os.W_OK):
            print(f"Error: No hay permisos de lectura/escritura en el directorio {directory_path}")
            return f"Error: No hay permisos de lectura/escritura en el directorio {directory_path}"
        
        # Buscar archivos .eml
        print("Buscando archivos .eml...")
        eml_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.eml'):
                    eml_files.append(os.path.join(root, file))
        
        if not eml_files:
            print(f"No se encontraron archivos .eml en el directorio: {directory_path}")
            return f"No se encontraron archivos .eml en el directorio: {directory_path}"
        
        print(f"Se encontraron {len(eml_files)} archivos .eml:")
        for file in eml_files:
            print(f"- {os.path.basename(file)}")
        
        # Crear directorio para reportes si no existe
        reportes_dir = os.path.join(directory_path, "reportes")
        if not os.path.exists(reportes_dir):
            os.makedirs(reportes_dir)
            print(f"Directorio de reportes creado: {reportes_dir}")
        
        # Crear directorio para adjuntos si es necesario
        attachments_dir = os.path.join(directory_path, "adjuntos")
        if extract_attachments and not os.path.exists(attachments_dir):
            os.makedirs(attachments_dir)
            print(f"Directorio de adjuntos creado: {attachments_dir}")
        
        all_analyses = []
        json_data = {
            "fecha_analisis": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_correos": len(eml_files),
            "correos": [],
            "estadisticas": {
                "total_verificados": 0,
                "total_parcialmente_verificados": 0,
                "total_no_verificados": 0,
                "dkim_presente": 0,
                "spf_presente": 0,
                "arc_presente": 0,
                "total_adjuntos": 0,
                "correos_procesados": 0,
                "correos_con_error": 0
            }
        }
        
        # Procesar cada archivo
        for i, eml_file in enumerate(eml_files, 1):
            try:
                print(f"\nProcesando archivo {i}/{len(eml_files)}: {os.path.basename(eml_file)}")
                
                with open(eml_file, 'rb') as f:
                    msg = email.message_from_bytes(f.read(), policy=policy.default)
                
                # Extraer información básica
                email_data = {
                    "archivo": os.path.basename(eml_file),
                    "remitente": msg.get('from', 'Desconocido'),
                    "destinatario": msg.get('to', 'Desconocido'),
                    "asunto": msg.get('subject', 'Sin asunto'),
                    "fecha": msg.get('date', 'Desconocida')
                }
                
                # Extraer dominio del remitente
                domain_match = re.search(r'@([\w.-]+)', email_data['remitente'])
                if domain_match:
                    domain = domain_match.group(1)
                    email_data['dominio'] = domain
                    print(f"Verificando dominio: {domain}")
                    domain_verification = verify_domain(domain)
                    email_data['verificacion_dominio'] = domain_verification
                
                # Analizar autenticación
                auth_data = analyze_email_authentication(msg)
                email_data.update(auth_data)
                
                # Extraer archivos adjuntos si se solicita
                if extract_attachments:
                    print("Extrayendo archivos adjuntos...")
                    try:
                        attachments = extract_attachments(msg, attachments_dir)
                        email_data['adjuntos'] = attachments
                        json_data["estadisticas"]["total_adjuntos"] += len(attachments)
                        print(f"Se extrajeron {len(attachments)} archivos adjuntos")
                    except Exception as e:
                        print(f"Error al extraer archivos adjuntos: {str(e)}")
                        email_data['adjuntos'] = []
                        json_data["estadisticas"]["total_adjuntos"] += 0
                
                # Actualizar estadísticas
                if auth_data['authentication_status'] == "Fully verificado":
                    json_data["estadisticas"]["total_verificados"] += 1
                elif "Parcialmente verificado" in auth_data['authentication_status']:
                    json_data["estadisticas"]["total_parcialmente_verificados"] += 1
                else:
                    json_data["estadisticas"]["total_no_verificados"] += 1
                
                if auth_data['dkim'] == "Presente":
                    json_data["estadisticas"]["dkim_presente"] += 1
                if "pass" in auth_data['spf'].lower():
                    json_data["estadisticas"]["spf_presente"] += 1
                if auth_data['arc'] != "No encontrado":
                    json_data["estadisticas"]["arc_presente"] += 1
                
                # Extraer contenido
                email_data["contenido"] = extract_email_content(msg)
                
                # Agregar a datos JSON
                json_data["correos"].append(email_data)
                json_data["estadisticas"]["correos_procesados"] += 1
                
                # Formatear análisis en texto
                analysis = f"""
                Análisis del correo electrónico: {email_data['archivo']}
                ------------------------------
                De: {email_data['remitente']}
                Para: {email_data['destinatario']}
                Asunto: {email_data['asunto']}
                Fecha: {email_data['fecha']}
                
                Análisis de Autenticación:
                ------------------------
                DKIM: {email_data['dkim']}
                SPF: {email_data['spf']}
                ARC: {email_data['arc']}
                Estado de Autenticación: {email_data['authentication_status']}
                
                Verificación de Dominio:
                ----------------------
                """
                
                if 'verificacion_dominio' in email_data:
                    domain_info = email_data['verificacion_dominio']
                    analysis += f"""
                    Dominio: {domain_info['domain']}
                    Registros MX: {', '.join(domain_info['mx_records'])}
                    Registro SPF: {domain_info['spf_record'] or 'No encontrado'}
                    Registro DKIM: {domain_info['dkim_record'] or 'No encontrado'}
                    Registro DMARC: {domain_info['dmarc_record'] or 'No encontrado'}
                    """
                
                if extract_attachments and 'adjuntos' in email_data:
                    analysis += f"""
                    Archivos Adjuntos:
                    -----------------
                    Total: {len(email_data['adjuntos'])}
                    """
                    for adj in email_data['adjuntos']:
                        analysis += f"- {adj['filename']} ({adj['type']}, {adj['size']} bytes)\n"
                
                analysis += f"""
                Contenido:
                {email_data['contenido']}
                """
                
                all_analyses.append(analysis)
                print(f"Archivo procesado exitosamente: {os.path.basename(eml_file)}")
                
            except Exception as e:
                print(f"Error al procesar el archivo {eml_file}: {str(e)}")
                json_data["estadisticas"]["correos_con_error"] += 1
                continue
        
        # Filtrar por fecha si se especifican
        if start_date or end_date:
            print(f"\nFiltrando correos por fecha: {start_date} - {end_date}")
            json_data["correos"] = filter_emails_by_date(
                json_data["correos"], start_date, end_date
            )
            print(f"Correos después del filtrado: {len(json_data['correos'])}")
        
        # Crear reporte final
        final_report = f"""
        Reporte de Análisis de Correos Electrónicos
        ========================================
        Total de correos analizados: {len(eml_files)}
        Fecha del análisis: {json_data['fecha_analisis']}
        
        Estadísticas:
        ------------
        - Total de correos procesados: {json_data['estadisticas']['correos_procesados']}
        - Correos con error: {json_data['estadisticas']['correos_con_error']}
        - Total de correos verificados: {json_data['estadisticas']['total_verificados']}
        - Total de correos parcialmente verificados: {json_data['estadisticas']['total_parcialmente_verificados']}
        - Total de correos no verificados: {json_data['estadisticas']['total_no_verificados']}
        - Correos con DKIM presente: {json_data['estadisticas']['dkim_presente']}
        - Correos con SPF exitoso: {json_data['estadisticas']['spf_presente']}
        - Correos con ARC presente: {json_data['estadisticas']['arc_presente']}
        - Total de archivos adjuntos: {json_data['estadisticas']['total_adjuntos']}
        
        {'='*50}
        """.join(all_analyses)
        
        # Guardar reportes
        print("\nGuardando reportes...")
        analysis_data = {
            "text_report": final_report,
            "json_data": json_data
        }
        output_files = save_analysis_report(analysis_data, reportes_dir)
        print(f"Reportes básicos guardados: {output_files}")
        
        # Exportar a CSV y Excel
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = os.path.join(reportes_dir, f"analisis_correos_{timestamp}.csv")
        excel_file = os.path.join(reportes_dir, f"analisis_correos_{timestamp}.xlsx")
        detailed_excel = os.path.join(reportes_dir, f"planilla_detallada_{timestamp}.xlsx")
        
        print("Exportando reportes adicionales...")
        export_to_csv(json_data["correos"], csv_file)
        print(f"CSV guardado en: {csv_file}")
        
        export_to_excel(json_data["correos"], excel_file)
        print(f"Excel básico guardado en: {excel_file}")
        
        generate_detailed_spreadsheet(json_data["correos"], detailed_excel)
        print(f"Planilla detallada guardada en: {detailed_excel}")
        
        print("\nAnálisis completado exitosamente!")
        return f"""
        Reportes generados:
        ------------------
        {output_files}
        
        Reportes adicionales:
        -------------------
        - CSV: {csv_file}
        - Excel básico: {excel_file}
        - Planilla detallada: {detailed_excel}
        
        {final_report}
        """
        
    except Exception as e:
        error_msg = f"Error al analizar los archivos: {str(e)}"
        print(f"\n{error_msg}")
        return error_msg

# Ejemplos de uso del agente
if __name__ == "__main__":
    try:
        # Ejemplo 1: Análisis completo de correos en un directorio
        print("\n=== Ejemplo 1: Análisis completo de correos ===")
        
        # Obtener el directorio actual
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Analizando correos en el directorio: {current_dir}")
        
        # Ejecutar el análisis
        result = analyze_eml_files(
            directory_path=current_dir,
            start_date=None,
            end_date=None,
            extract_attachments=True
        )
        
        print("\nResultado del análisis:")
        print(result)
        
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        print("\nAsegúrate de que:")
        print("1. El servidor Ollama está ejecutándose en http://172.17.0.2:11434")
        print("2. Tienes archivos .eml en el directorio actual")
        print("3. Tienes permisos de lectura/escritura en el directorio")
        print("4. Todas las dependencias están instaladas (pandas, openpyxl, dnspython)")
        print("\nPara instalar las dependencias, ejecuta:")
        print("pip install pandas openpyxl dnspython smolagents huggingface_hub")


