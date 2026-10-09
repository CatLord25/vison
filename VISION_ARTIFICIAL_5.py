# -*- coding: utf-8 -*-  # Guarda los comentarios y mensajes de este módulo en español mediante UTF-8.
# PASO 5: leer dígitos de SIETE SEGMENTOS mediante ocupación geométrica; no incluye modelos externos ni OCR de texto impreso.
# Ejecutar después del paso 4: python VISION_ARTIFICIAL_5.py; añadir --vivo para adquirir y registrar lecturas de la cámara fija.
# Las ROIs deben encerrar una sola fila de cifras, signo y punto; las unidades y los marcos deben quedar fuera.
# Para un display con otra tipografía se necesita diseñar plantillas propias; este lector no afirma reconocer cualquier pantalla.
import argparse  # Interpreta opciones de ejecución sin modificar constantes dentro del archivo.
import csv  # Escribe mediciones tabulares usando exclusivamente la biblioteca estándar de Python.
import json  # Lee los manifiestos anteriores y guarda resultados auditables con texto, estados y segmentos.
import math  # Comprueba que el intervalo de muestreo sea un número finito antes de utilizarlo en comparaciones temporales.
import re  # Comprueba el formato numérico y evita convertir signos o decimales mal colocados en una medición válida.
import time  # Usa un reloj monótono para espaciar las filas de adquisición sin depender de ajustes del reloj civil.
from datetime import datetime, timedelta, timezone  # Construye marcas temporales explícitas en UTC-4, zona horaria de Bolivia.
from decimal import Decimal  # Conserva la representación decimal exacta evitando el redondeo binario propio de float.
from pathlib import Path  # Resuelve archivos respecto del proyecto y admite directorios personalizados para las pruebas.
try:  # Comprueba las dependencias de imagen antes de intentar usar cámara o archivos anteriores.
    import cv2  # Utiliza OpenCV para matrices, geometría, visualización y adquisición; no importa motores de OCR.
    import numpy as np  # Usa NumPy únicamente para arreglos de píxeles y operaciones numéricas auxiliares.
except ImportError as error:  # Presenta un diagnóstico accionable si falta una dependencia del procesamiento.
    raise SystemExit(f"ERROR: instala las dependencias con python -m pip install opencv-python numpy. Detalle: {error}")  # Detiene la ejecución sin simular una lectura.
from VISION_ARTIFICIAL_2 import aplicar_filtros  # Reutiliza exactamente el procesamiento guardado durante la calibración del paso 2.
from VISION_ARTIFICIAL_4 import detectar_elementos  # Reutiliza la segmentación del paso 4 tanto en fotografías como en adquisición viva.
BASE = Path(__file__).resolve().parent  # Fija la raíz del proyecto independientemente del directorio desde el que se invoque Python.
ZONA_BOLIVIA = timezone(timedelta(hours=-4))  # UTC-4 expresa el desplazamiento horario solicitado para fechar las mediciones.
PATRONES = {  # Define el código determinista de cada cifra en orden (a,b,c,d,e,f,g), sin entrenamiento ni probabilidades.
    (1, 1, 1, 1, 1, 1, 0): "0",  # El cero enciende los seis segmentos exteriores y deja apagada la barra central g.
    (0, 1, 1, 0, 0, 0, 0): "1",  # El uno enciende únicamente los segmentos derechos superior b e inferior c.
    (1, 1, 0, 1, 1, 0, 1): "2",  # El dos enciende a,b,d,e,g y mantiene apagados c y f.
    (1, 1, 1, 1, 0, 0, 1): "3",  # El tres enciende las tres barras horizontales y los dos segmentos derechos.
    (0, 1, 1, 0, 0, 1, 1): "4",  # El cuatro combina la parte superior izquierda, el centro y las dos barras derechas.
    (1, 0, 1, 1, 0, 1, 1): "5",  # El cinco enciende a,c,d,f,g y mantiene apagados b y e.
    (1, 0, 1, 1, 1, 1, 1): "6",  # El seis enciende todo salvo la barra superior derecha b.
    (1, 1, 1, 0, 0, 0, 0): "7",  # El siete combina la barra superior con las dos barras derechas.
    (1, 1, 1, 1, 1, 1, 1): "8",  # El ocho enciende los siete segmentos y por ello ocupa todas las zonas de lectura.
    (1, 1, 1, 1, 0, 1, 1): "9",  # El nueve enciende todo salvo la barra inferior izquierda e.
}  # Termina la tabla; los patrones que no aparecen aquí se representarán por '?' sin aproximarlos a otra cifra.
ZONAS = (  # Define siete rectángulos normalizados (x0,y0,x1,y1); los límites derechos e inferiores son exclusivos.
    (0.20, 0.00, 0.80, 0.15),  # Segmento a: barra horizontal superior, lejos de las barras verticales laterales.
    (0.75, 0.16, 1.00, 0.44),  # Segmento b: barra derecha situada en la mitad superior del dígito.
    (0.75, 0.56, 1.00, 0.84),  # Segmento c: barra derecha situada en la mitad inferior del dígito.
    (0.20, 0.85, 0.80, 1.00),  # Segmento d: barra horizontal inferior.
    (0.00, 0.56, 0.25, 0.84),  # Segmento e: barra izquierda inferior.
    (0.00, 0.16, 0.25, 0.44),  # Segmento f: barra izquierda superior.
    (0.20, 0.43, 0.80, 0.57),  # Segmento g: barra horizontal central.
)  # Las proporciones presuponen dígitos verticales de siete segmentos; inclinación o perspectiva necesitan otra calibración.
def leer_json(ruta):  # Recibe una ruta a JSON y devuelve su objeto Python; informa al usuario si falta un paso anterior.
    if not ruta.is_file():  # Comprueba que el manifiesto existe antes de intentar interpretarlo.
        raise ValueError(f"Falta {ruta}. Ejecuta primero el paso que lo genera.")  # Evita continuar con datos inventados o configuración implícita.
    return json.loads(ruta.read_text(encoding="utf-8"))  # Decodifica UTF-8 y analiza el JSON mediante la biblioteca estándar.
def cargar_imagen(ruta, gris=False):  # Recibe una ruta y un indicador de gris; devuelve la matriz decodificada o lanza un error claro.
    contenido = np.fromfile(str(ruta), dtype=np.uint8)  # Lee los bytes con una ruta Unicode de Windows, sin procesar todavía los píxeles.
    imagen = cv2.imdecode(contenido, cv2.IMREAD_GRAYSCALE if gris else cv2.IMREAD_COLOR)  # Recibe buffer y flags; decodifica en gris o en BGR.
    if imagen is None or imagen.size == 0:  # Verifica que los bytes correspondan a una imagen utilizable.
        raise ValueError(f"Imagen inválida: {ruta}")  # No permite interpretar como medición una lectura de archivo fallida.
    return imagen  # Devuelve el arreglo que OpenCV produjo a partir del archivo.
def reconocer_digito(recorte, umbral=0.40):  # Recibe una matriz binaria del dígito y una ocupación mínima entre 0 y 1; devuelve símbolo y diagnóstico.
    if recorte.ndim != 2 or recorte.size == 0 or recorte.dtype != np.uint8:  # Exige una región bidimensional uint8 no vacía, requisito del conteo binario de píxeles.
        raise ValueError("El recorte del dígito debe ser una matriz binaria no vacía.")  # Explica una precondición que no puede sustituirse por una cifra.
    if not 0 < umbral < 1 or np.any((recorte != 0) & (recorte != 255)):  # Comprueba el umbral como proporción y que cada intensidad pertenezca al conjunto {0,255}.
        raise ValueError("Se requieren píxeles 0/255 y un umbral de segmento estrictamente entre 0 y 1.")  # Rechaza máscaras de gris que falsearían el conteo de segmentos encendidos.
    alto, ancho = recorte.shape  # Obtiene H filas y W columnas para normalizar su geometría.
    lienzo = np.zeros((112, 64), dtype=np.uint8)  # Crea una matriz negra de tamaño fijo para comparar las mismas siete zonas en cada cifra.
    estrecho = ancho / alto < 0.30  # La razón W/H identifica candidatos al uno, cuyo rectángulo ajustado no contiene las barras izquierdas.
    ancho_destino = max(16, min(18, round(112 * ancho / alto))) if estrecho else 64  # Un candidato estrecho ocupa al menos las 16 columnas de b,c, evitando rechazar unos de trazo fino; los demás normalizan el ancho a 64.
    normalizado = cv2.resize(recorte, (ancho_destino, 112), interpolation=cv2.INTER_NEAREST)  # Recibe src,dsize=(W,H),interpolation; vecino más cercano conserva exclusivamente 0 y 255.
    lienzo[:, 64 - ancho_destino:64] = normalizado  # Alinea los candidatos estrechos a la derecha para evaluar el patrón b,c del uno sin expandirlo a un ocho.
    ocupaciones = []  # Guarda la fracción de píxeles encendidos de cada segmento para auditar por qué se eligió una cifra.
    for x0, y0, x1, y1 in ZONAS:  # Recorre a,b,c,d,e,f,g en el mismo orden de la tabla de patrones.
        zona = lienzo[round(y0 * 112):round(y1 * 112), round(x0 * 64):round(x1 * 64)]  # Convierte proporciones en índices de píxel y extrae cada rectángulo con límites exclusivos.
        encendidos = cv2.countNonZero(zona)  # countNonZero(src) recibe una matriz de un canal y cuenta los píxeles distintos de cero.
        ocupaciones.append(encendidos / zona.size)  # Calcula r=N_blancos/(ancho_zona*alto_zona); la división produce una ocupación entre 0 y 1.
    patron = tuple(int(valor >= umbral) for valor in ocupaciones)  # Binariza cada ocupación: segmento encendido si r>=umbral; apagado si r<umbral.
    simbolo = PATRONES.get(patron, "?")  # Acepta solo coincidencias exactas de los siete estados; no corrige automáticamente segmentos incompletos.
    interiores = (lienzo[24:44, 20:44], lienzo[68:88, 20:44])  # Examina el interior superior e inferior; los siete segmentos estándar no encienden estas zonas centrales.
    ocupacion_interior = max(cv2.countNonZero(zona) / zona.size for zona in interiores)  # Calcula la mayor fracción de primer plano entre los dos interiores, cada uno con 20*24 píxeles.
    if ocupacion_interior > 0.30:  # Un interior mayormente lleno indica fondo invertido, caracteres fusionados o geometría ajena al modelo de siete segmentos.
        simbolo = "?"  # Evita clasificar un rectángulo sólido de primer plano como un ocho solo porque cubre las siete zonas.
    diagnostico = {"patron": list(patron), "ocupaciones": [round(r, 4) for r in ocupaciones], "estrecho": estrecho, "ocupacion_interior": round(ocupacion_interior, 4)}  # Devuelve mediciones geométricas, sin llamarlas probabilidades.
    return simbolo, diagnostico  # Entrega el símbolo y el motivo verificable de su reconocimiento.
def interpretar_roi(binaria, elementos, umbral=0.40):  # Recibe máscara binaria, cajas del paso 4 y umbral de segmentos; devuelve texto y validación numérica.
    simbolos = []  # Conserva el orden espacial de los símbolos para construir el valor mostrado.
    detalles = []  # Guarda las cajas y estados de los siete segmentos de cada elemento encontrado.
    for elemento in sorted(elementos, key=lambda caja: caja["x"]):  # Ordena por coordenada horizontal para leer de izquierda a derecha.
        x, y, w, h = (int(elemento[clave]) for clave in ("x", "y", "w", "h"))  # Extrae coordenadas locales de la ROI, sin confundirlas con la imagen completa.
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > binaria.shape[1] or y + h > binaria.shape[0]:  # Exige cajas enteramente dentro de la máscara; [x,x+w) y [y,y+h) deben ser válidos.
            raise ValueError("Las cajas del paso 4 no coinciden con la imagen binaria; repite ese paso.")  # Evita que NumPy recorte silenciosamente una caja incorrecta.
        tipo = elemento["tipo"]  # Obtiene la clasificación geométrica previa: dígito, punto, signo o desconocido.
        diagnostico = {}  # Inicializa los datos de segmento solo cuando el elemento es un dígito.
        if tipo == "digito":  # Aplica la lógica de siete segmentos exclusivamente a las cajas de cifras.
            simbolo, diagnostico = reconocer_digito(binaria[y:y + h, x:x + w], umbral)  # Extrae una submatriz HxW y evalúa sus siete regiones encendidas.
        elif tipo == "decimal":  # Interpreta la geometría de punto inferior definida durante la segmentación.
            simbolo = "."  # Conserva el separador como carácter explícito; su posición y unicidad se validan después.
        elif tipo == "signo":  # Interpreta una barra horizontal central detectada como candidato a signo.
            simbolo = "-"  # Conserva la barra como menos; un menos interior hará fallar la validación numérica.
        else:  # Mantiene visible cualquier componente que no pueda identificarse mediante la lógica implementada.
            simbolo = "?"  # No elimina componentes desconocidos para fabricar una lectura numérica aparentemente correcta.
        simbolos.append(simbolo)  # Añade un carácter por elemento, manteniendo los ceros iniciales y finales del display.
        detalles.append({**elemento, "simbolo": simbolo, **diagnostico})  # Asocia símbolo, caja y ocupaciones para revisión visual posterior.
    texto = "".join(simbolos)  # Concatena signos, cifras y decimal sin insertar espacios ni adivinar posiciones decimales.
    estado = "sin_elementos" if not texto else "no_reconocido" if "?" in texto else "formato_invalido"  # Diferencia ausencia, reconocimiento incompleto y formato incorrecto.
    valor = None  # Un resultado inválido permanece nulo y no se registra como cero.
    if re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", texto):  # Acepta un menos opcional al principio, al menos una cifra y un único decimal con cifras después.
        valor = str(Decimal(texto))  # Convierte el texto reconocido a decimal exacto y lo serializa como texto para evitar pérdida de precisión.
        estado = "lectura_valida"  # Indica que el patrón de cada cifra y el formato sintáctico fueron aceptados, sin afirmar exactitud metrológica.
    return {"texto": texto, "valor_decimal": valor, "estado": estado, "elementos": detalles}  # Devuelve tanto la lectura como los componentes necesarios para auditarla.
def guardar_resultados(salida, lecturas):  # Recibe directorio y lecturas con fecha; guarda un JSON de la última captura y agrega filas al historial CSV.
    salida.mkdir(parents=True, exist_ok=True)  # Crea el directorio solicitado si aún no existe.
    temporal = salida / "lecturas.json.tmp"  # Usa un nombre temporal para no dejar el JSON principal truncado ante una interrupción de escritura.
    temporal.write_text(json.dumps({"version": 1, "lecturas": lecturas}, ensure_ascii=False, indent=2), encoding="utf-8")  # Serializa fechas, textos y diagnósticos sin perder tildes.
    temporal.replace(salida / "lecturas.json")  # Sustituye la última instantánea solo después de haber terminado la escritura del archivo temporal.
    ruta_csv = salida / "lecturas.csv"  # Conserva un historial de mediciones entre ejecuciones del paso 5.
    nuevo = not ruta_csv.exists() or ruta_csv.stat().st_size == 0  # Detecta si hace falta escribir la cabecera del archivo tabular.
    with ruta_csv.open("a", newline="", encoding="utf-8") as archivo:  # Abre en modo agregar y evita líneas vacías adicionales en Windows.
        escritor = csv.writer(archivo)  # Crea un escritor que escapa comas y comillas correctamente sin concatenación manual de columnas.
        if nuevo:  # Escribe los nombres de columnas una sola vez al crear el historial.
            escritor.writerow(("fecha", "display", "texto", "valor_decimal", "unidad", "estado"))  # Identifica exactamente la fecha, lectura, unidad declarada y estado de cada registro.
        for lectura in lecturas:  # Agrega una fila por display y por instante de muestreo.
            escritor.writerow((lectura["fecha"], lectura["nombre"], lectura["texto"], lectura["valor_decimal"], lectura["unidad"], lectura["estado"]))  # Los valores nulos quedan vacíos y el estado explica su invalidez.
def opciones():  # Define y devuelve los argumentos de ejecución para fotografía procesada o adquisición viva.
    parser = argparse.ArgumentParser(description="Paso 5: lectura geométrica de displays de siete segmentos con OpenCV.")  # Presenta una ayuda de uso en español mediante --help.
    parser.add_argument("--salida", type=Path, default=BASE / "outputs")  # Selecciona el directorio de manifiestos y resultados, por defecto outputs junto a los scripts.
    parser.add_argument("--sin-ventana", action="store_true")  # Omite las ventanas para pruebas reproducibles o adquisición controlada desde consola.
    parser.add_argument("--vivo", action="store_true")  # Activa la captura periódica con la cámara fija en lugar de leer las máscaras guardadas por el paso 4.
    parser.add_argument("--camara", type=int)  # Permite sustituir el índice guardado en la calibración sin modificar el archivo JSON.
    parser.add_argument("--api", choices=("auto", "dshow", "msmf"))  # Permite seleccionar la interfaz de captura o reutilizar la guardada.
    parser.add_argument("--imagen", type=Path)  # Procesa una fotografía nueva con las mismas ROIs, sin abrir ningún dispositivo.
    parser.add_argument("--umbral-segmento", type=float, default=0.40)  # Establece la ocupación mínima r para considerar encendido cada segmento.
    parser.add_argument("--hueco", type=int, default=2)  # Indica cuántas columnas vacías internas tolera la agrupación del paso 4.
    parser.add_argument("--altura-min", type=float, default=0.30)  # Fija la altura mínima del dígito como fracción de la altura de la ROI.
    parser.add_argument("--area-min", type=int, default=2)  # Fija el mínimo de píxeles blancos por componente para filtrar ruido aislado.
    parser.add_argument("--intervalo", type=float, default=1.0)  # Espacia los registros de cámara en segundos mediante el reloj monótono.
    parser.add_argument("--estabilidad", type=int, default=3)  # Exige varias lecturas idénticas consecutivas antes de aceptar un valor durante la captura viva.
    parser.add_argument("--max-frames", type=int, default=0)  # Limita la captura viva; cero permite ejecución continua hasta Q, Esc o Ctrl+C.
    parser.add_argument("--unidad", action="append", default=[], metavar="NOMBRE=UNIDAD")  # Declara unidades por display, por ejemplo roi_01=g o roi_02=C, sin inferirlas de los píxeles.
    args = parser.parse_args()  # Analiza la línea de comandos con los tipos definidos anteriormente.
    if not 0 < args.umbral_segmento < 1 or not math.isfinite(args.intervalo) or args.intervalo <= 0 or args.max_frames < 0 or args.hueco < 0 or not 0 < args.altura_min <= 1 or args.area_min < 1 or args.estabilidad < 1 or args.camara is not None and args.camara < 0:  # Valida dominios, finitud del intervalo y un índice de cámara no negativo.
        parser.error("Revisa umbral (0,1), intervalo finito>0, max-frames>=0, hueco>=0, altura-min (0,1], area-min>=1, estabilidad>=1 y camara>=0.")  # Explica por qué una configuración carece de significado.
    if args.vivo and args.imagen:  # Evita seleccionar simultáneamente dos fuentes de imagen incompatibles.
        parser.error("Usa --vivo o --imagen, uno por ejecución.")  # Solicita una fuente inequívoca sin intentar abrir la cámara.
    if args.vivo and args.sin_ventana and args.max_frames == 0:  # Impide un bucle desatendido sin límite en las verificaciones automáticas.
        parser.error("Con --vivo --sin-ventana especifica --max-frames mayor que cero.")  # Permite terminar la adquisición de consola de forma predecible.
    return args  # Entrega la configuración validada a main().
def ventana_visible(nombre):  # Recibe el nombre HighGUI y devuelve si la ventana sigue abierta, tolerando su destrucción manual.
    try:  # Algunas interfaces dejan de aceptar consultas inmediatamente después de cerrar la ventana con la X.
        return cv2.getWindowProperty(nombre, cv2.WND_PROP_VISIBLE) >= 1  # Recibe winname y prop_id; WND_PROP_VISIBLE es al menos 1 cuando la ventana está visible.
    except cv2.error:  # Una ventana destruida puede producir un error en lugar de una propiedad de visibilidad cero.
        return False  # Trata la desaparición de la ventana como salida solicitada, conservando las lecturas ya guardadas.
def main():  # Ejecuta la lectura y devuelve un código de estado; no recibe argumentos porque usa opciones().
    args = opciones()  # Obtiene las opciones antes de empezar la adquisición o escribir archivos.
    camara = None  # Permite liberar la captura desde finally aunque falle la inicialización.
    ventanas = False  # Registra si este proceso llegó a crear ventanas y necesita cerrarlas.
    try:  # Agrupa errores de archivos, configuración, cámara y procesamiento en un diagnóstico explícito.
        unidades = {}  # Conserva las unidades proporcionadas por el usuario, sin asignar por defecto una magnitud física.
        for entrada in args.unidad:  # Procesa cada declaración NOMBRE=UNIDAD.
            if "=" not in entrada:  # Comprueba que nombre y unidad se separan mediante el signo igual.
                raise ValueError("Cada --unidad debe tener formato nombre=unidad.")  # Evita registrar unidades bajo un nombre ambiguo.
            nombre, unidad = entrada.split("=", 1)  # Divide en dos partes y conserva signos adicionales si forman parte de la unidad.
            unidades[nombre] = unidad  # Asocia la unidad al identificador de la ROI seleccionado en el paso 3.
        if not args.vivo and not args.imagen:  # Por defecto consume las máscaras y cajas ya verificadas visualmente en el paso 4.
            manifiesto = leer_json(args.salida / "contornos.json")  # Exige que el paso 4 haya generado su manifiesto de detecciones.
            fecha = datetime.now(ZONA_BOLIVIA).isoformat(timespec="milliseconds")  # Fecha el procesamiento offline; no afirma que sea la hora original de adquisición de la foto.
            lecturas = []  # Acumula resultados de todos los displays recortados.
            for roi in manifiesto["rois"]:  # Recorre las regiones procesadas con sus máscaras independientes.
                binaria = cargar_imagen(args.salida / roi["binaria"], gris=True)  # Lee la máscara de un canal guardada en el paso 4.
                if binaria.shape != (roi["h"], roi["w"]):  # Exige correspondencia exacta entre máscara y tamaño declarado.
                    raise ValueError("La máscara no coincide con la ROI del manifiesto; repite el paso 4.")  # No mezcla cajas de una imagen con otra resolución.
                lectura = interpretar_roi(binaria, roi["elementos"], args.umbral_segmento)  # Reconoce cada cifra a partir de las cajas guardadas.
                lecturas.append({"nombre": roi["nombre"], "fecha": fecha, "unidad": unidades.get(roi["nombre"], ""), **lectura})  # Añade identificación y unidad declarada conservando los diagnósticos.
            guardar_resultados(args.salida, lecturas)  # Guarda los datos incluso si algunas ROIs necesitan revisión, con su estado explícito.
            for lectura in lecturas:  # Muestra cada lectura en consola para inspección inmediata.
                print(f"{lectura['nombre']}: {lectura['texto'] or '(vacío)'} {lectura['unidad']} [{lectura['estado']}]")  # Expone la interpretación sin convertir una lectura incompleta en cero.
            return 0 if lecturas and all(l["estado"] == "lectura_valida" for l in lecturas) else 2  # Comunica si todos los displays fueron reconocidos o quedan lecturas por revisar.
        configuracion = leer_json(args.salida / "calibracion.json")  # Carga la misma binarización y cámara utilizadas en el paso 2.
        manifiesto = leer_json(args.salida / "rois.json")  # Carga las coordenadas fijas seleccionadas en el paso 3.
        resolucion = manifiesto["resolucion"]  # Define [ancho,alto] contra los que se validará cada imagen nueva.
        if not manifiesto["rois"]:  # Comprueba que existan displays antes de iniciar una adquisición sin destinos.
            raise ValueError("No hay ROIs: ejecuta primero el paso 3.")  # Evita capturar frames que no pueden producir lecturas.
        if args.vivo:  # Abre la cámara solo si el usuario eligió adquisición viva.
            indice = args.camara if args.camara is not None else configuracion["camara"]["indice"]  # Respeta el índice explícito o el guardado en calibración.
            api = args.api or configuracion["camara"]["api"]  # Recupera la interfaz elegida para ese dispositivo.
            camara = cv2.VideoCapture(indice, {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[api])  # Recibe índice y apiPreference; abre la cámara fija con la interfaz seleccionada.
            if not camara.isOpened():  # isOpened() sin argumentos comprueba la inicialización antes de solicitar resolución o frames.
                raise ValueError(f"No se pudo abrir la cámara {indice}. Revisa permisos, conexión e índice.")  # Informa el dispositivo que falló sin producir registros falsos.
            camara.set(cv2.CAP_PROP_FRAME_WIDTH, resolucion[0])  # set(propId,value) solicita al controlador el ancho de la foto en que se seleccionaron las ROIs.
            camara.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucion[1])  # set(propId,value) solicita el alto; cada frame se comprobará porque el controlador puede rechazarlo.
        contador = 0  # Cuenta frames efectivamente leídos para aplicar el límite de adquisición.
        ultimo_guardado = float("-inf")  # Permite registrar el primer frame sin esperar un intervalo inicial.
        anteriores = {}  # Guarda texto y número de repeticiones por ROI para comprobar estabilidad temporal de la lectura viva.
        ultimo_registro_valido = False  # Registra si la última muestra efectivamente guardada contiene todas las lecturas aceptadas; el CSV conserva incidencias anteriores.
        print("Lectura activa: Q o Esc en la ventana, o Ctrl+C en consola, para finalizar.")  # Explica cómo terminar la adquisición continua.
        while True:  # Repite para cámara viva o ejecuta una única iteración para --imagen.
            if args.vivo:  # Obtiene un nuevo frame del dispositivo únicamente en modo vivo.
                correcta, imagen = camara.read()  # read() sin argumentos devuelve éxito y matriz BGR del siguiente frame.
                if not correcta or imagen is None or imagen.size == 0:  # Comprueba que la lectura produjo píxeles utilizables.
                    raise ValueError("La cámara dejó de entregar frames válidos; adquisición detenida.")  # Detiene el proceso ante desconexión sin repetir el dato anterior como si fuera nuevo.
            else:  # Obtiene una fotografía suministrada para comprobar otra captura con las mismas ROIs.
                imagen = cargar_imagen(args.imagen)  # Decodifica el archivo indicado sin abrir la cámara.
            if [imagen.shape[1], imagen.shape[0]] != resolucion:  # Exige ancho y alto idénticos a los de la selección; no escala coordenadas silenciosamente.
                raise ValueError(f"Resolución {imagen.shape[1]}x{imagen.shape[0]} incompatible con ROIs {resolucion}; repite la selección.")  # Explica que la geometría cambió y necesita selección nuevamente.
            fecha = datetime.now(ZONA_BOLIVIA).isoformat(timespec="milliseconds")  # Fecha el frame procesado con desplazamiento horario explícito UTC-4.
            lecturas = []  # Inicializa las mediciones de este frame para no arrastrar lecturas anteriores.
            anotada = imagen.copy()  # Conserva la imagen original intacta mientras añade cajas y texto de inspección a una copia.
            for roi in manifiesto["rois"]:  # Aplica la cadena de procesamiento independientemente a cada display seleccionado.
                x, y, w, h = (int(roi[c]) for c in ("x", "y", "w", "h"))  # Extrae las coordenadas absolutas de cada display.
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > imagen.shape[1] or y + h > imagen.shape[0]:  # Exige que [x,x+w) y [y,y+h) queden dentro del frame.
                    raise ValueError(f"ROI inválida: {roi['nombre']}; repite el paso 3.")  # Impide procesar un recorte truncado por una configuración errónea.
                binaria = aplicar_filtros(imagen[y:y + h, x:x + w], configuracion["filtros"])["binaria"]  # Recorta la ROI y aplica los filtros exactos calibrados anteriormente.
                elementos = detectar_elementos(binaria, args.hueco, args.altura_min, args.area_min)  # Localiza símbolos mediante la misma agrupación del paso 4.
                lectura = interpretar_roi(binaria, elementos, args.umbral_segmento)  # Convierte los segmentos a texto y valida su formato.
                if args.vivo:  # Comprueba estabilidad solo cuando existen frames consecutivos, sin fingirla en fotografías únicas.
                    previo, repeticiones = anteriores.get(roi["nombre"], (None, 0))  # Recupera el texto anterior y su número de coincidencias consecutivas.
                    firma = lectura["texto"] if lectura["estado"] == "lectura_valida" else None  # Una lectura rechazada interrumpe la secuencia de confirmaciones.
                    repeticiones = repeticiones + 1 if firma is not None and firma == previo else 1 if firma is not None else 0  # Cuenta coincidencias contiguas; un cambio reinicia el conteo.
                    anteriores[roi["nombre"]] = (firma, repeticiones)  # Actualiza la historia temporal independiente de este display.
                    lectura["repeticiones"] = repeticiones  # Expone cuántos frames respaldan el texto sin presentar el conteo como probabilidad.
                    if lectura["estado"] == "lectura_valida" and repeticiones < args.estabilidad:  # Comprueba si faltan confirmaciones para aceptar un número.
                        lectura["estado"], lectura["valor_decimal"] = "inestable", None  # Conserva el texto visible, pero deja el valor nulo hasta reunir las confirmaciones solicitadas.
                lecturas.append({"nombre": roi["nombre"], "fecha": fecha, "unidad": unidades.get(roi["nombre"], ""), **lectura})  # Conserva tanto la lectura como su diagnóstico por display.
                color = (0, 180, 0) if lectura["estado"] == "lectura_valida" else (0, 0, 255)  # Usa verde para patrón aceptado y rojo para una lectura que necesita revisión.
                cv2.rectangle(anotada, (x, y), (x + w - 1, y + h - 1), color, 1)  # Recibe img,pt1,pt2,color,thickness; dibuja el límite inclusivo de la ROI con grosor de un píxel.
                cv2.putText(anotada, f"{roi['nombre']}: {lectura['texto'] or '?'}", (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Recibe imagen,texto,origen,fuente,escala,color,grosor; dibuja el diagnóstico sobre la copia.
            ahora = time.monotonic()  # Obtiene segundos de un reloj que no retrocede por cambios de fecha del sistema.
            if ahora - ultimo_guardado >= args.intervalo or not args.vivo:  # Calcula el tiempo transcurrido y registra solo cuando alcanza el intervalo solicitado.
                guardar_resultados(args.salida, lecturas)  # Agrega lecturas válidas e inválidas junto con su estado para dejar evidencia de fallos de adquisición.
                ultimo_guardado = ahora  # Reinicia la referencia temporal después de guardar este conjunto de lecturas.
                ultimo_registro_valido = all(l["estado"] == "lectura_valida" for l in lecturas)  # Evalúa el registro actual; una fase inicial inestable no invalida para siempre las muestras posteriores.
                for lectura in lecturas:  # Imprime únicamente los frames registrados, evitando saturar la consola a la tasa de cámara.
                    print(f"{fecha} | {lectura['nombre']}: {lectura['texto'] or '(vacío)'} {lectura['unidad']} [{lectura['estado']}]")  # Asocia cada dato con su fecha y estado.
            contador += 1  # Incrementa en uno el número de imágenes adquiridas y procesadas.
            if not args.sin_ventana:  # Mantiene la vista de inspección únicamente en una sesión gráfica solicitada.
                cv2.imshow("Paso 5 - Lecturas", anotada)  # imshow(winname,mat) muestra la copia anotada; el original sigue disponible para procesamiento.
                ventanas = True  # Marca que hay una ventana que debe cerrarse en finally.
                tecla = cv2.waitKey(30)  # waitKey(delay) procesa eventos durante al menos aproximadamente 30 ms y devuelve una tecla o -1.
                if not args.vivo:  # Permite inspeccionar una fotografía fija hasta que el usuario cierre su ventana o pulse una tecla.
                    while tecla == -1 and ventana_visible("Paso 5 - Lecturas"):  # Mantiene activa la vista fija mientras la ventana siga visible y no haya tecla.
                        tecla = cv2.waitKey(50)  # Procesa eventos cada aproximadamente 50 ms sin volver a capturar, procesar ni registrar la fotografía.
                if tecla in (27, ord("q"), ord("Q")) or not ventana_visible("Paso 5 - Lecturas"):  # Reconoce Esc,Q o cierre de la ventana sin convertir su destrucción manual en un error de adquisición.
                    break  # Termina limpiamente la adquisición por solicitud del usuario.
            if not args.vivo or args.max_frames and contador >= args.max_frames:  # Una fotografía se procesa una vez; la cámara respeta el máximo de frames cuando está definido.
                break  # Sale al alcanzar el alcance explícito de esta ejecución.
        return 0 if ultimo_registro_valido else 2  # Indica si la última muestra guardada fue aceptada; las filas previas rechazadas siguen auditables en el CSV.
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:  # Captura fallos de archivos, manifiestos, parámetros o procesamiento sin inventar un resultado.
        print(f"ERROR del paso 5: {error}")  # Presenta el motivo preciso que impidió completar la adquisición o lectura.
        return 1  # Comunica el fallo a consola y a verificaciones automáticas.
    except KeyboardInterrupt:  # Permite terminar la adquisición viva con Ctrl+C incluso en modo sin ventana.
        print("Adquisición interrumpida por el usuario; el historial ya guardado se conserva.")  # Confirma que la salida fue manual.
        return 130  # Identifica una interrupción de teclado mediante un código convencional.
    finally:  # Libera recursos tanto al terminar normalmente como tras un error.
        if camara is not None:  # Comprueba si se llegó a construir una captura de cámara.
            camara.release()  # release() sin argumentos cierra la cámara y permite reutilizarla desde otra aplicación.
        if ventanas:  # Evita invocar HighGUI si toda la ejecución se realizó sin ventanas.
            cv2.destroyAllWindows()  # destroyAllWindows() sin argumentos cierra las ventanas creadas por este proceso.
if __name__ == "__main__":  # Importar reconocer_digito o interpretar_roi no inicia ninguna adquisición ni crea resultados.
    raise SystemExit(main())  # Ejecuta el programa únicamente al iniciar este archivo y comunica su código de salida.
