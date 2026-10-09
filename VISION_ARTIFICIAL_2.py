# -*- coding: utf-8 -*-  # Conserva las tildes de los comentarios y mensajes en español.
# PASO 2: calibración de filtros en cámara en vivo o en una imagen fija.
# Ejecutar: python VISION_ARTIFICIAL_2.py
# Prueba sin ventanas: python VISION_ARTIFICIAL_2.py --imagen foto.png --sin-ventana
# Guardar con S; terminar con Q, Escape o cerrando una ventana.
# Los siguientes pasos importan aplicar_filtros sin abrir una cámara ni crear ventanas.
# Dependencias de imagen: únicamente OpenCV y NumPy; argparse, json y pathlib son biblioteca estándar.
import argparse  # Analiza los argumentos de consola sin modificar los píxeles ni utilizar bibliotecas de reconocimiento.
import json  # Serializa los parámetros calibrados para que los pasos posteriores apliquen exactamente los mismos filtros.
from pathlib import Path  # Construye rutas respecto de este archivo para que el resultado no dependa del directorio de la terminal.
try:  # Detecta las dependencias antes de intentar crear una ventana o acceder a una cámara.
    import cv2  # Proporciona adquisición, conversión de color, filtros, binarización, bordes y controles gráficos.
    import numpy as np  # Proporciona las matrices y la lectura binaria necesaria para rutas con caracteres españoles.
except ImportError as error:  # Conserva el detalle del paquete que falta para informar un error útil.
    raise SystemExit(f"ERROR: faltan OpenCV/NumPy. Instala: python -m pip install opencv-python numpy. Detalle: {error}") from error  # Detiene el script porque no puede procesar imágenes sin estas bibliotecas.

DIRECTORIO_PROYECTO = Path(__file__).resolve().parent  # Obtiene la carpeta raíz del proyecto a partir de la ubicación física del script.
PARAMETROS_DEFECTO = {"grises": 1, "umbral_activo": 1, "umbral": 127, "invertir": 1, "gauss": 1, "canny_activo": 0, "canny_min": 50, "canny_max": 150, "otsu": 0}  # Usa dígitos oscuros sobre fondo claro como punto inicial; invertir=1 los vuelve blancos al binarizar.
VENTANA_CONTROLES = "VISION_ARTIFICIAL_2 - Filtros (S guardar, Q salir)"  # La ventana de resultado contiene también todas las barras deslizantes.
VENTANA_ORIGINAL = "VISION_ARTIFICIAL_2 - Imagen original"  # Mantiene una referencia visual del contenido capturado antes de filtrar.

def normalizar_parametros(parametros):  # Recibe un diccionario parcial y devuelve valores enteros válidos para todas las operaciones de OpenCV.
    valores = PARAMETROS_DEFECTO.copy()  # Copia el diccionario para no modificar los valores globales cuando se mueve una barra.
    valores.update(parametros or {})  # Incorpora solamente los valores suministrados; un diccionario vacío conserva los valores iniciales.
    for nombre in ("grises", "umbral_activo", "invertir", "canny_activo", "otsu"):  # Recorre los interruptores que matemáticamente representan estados binarios.
        valores[nombre] = int(bool(int(valores[nombre])))  # Convierte cero en 0 y cualquier entero distinto de cero en 1.
    for nombre in ("umbral", "canny_min", "canny_max"):  # Los umbrales de la interfaz se expresan en el intervalo de intensidades de 8 bits.
        valores[nombre] = max(0, min(255, int(valores[nombre])))  # Proyecta cada valor al intervalo [0,255] mediante máximo y mínimo para evitar valores fuera de escala.
    valores["gauss"] = max(0, min(20, int(valores["gauss"])))  # Limita el radio r a [0,20]; el tamaño de núcleo será 2*r+1 y nunca será par.
    valores["canny_min"], valores["canny_max"] = sorted((valores["canny_min"], valores["canny_max"]))  # Ordena T_bajo <= T_alto para conservar la interpretación de la histéresis de Canny.
    return {nombre: valores[nombre] for nombre in PARAMETROS_DEFECTO}  # Devuelve únicamente los campos del contrato, sin claves ajenas ni referencias al diccionario recibido.

def aplicar_filtros(frame, parametros):  # Recibe una matriz uint8 gris/BGR/BGRA y parámetros; devuelve gris, suavizada, binaria, bordes y resultado.
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:  # Rechaza imágenes ausentes o vacías antes de acceder a su forma o llamar a OpenCV.
        raise ValueError("La imagen es inexistente, está vacía o no es una matriz de NumPy.")  # Permite que cada paso informe el fallo de procesamiento de forma clara.
    if frame.dtype != np.uint8:  # Exige intensidades enteras de 8 bits para que los umbrales [0,255] tengan el mismo significado en todos los pasos.
        raise ValueError("La imagen debe tener tipo uint8: intensidades entre 0 y 255.")  # Evita interpretar matrices de 16 bits o flotantes con una escala incorrecta.
    if frame.ndim == 2:  # Una imagen con dos ejes ya es una matriz H por W de intensidades de gris.
        gris = frame.copy()  # Crea una copia para que este procesamiento nunca altere la imagen original.
    elif frame.ndim == 3 and frame.shape[2] == 3:  # Una imagen con tres canales utiliza el orden azul, verde, rojo de OpenCV.
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # cvtColor(src,code): calcula Y≈0.114*B+0.587*G+0.299*R y reduce los tres canales a una intensidad por píxel.
    elif frame.ndim == 3 and frame.shape[2] == 4:  # Las imágenes BGRA contienen además un canal de transparencia que no representa intensidad del display.
        gris = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)  # cvtColor(src,code): transforma BGRA a gris utilizando B,G,R e ignorando alfa.
    elif frame.ndim == 3 and frame.shape[2] == 1:  # Algunas fuentes representan el gris como H por W por 1 en lugar de H por W.
        gris = frame[:, :, 0].copy()  # Selecciona el único canal y elimina solamente el eje redundante, sin reducir resolución.
    else:  # Cualquier otra cantidad de ejes o canales no corresponde a los formatos admitidos por esta herramienta.
        raise ValueError("Formato inválido: se requiere gris, BGR o BGRA.")  # Comunica el formato esperado antes de que OpenCV produzca un error menos descriptivo.
    p = normalizar_parametros(parametros)  # Garantiza tipos, rangos y orden de umbrales antes de utilizar la configuración recibida.
    radio = p["gauss"]  # Lee el radio r, que representa cuántos píxeles vecinos se consideran a cada lado del centro.
    tamano = 2 * radio + 1  # Construye un núcleo impar k por k con un píxel central; r=1 produce k=3 y r=2 produce k=5.
    if radio > 0:  # El radio cero significa desactivar el desenfoque, sin realizar una convolución innecesaria.
        suavizada = cv2.GaussianBlur(gris, (tamano, tamano), 0)  # GaussianBlur(src,ksize,sigmaX): convoluciona con pesos G(x,y)∝exp(-(x²+y²)/(2*σ²)); sigmaX=0 calcula σ desde k, sigmaY se deriva automáticamente y los pesos suman 1.
    else:  # La configuración sin Gaussian Blur debe conservar exactamente la matriz de gris original.
        suavizada = gris.copy()  # Copia los píxeles, sin sumar vecinos ni introducir redondeo por convolución.
    tipo_umbral = cv2.THRESH_BINARY_INV if p["invertir"] else cv2.THRESH_BINARY  # BINARY produce 255 si I>T y 0 si I<=T; BINARY_INV intercambia ambas salidas para definir el dígito como blanco.
    if p["otsu"]:  # Otsu sustituye el valor manual T por un umbral elegido a partir del histograma de intensidades.
        tipo_umbral |= cv2.THRESH_OTSU  # Combina banderas; Otsu maximiza w0*w1*(μ0-μ1)², la varianza entre las clases de fondo y primer plano.
    umbral_real, binaria = cv2.threshold(suavizada, p["umbral"], 255, tipo_umbral)  # threshold(src,thresh,maxval,type) devuelve (T empleado,imagen); maxval=255 genera exclusivamente 0/255, incluso si la vista previa de umbral está desactivada.
    bordes = cv2.Canny(suavizada, p["canny_min"], p["canny_max"], apertureSize=3, L2gradient=True)  # Canny(image,T_bajo,T_alto,apertureSize,L2gradient): Sobel de 3x3 calcula Gx,Gy, |G|=sqrt(Gx²+Gy²); conserva bordes fuertes y débiles conectados por histéresis.
    if p["canny_activo"]:  # El interruptor de Canny define la vista final como mapa de bordes para evaluar contraste y ruido.
        resultado = bordes.copy()  # Preserva el mapa calculado y evita que una modificación posterior de la vista afecte a la salida independiente.
    elif p["umbral_activo"]:  # La vista de binarización permite ajustar visualmente el primer plano blanco de los futuros dígitos.
        resultado = binaria.copy()  # Muestra las clases binarias, manteniendo la matriz original disponible para los pasos de contornos.
    elif p["grises"]:  # Sin binarización ni Canny, la barra de gris selecciona la vista monocromática.
        resultado = suavizada.copy()  # Muestra el gris ya suavizado para observar el efecto del radio Gaussian Blur.
    elif radio > 0:  # Sin conversión visible a gris, la barra de Gaussian Blur también debe afectar a la imagen de color.
        resultado = cv2.GaussianBlur(frame, (tamano, tamano), 0)  # GaussianBlur(src,ksize,sigmaX): aplica la misma convolución normalizada a cada canal, con σ automático y núcleo impar.
    else:  # Todos los efectos visibles desactivados deben permitir comparar directamente con la captura original.
        resultado = frame.copy()  # Devuelve una copia del frame original sin efectos de visualización.
    return {"gris": gris, "suavizada": suavizada, "binaria": binaria, "bordes": bordes, "resultado": resultado}  # Expone las cinco matrices; binaria siempre es uint8 de valores 0/255, independientemente de la vista elegida.

def leer_imagen(ruta):  # Recibe una ruta existente y devuelve una imagen BGR uint8 con lectura compatible con caracteres Unicode de Windows.
    archivo = Path(ruta).expanduser().resolve()  # Expande el usuario y obtiene una ruta absoluta para que los errores identifiquen el archivo real.
    if not archivo.is_file():  # Comprueba que la entrada es un archivo y no una carpeta ni una ruta ausente.
        raise ValueError(f"No existe la imagen: {archivo}")  # Informa la ruta concreta que debe corregirse en --imagen.
    datos = np.fromfile(str(archivo), dtype=np.uint8)  # Lee bytes sin convertir los píxeles; permite abrir rutas Unicode sin depender del lector de nombres de OpenCV.
    frame = cv2.imdecode(datos, cv2.IMREAD_COLOR)  # imdecode(buf,flags): decodifica los bytes con IMREAD_COLOR y devuelve BGR de tres canales y 8 bits.
    if frame is None or frame.size == 0:  # Verifica que el contenido es una imagen compatible y que la decodificación tuvo éxito.
        raise ValueError(f"No se pudo decodificar la imagen: {archivo}")  # Distingue un archivo existente de un archivo con formato inválido o dañado.
    return frame  # Entrega la matriz original para filtrarla sin modificar el archivo de entrada.

def guardar_imagen(ruta, matriz):  # Guarda una matriz como PNG usando únicamente la codificación de OpenCV y escritura de bytes de NumPy.
    correcto, datos = cv2.imencode(".png", matriz)  # imencode(ext,img) recibe extensión y matriz y devuelve (éxito,bytes); PNG conserva exactamente los valores binarios sin compresión con pérdida.
    if not correcto:  # Verifica el indicador de codificación antes de escribir datos incompletos.
        raise OSError(f"No se pudo codificar la imagen: {ruta}")  # Propaga el fallo para impedir que se anuncie un guardado exitoso.
    datos.tofile(str(ruta))  # Escribe los bytes PNG en una ruta Unicode sin necesitar ninguna biblioteca adicional de imagen.

def guardar_calibracion(carpeta, frame, parametros, indice, api):  # Guarda parámetros reproducibles y las matrices de diagnóstico; solamente se llama con S o --sin-ventana.
    destino = Path(carpeta).expanduser().resolve()  # Convierte la carpeta de salida en una ruta absoluta para mostrar dónde quedó la calibración.
    destino.mkdir(parents=True, exist_ok=True)  # Crea la carpeta y sus padres si faltan; conserva cualquier archivo existente al crearla.
    p = normalizar_parametros(parametros)  # Guarda los valores realmente aplicados, incluyendo el orden corregido de los umbrales de Canny.
    alto, ancho = frame.shape[:2]  # La matriz usa (H,W,C); se extraen altura y anchura, que el JSON almacenará como [W,H].
    datos = {"version": 1, "camara": {"indice": int(indice), "api": str(api)}, "resolucion": [int(ancho), int(alto)], "filtros": p}  # Define el contrato que compartirán selección de ROI, contornos y lectura de segmentos.
    procesadas = aplicar_filtros(frame, p)  # Recalcula todas las matrices con los parámetros exactos del documento que se guardará.
    guardar_imagen(destino / "calibracion_original.png", frame)  # Guarda la referencia original con la que se evaluaron los parámetros, conservando sus píxeles.
    for nombre, matriz in procesadas.items():  # Recorre las cinco salidas para permitir su revisión independiente incluso sin ventanas.
        guardar_imagen(destino / f"calibracion_{nombre}.png", matriz)  # Genera PNGs de gris, suavizada, binaria, bordes y resultado en la carpeta seleccionada.
    archivo = destino / "calibracion.json"  # Establece el nombre público del archivo que consumirán los demás scripts.
    temporal = destino / "calibracion.json.tmp"  # Usa un archivo temporal en la misma carpeta para no reemplazar el JSON con una escritura incompleta.
    temporal.write_text(json.dumps(datos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # Serializa el esquema legible con indentación y UTF-8; el salto final facilita herramientas de texto.
    temporal.replace(archivo)  # Sustituye el JSON anterior solamente después de escribir el nuevo documento completo.
    print(f"Calibración guardada: {archivo}")  # Confirma la ruta exacta y el éxito del guardado explícito solicitado.
    print("El primer plano de calibracion_binaria.png debe ser blanco para los dígitos y negro para el fondo; ajusta Invertir si es necesario.")  # Explica cómo evaluar la polaridad correcta antes de detectar contornos.
    return archivo  # Devuelve la ruta de salida para posibles verificaciones programáticas sin abrir ventanas.

def al_mover_barra(valor):  # createTrackbar exige un callback que recibe el nuevo entero; la lectura centralizada se realiza en el bucle.
    return None  # No procesa aquí para evitar repetir convoluciones por cada evento de una barra; el bucle leerá todos los valores juntos.

def crear_controles(parametros):  # Crea las ventanas y todas las barras para editar los parámetros sobre la cámara o una imagen fija.
    cv2.namedWindow(VENTANA_CONTROLES, cv2.WINDOW_NORMAL)  # namedWindow(winname,flags): registra la ventana de filtros y permite redimensionarla con WINDOW_NORMAL.
    cv2.namedWindow(VENTANA_ORIGINAL, cv2.WINDOW_NORMAL)  # namedWindow(winname,flags): crea una segunda ventana redimensionable para comparar la imagen sin filtros.
    barras = (("Grises", "grises", 1), ("Threshold activo", "umbral_activo", 1), ("Umbral", "umbral", 255), ("Invertir", "invertir", 1), ("Gauss radio", "gauss", 20), ("Canny activo", "canny_activo", 1), ("Canny minimo", "canny_min", 255), ("Canny maximo", "canny_max", 255), ("Otsu", "otsu", 1))  # Cada tupla relaciona etiqueta visible, campo JSON y máximo permitido por el control.
    for etiqueta, nombre, maximo in barras:  # Recorre las nueve barras sin duplicar la lógica de creación ni sus límites.
        cv2.createTrackbar(etiqueta, VENTANA_CONTROLES, parametros[nombre], maximo, al_mover_barra)  # createTrackbar(trackbarname,winname,value,count,onChange): crea el control con valor inicial, límite máximo inclusivo y callback de cambio.
    return barras  # Devuelve el mapa de controles para recuperar luego los campos del contrato de parámetros.

def leer_controles(barras):  # Recibe el mapa de etiquetas/campos y obtiene la configuración actual de todas las barras.
    parametros = {}  # Inicializa un diccionario independiente para representar un estado completo de la interfaz.
    for etiqueta, nombre, maximo in barras:  # Los mismos nombres usados al crear las barras permiten leer exactamente sus posiciones.
        parametros[nombre] = cv2.getTrackbarPos(etiqueta, VENTANA_CONTROLES)  # getTrackbarPos(trackbarname,winname): devuelve la posición entera actual del control especificado.
    return normalizar_parametros(parametros)  # Ajusta los umbrales de Canny si el usuario movió el mínimo por encima del máximo.

def crear_argumentos():  # Define los argumentos públicos del script y sus valores iniciales para uso interactivo o automatizado.
    parser = argparse.ArgumentParser(description="PASO 2: calibrar filtros OpenCV; S guarda, Q/Escape termina.")  # Construye la ayuda de consola sin ejecutar la adquisición de imágenes.
    parser.add_argument("--camara", type=int, default=0, help="Índice de cámara: 0, 1, 2...")  # Selecciona el dispositivo VideoCapture por índice entero.
    parser.add_argument("--api", choices=("auto", "dshow", "msmf"), default="auto", help="Interfaz de cámara; dshow y msmf son opciones de Windows.")  # Restringe la API a valores traducibles a constantes de OpenCV.
    parser.add_argument("--imagen", type=Path, help="Imagen fija para calibrar sin utilizar cámara.")  # Permite calibración reproducible a partir de un archivo capturado previamente.
    parser.add_argument("--salida", type=Path, default=DIRECTORIO_PROYECTO / "outputs", help="Carpeta de calibracion.json y PNGs de diagnóstico.")  # Guarda por defecto junto al proyecto en outputs, independientemente del directorio de ejecución.
    parser.add_argument("--sin-ventana", action="store_true", help="Procesa una captura o imagen, guarda la calibración y termina sin GUI.")  # Permite pruebas sin pantalla ni controles y evita llamadas a HighGUI en este modo.
    parser.add_argument("--umbral", type=int, choices=range(256), default=127, metavar="0..255", help="Umbral manual de intensidad.")  # Define T para la comparación I>T cuando Otsu está desactivado.
    parser.add_argument("--invertir", type=int, choices=(0, 1), default=1, help="1 para dígitos oscuros sobre fondo claro; 0 para dígitos claros.")  # Establece la polaridad para representar los dígitos como primer plano blanco.
    parser.add_argument("--gauss", type=int, choices=range(21), default=1, metavar="0..20", help="Radio r; 0 desactiva, r>0 usa núcleo 2*r+1.")  # Configura un radio que siempre genera un núcleo impar compatible con GaussianBlur.
    parser.add_argument("--otsu", type=int, choices=(0, 1), default=0, help="1 calcula el umbral automáticamente con Otsu.")  # Permite alternar entre el umbral manual y la maximización de varianza entre clases.
    parser.add_argument("--grises", type=int, choices=(0, 1), default=1, help="Activa la vista de gris cuando threshold y Canny están desactivados.")  # Controla solamente la vista; la matriz gris siempre se calcula para binarizar.
    parser.add_argument("--umbral-activo", type=int, choices=(0, 1), default=1, help="Activa la vista binaria; la salida binaria se calcula siempre.")  # Mantiene disponible el primer plano para los pasos posteriores aunque se vea otra etapa.
    parser.add_argument("--canny-activo", type=int, choices=(0, 1), default=0, help="Activa la vista del mapa de bordes.")  # Selecciona el diagnóstico de bordes sin sustituir la salida de binarización.
    parser.add_argument("--canny-min", type=int, choices=range(256), default=50, metavar="0..255", help="Umbral inferior de histéresis.")  # Define el límite para bordes débiles que solo sobreviven conectados a uno fuerte.
    parser.add_argument("--canny-max", type=int, choices=range(256), default=150, metavar="0..255", help="Umbral superior de histéresis.")  # Define el límite para declarar un borde fuerte durante Canny.
    return parser  # Expone el analizador para que main convierta la línea de comandos en una configuración validada.

def main(argv=None):  # Recibe opcionalmente una lista de argumentos para pruebas; devuelve 0 al terminar correctamente y 1 ante errores.
    argumentos = crear_argumentos().parse_args(argv)  # Analiza argv o sys.argv automáticamente; --help termina sin abrir cámaras ni ventanas.
    parametros = normalizar_parametros({nombre: getattr(argumentos, nombre) for nombre in PARAMETROS_DEFECTO})  # Copia la configuración CLI al mismo contrato empleado por trackbars y JSON.
    camara = None  # Inicializa el recurso de captura para poder liberarlo incluso si ocurre un fallo antes del primer frame.
    ventanas_creadas = False  # Registra si es necesario limpiar ventanas; --sin-ventana mantiene este estado falso.
    try:  # Garantiza la limpieza de la cámara y de las ventanas al salir por teclado, error o interrupción.
        if argumentos.imagen is not None:  # La entrada de archivo debe evitar completamente VideoCapture para permitir pruebas offline.
            frame = leer_imagen(argumentos.imagen)  # Decodifica la fotografía BGR que permanecerá fija mientras se ajustan los controles.
        else:  # Sin imagen de entrada, la fuente prevista es la cámara indicada en la consola.
            apis = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}  # Traduce los nombres públicos a constantes de backend usadas por VideoCapture.
            camara = cv2.VideoCapture(argumentos.camara, apis[argumentos.api])  # VideoCapture(index,apiPreference): intenta abrir el dispositivo elegido con la interfaz solicitada.
            if not camara.isOpened():  # isOpened() no recibe argumentos y comprueba que OpenCV pudo inicializar la captura.
                raise RuntimeError(f"No se pudo abrir la cámara {argumentos.camara} con API {argumentos.api}. Revisa conexión, permisos y otras aplicaciones; prueba --api dshow o --camara 1.")  # Ofrece pasos concretos para diagnosticar el fallo de apertura.
            correcto, frame = camara.read()  # read() no recibe argumentos y devuelve (éxito,frame), adquiriendo y decodificando una captura inicial.
            if not correcto or frame is None or frame.size == 0:  # Rechaza un dispositivo abierto que no entrega una imagen utilizable.
                raise RuntimeError("La cámara se abrió, pero no entregó una imagen válida.")  # Distingue un fallo de adquisición del fallo de apertura.
        alto, ancho = frame.shape[:2]  # Obtiene H y W de la matriz adquirida, en lugar de asumir una resolución solicitada al controlador.
        print(f"OpenCV {cv2.__version__}; captura {ancho} x {alto} píxeles. S: guardar, Q/Escape: salir.")  # Informa el entorno y la resolución real para relacionarla con la calibración guardada.
        if argumentos.sin_ventana:  # El modo automatizado procesa exactamente esta imagen y termina sin mostrar controles.
            guardar_calibracion(argumentos.salida, frame, parametros, argumentos.camara, argumentos.api)  # Guarda explícitamente los parámetros CLI y todas las imágenes de diagnóstico.
            return 0  # Termina correctamente; finally liberará el dispositivo si la fuente fue una cámara.
        ventanas_creadas = True  # Activa limpieza antes de crear las ventanas para cubrir también una creación parcial que falle.
        barras = crear_controles(parametros)  # Crea controles inicializados con los valores CLI, tanto para cámara como para fotografía fija.
        print("Para contornos posteriores, ajusta Threshold, Invertir, Otsu y Gauss hasta separar los dígitos blancos del fondo negro.")  # Describe la finalidad práctica de la calibración del primer plano.
        while True:  # Mantiene una vista viva hasta recibir una orden explícita de terminar o un fallo de adquisición.
            parametros = leer_controles(barras)  # Lee un estado completo de todas las barras en cada iteración para aplicar cambios en tiempo real.
            procesadas = aplicar_filtros(frame, parametros)  # Calcula las etapas independientes y selecciona la vista final según los interruptores.
            cv2.imshow(VENTANA_ORIGINAL, frame)  # imshow(winname,mat): presenta la matriz original en su ventana; los píxeles de frame permanecen intactos.
            cv2.imshow(VENTANA_CONTROLES, procesadas["resultado"])  # imshow(winname,mat): presenta la etapa seleccionada mientras conserva visibles las barras deslizantes.
            tecla = cv2.waitKey(30) & 0xFF  # waitKey(delay) atiende eventos y espera hasta 30 ms; AND con 255 conserva el código ASCII inferior para comparar teclas.
            if tecla in (ord("s"), ord("S")):  # ord transforma el carácter en su código entero y reconoce S en minúscula o mayúscula.
                guardar_calibracion(argumentos.salida, frame, parametros, argumentos.camara, argumentos.api)  # Guarda exactamente el frame mostrado y la configuración visible, solo ante esta orden explícita.
            if tecla in (27, ord("q"), ord("Q")):  # 27 es Escape; Q ofrece una salida accesible desde cualquiera de las ventanas con foco.
                break  # Termina el bucle sin sobrescribir automáticamente una calibración que el usuario no haya guardado.
            if cv2.getWindowProperty(VENTANA_CONTROLES, cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty(VENTANA_ORIGINAL, cv2.WND_PROP_VISIBLE) < 1:  # getWindowProperty(winname,prop_id) consulta visibilidad; cerrar cualquiera de las ventanas finaliza la sesión.
                break  # Evita seguir capturando imágenes después de cerrar la interfaz.
            if camara is not None:  # Una fotografía fija mantiene el mismo frame; solo la fuente de cámara debe adquirir otro.
                correcto, siguiente = camara.read()  # read() devuelve otro frame y su indicador de éxito para la vista viva del siguiente ciclo.
                if not correcto or siguiente is None or siguiente.size == 0:  # Detecta desconexiones o pérdida de adquisición durante la sesión.
                    raise RuntimeError("La cámara dejó de entregar imágenes. Revisa la conexión y vuelve a ejecutar el paso 2.")  # Comunica por qué se detuvo la calibración en vivo.
                frame = siguiente  # Actualiza la matriz de referencia después de haber atendido teclado y guardado del frame anterior.
        return 0  # Señala una salida voluntaria correcta aunque el usuario decida salir sin guardar.
    except (cv2.error, OSError, ValueError, RuntimeError) as error:  # Captura fallos de OpenCV, acceso a archivos, parámetros y dispositivos para evitar una salida silenciosa.
        print(f"ERROR en paso 2: {error}")  # Muestra la causa concreta; un error HighGUI suele indicar una instalación sin soporte gráfico.
        return 1  # Permite a la terminal o a otra herramienta reconocer el fallo mediante un código de salida distinto de cero.
    except KeyboardInterrupt:  # Interpreta Ctrl+C como una salida solicitada durante una sesión larga.
        print("Calibración interrumpida por el usuario.")  # Avisa que el cierre responde al teclado y no a una desconexión del dispositivo.
        return 130  # Devuelve el código habitual asociado a una interrupción de teclado.
    finally:  # Ejecuta la liberación de recursos para todos los caminos, incluyendo return y excepciones.
        if camara is not None:  # Solo existe una captura que liberar si la fuente elegida fue un dispositivo.
            camara.release()  # release() no recibe argumentos y cierra el dispositivo para permitir su uso por los siguientes pasos.
        if ventanas_creadas:  # El modo sin ventanas evita por completo las llamadas de limpieza de HighGUI.
            try:  # Una instalación sin soporte gráfico puede fallar al crear y también al destruir ventanas.
                cv2.destroyAllWindows()  # destroyAllWindows() no recibe argumentos y cierra las ventanas de OpenCV de este proceso.
            except cv2.error:  # Evita que un error secundario de limpieza oculte el mensaje de error principal.
                pass  # La cámara ya fue liberada; no hay otra operación gráfica útil que intentar en este caso.

if __name__ == "__main__":  # Esta guarda impide ejecutar la cámara al importar aplicar_filtros desde los pasos siguientes.
    raise SystemExit(main())  # Ejecuta el script únicamente desde consola y transmite a la terminal el código de éxito, fallo o interrupción.
