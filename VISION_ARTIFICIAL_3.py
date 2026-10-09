#!/usr/bin/env python3  # Permite ejecutar este archivo con Python 3 también en sistemas que reconocen la línea shebang.
# PASO 3: fotografía de referencia, selección manual de displays y guardado de sus regiones de interés.
# Ejemplo interactivo: python VISION_ARTIFICIAL_3.py --camara 0
# Ejemplo sin cámara ni ventanas: python VISION_ARTIFICIAL_3.py --imagen foto.png --sin-ventana --roi roi_01:100,80,240,60
# Cada ROI debe contener preferentemente una única fila del display; excluir unidades y rótulos facilita los pasos siguientes.
# Las coordenadas se expresan en píxeles: x crece hacia la derecha e y crece hacia abajo desde la esquina superior izquierda.
# Las imágenes se representan mediante matrices NumPy; todas las operaciones de visión artificial usan exclusivamente OpenCV.

import argparse  # Interpreta opciones de consola de forma reproducible y proporciona ayuda con --help.
import json  # Serializa las coordenadas en un manifiesto legible por los pasos siguientes.
import re  # Valida los nombres mediante una expresión regular para impedir rutas o caracteres inseguros.
import sys  # Permite emitir errores en stderr y comunicar un código de salida al sistema operativo.
from pathlib import Path  # Construye rutas multiplataforma sin concatenar separadores de Windows manualmente.

try:  # Detecta al iniciar si faltan las únicas dependencias externas utilizadas por el procesamiento de imágenes.
    import cv2  # OpenCV realiza la captura, decodificación, selección de rectángulos y visualización.
    import numpy as np  # NumPy representa matrices de píxeles y permite leer/escribir archivos con rutas Unicode.
except ImportError as error:  # Captura exclusivamente la ausencia de una dependencia, sin ocultar otros errores de programación.
    print(f"ERROR: falta una dependencia ({error}). Instale con: python -m pip install opencv-python numpy", file=sys.stderr)  # Explica el problema y el comando de instalación sin instalar automáticamente paquetes.
    raise SystemExit(1)  # Un código distinto de cero indica que este paso no pudo completarse.


def cargar_imagen(ruta):  # Recibe una ruta y devuelve una matriz BGR; devuelve errores claros si el archivo no es válido.
    ruta = Path(ruta).expanduser().resolve()  # Normaliza la ruta para que los mensajes muestren el archivo exacto que se está usando.
    if not ruta.is_file():  # Verifica que existe un archivo regular antes de intentar decodificar sus bytes.
        raise ValueError(f"No existe la imagen de entrada: {ruta}")  # Evita un mensaje ambiguo de captura cuando se proporcionó una fotografía.
    contenido = np.fromfile(str(ruta), dtype=np.uint8)  # Lee bytes de 0 a 255; este acceso soporta caracteres Unicode en las rutas de Windows.
    if contenido.size == 0:  # Una matriz de cero bytes representa un archivo vacío que no puede formar una imagen.
        raise ValueError(f"La imagen de entrada está vacía: {ruta}")  # Informa la causa exacta y conserva el nombre del archivo implicado.
    imagen = cv2.imdecode(contenido, cv2.IMREAD_COLOR)  # imdecode recibe bytes codificados y una bandera; IMREAD_COLOR produce tres canales B,G,R de 8 bits.
    if imagen is None or imagen.size == 0:  # OpenCV devuelve None cuando los bytes no corresponden a una imagen decodificable.
        raise ValueError(f"OpenCV no pudo decodificar la imagen: {ruta}")  # Impide utilizar una matriz inexistente en la selección de regiones.
    return imagen  # Retorna la matriz real de píxeles, cuya forma es (alto, ancho, 3).


def guardar_imagen(ruta, imagen):  # Recibe la ruta PNG de salida y una matriz; utiliza OpenCV para codificar y NumPy para escribirla.
    ruta = Path(ruta)  # Acepta tanto strings como objetos Path y mantiene una única representación de las rutas.
    correcto, codificada = cv2.imencode(".png", imagen)  # imencode recibe extensión y matriz; PNG preserva los valores de los píxeles sin compresión con pérdida.
    if not correcto:  # Comprueba la bandera devuelta por OpenCV antes de escribir bytes incompletos.
        raise OSError(f"OpenCV no pudo codificar la imagen PNG: {ruta}")  # Propaga un error de guardado que main mostrará en consola.
    temporal = ruta.with_name(ruta.stem + ".temporal.png")  # Define un archivo vecino temporal para reducir el riesgo de entregar un PNG parcialmente escrito.
    codificada.tofile(str(temporal))  # Escribe la matriz unidimensional de bytes PNG y admite caracteres Unicode en Windows.
    temporal.replace(ruta)  # Sustituye el destino únicamente una vez que la escritura temporal termina correctamente.


def capturar_fotografia(indice, api="auto"):  # Recibe índice no negativo de cámara y elección del backend; devuelve un único frame válido.
    if indice < 0:  # Los índices de dispositivos de cámara se seleccionan mediante enteros no negativos.
        raise ValueError("El índice de cámara debe ser un entero mayor o igual a cero.")  # Evita que un índice negativo sea interpretado como una fuente diferente.
    alternativas = {"auto": [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY], "dshow": [cv2.CAP_DSHOW], "msmf": [cv2.CAP_MSMF]}  # En Windows se intenta DirectShow, Media Foundation y después la selección automática.
    incidencias = []  # Acumula las causas de fallo para explicar todos los intentos de apertura y lectura.
    for backend in alternativas[api]:  # Recorre las alternativas de forma secuencial y se detiene ante el primer frame válido.
        camara = None  # Permite liberar la cámara de manera segura aunque la construcción del objeto falle.
        try:  # Garantiza que cada intento libere el dispositivo al terminar o al producirse una excepción.
            camara = cv2.VideoCapture(indice, backend)  # VideoCapture recibe el índice del dispositivo y el identificador de la API usada para abrirlo.
            if not camara.isOpened():  # isOpened, sin argumentos, informa si OpenCV consiguió abrir la fuente seleccionada.
                incidencias.append(f"API {backend}: no se pudo abrir el dispositivo")  # Conserva el resultado de este intento en el diagnóstico final.
                continue  # Prueba la siguiente API cuando el modo automático permite una alternativa.
            correcto, imagen = camara.read()  # read, sin argumentos, devuelve una bandera de éxito y la matriz del siguiente frame adquirido.
            if not correcto or imagen is None or imagen.size == 0:  # Exige tanto éxito de lectura como píxeles reales antes de usar la fotografía.
                incidencias.append(f"API {backend}: la cámara abrió pero no entregó una imagen válida")  # Distingue un fallo de adquisición de un fallo de apertura.
                continue  # Intenta la siguiente API en lugar de guardar una fotografía vacía.
            return imagen  # Una vez adquirida la fotografía de referencia no se solicitan más frames de esa cámara.
        except cv2.error as error:  # OpenCV puede lanzar una excepción si el backend no está disponible o el dispositivo responde con error.
            incidencias.append(f"API {backend}: {error}")  # Conserva el diagnóstico original para no presentar un éxito inexistente.
        finally:  # Este bloque se ejecuta incluso cuando se retorna una fotografía válida dentro del try.
            if camara is not None:  # Verifica que el objeto se creó antes de intentar liberarlo.
                camara.release()  # release, sin argumentos, cierra el dispositivo y permite que otros pasos o programas utilicen la cámara.
    detalle = "; ".join(incidencias)  # Agrupa las causas de todos los intentos en un único mensaje explicativo.
    raise RuntimeError(f"No se pudo capturar con la cámara {indice}. Revise conexión, permisos e índice. Detalle: {detalle}")  # Detiene el paso sin generar archivos de una captura fallida.


def analizar_roi(texto):  # Convierte una opción nombre:x,y,w,h en un diccionario compatible con el manifiesto del proyecto.
    if ":" not in texto:  # Separa explícitamente el nombre de las cuatro coordenadas para evitar formatos ambiguos.
        raise ValueError(f"ROI inválida '{texto}': use nombre:x,y,w,h.")  # Explica el formato esperado antes de cualquier guardado.
    nombre, coordenadas = texto.split(":", 1)  # Solo el primer separador divide el nombre y los valores; cualquier separador adicional se rechazará al convertir números.
    nombre = nombre.strip()  # Elimina espacios accidentales alrededor del nombre sin permitirlos dentro de él.
    valores = coordenadas.split(",")  # Obtiene los cuatro componentes del rectángulo indicado en píxeles.
    if len(valores) != 4:  # Un rectángulo requiere posición horizontal, posición vertical, ancho y alto, exactamente en ese orden.
        raise ValueError(f"ROI inválida '{texto}': se requieren cuatro enteros x,y,w,h.")  # Rechaza coordenadas incompletas o componentes adicionales.
    try:  # Distingue los errores de formato numérico del resto de comprobaciones geométricas.
        x, y, ancho, alto = (int(valor.strip()) for valor in valores)  # Convierte cada valor a un entero; las coordenadas de píxeles no admiten fracciones.
    except ValueError as error:  # Intercepta valores decimales, texto o componentes vacíos.
        raise ValueError(f"ROI inválida '{texto}': x,y,w,h deben ser enteros.") from error  # Explica el error conservando su causa para depuración.
    return {"nombre": nombre, "x": x, "y": y, "w": ancho, "h": alto, "archivo": f"{nombre}.png"}  # Conserva las claves del contrato JSON y el nombre del recorte correspondiente.


def validar_rois(imagen, rois):  # Comprueba nombres y límites de todos los rectángulos antes de recortar o escribir archivos.
    alto_imagen, ancho_imagen = imagen.shape[:2]  # Las dos primeras dimensiones de una matriz de imagen son número de filas H y columnas W.
    nombres = set()  # Un conjunto permite detectar nombres duplicados antes de que un recorte sobrescriba a otro.
    for roi in rois:  # Valida cada rectángulo contra la misma fotografía de referencia.
        nombre = roi["nombre"]  # Extrae la etiqueta que se utilizará como identificador y como base del archivo PNG.
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", nombre) is None:  # Solo admite 1 a 64 caracteres ASCII seguros; excluye barras, puntos, espacios y rutas relativas.
            raise ValueError(f"Nombre de ROI inseguro '{nombre}': use letras, números, guion o guion bajo; máximo 64 caracteres.")  # Impide escapar de la carpeta de salida mediante el nombre del archivo.
        if nombre.upper() in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?:COM|LPT)[1-9]", nombre.upper()):  # Windows reserva estos nombres de dispositivo incluso cuando llevan una extensión.
            raise ValueError(f"El nombre '{nombre}' está reservado por Windows; elija otro.")  # Evita que una etiqueta válida en apariencia provoque un fallo de escritura.
        nombre_comparable = nombre.casefold()  # Normaliza mayúsculas para la comparación porque Windows normalmente ignora el caso en sus nombres de archivo.
        if nombre_comparable == "captura":  # captura.png pertenece a la fotografía completa y no puede reutilizarse como archivo de una región.
            raise ValueError("El nombre 'captura' está reservado para la fotografía completa; elija otra etiqueta de ROI.")  # Impide que un recorte sobrescriba su propia imagen de referencia.
        if nombre_comparable in nombres:  # Dos etiquetas que difieren solo en mayúsculas también producirían una colisión de archivos en Windows.
            raise ValueError(f"El nombre de ROI '{nombre}' está repetido.")  # Exige identificadores únicos para conservar todos los recortes.
        nombres.add(nombre_comparable)  # Registra la etiqueta actual una vez comprobada su unicidad.
        x, y, ancho, alto = roi["x"], roi["y"], roi["w"], roi["h"]  # Descompone la geometría almacenada en el manifiesto.
        if any(isinstance(valor, bool) or not isinstance(valor, int) for valor in (x, y, ancho, alto)):  # Exige enteros reales y excluye booleanos, aunque Python considere bool una subclase de int.
            raise ValueError(f"La ROI '{nombre}' debe tener coordenadas enteras.")  # Evita índices de matriz ambiguos o no válidos.
        if x < 0 or y < 0 or ancho <= 0 or alto <= 0:  # La esquina superior izquierda debe ser no negativa y ambos tamaños deben ser estrictamente positivos.
            raise ValueError(f"ROI '{nombre}' inválida: x,y deben ser >= 0 y w,h deben ser > 0.")  # Rechaza regiones vacías y coordenadas que NumPy interpretaría desde el final de la matriz.
        if x + ancho > ancho_imagen or y + alto > alto_imagen:  # Un ROI ocupa columnas [x,x+w) y filas [y,y+h), por lo que sus límites superiores pueden coincidir con W,H pero no superarlos.
            raise ValueError(f"ROI '{nombre}' fuera de la imagen {ancho_imagen}x{alto_imagen}: ({x}, {y}, {ancho}, {alto}).")  # Evita el truncamiento silencioso que produciría un slicing fuera de límites.
        if roi["archivo"] != f"{nombre}.png":  # El nombre del PNG debe derivarse exactamente de la etiqueta segura para impedir rutas arbitrarias en el manifiesto.
            raise ValueError(f"El archivo de la ROI '{nombre}' no coincide con su nombre.")  # Garantiza que los pasos siguientes encuentren el recorte descrito por el JSON.


def recortar_rois(imagen, rois):  # Devuelve un diccionario de matrices recortadas sin modificar la fotografía original.
    validar_rois(imagen, rois)  # Comprueba toda la geometría antes de efectuar cualquier selección de elementos de la matriz.
    recortes = {}  # Asocia cada etiqueta única con su matriz BGR independiente.
    for roi in rois:  # Realiza una extracción por display seleccionado.
        x, y, ancho, alto = roi["x"], roi["y"], roi["w"], roi["h"]  # Reutiliza las cuatro coordenadas validadas del rectángulo.
        recortes[roi["nombre"]] = imagen[y:y + alto, x:x + ancho].copy()  # El slicing selecciona h filas y w columnas: R[j,i,c]=I[y+j,x+i,c] para 0<=j<h y 0<=i<w; copy independiza los datos.
    return recortes  # Entrega las matrices con forma (h,w,3) para visualización y guardado.


def seleccionar_rois(imagen):  # Permite añadir rectángulos uno por uno hasta que el usuario cancela una selección.
    rois = []  # Empieza con una lista vacía; cancelar la primera selección no sobrescribe ningún manifiesto existente.
    ventana = "Paso 3 - Seleccione una ROI; Enter confirma, Esc termina"  # Define una única ventana para seleccionar cada nuevo display.
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)  # namedWindow recibe nombre y modo; WINDOW_NORMAL permite ajustar la ventana a pantallas pequeñas sin modificar coordenadas de la imagen original.
    print("Seleccione cada display arrastrando el ratón. Enter o Espacio confirma; Esc o cancelar termina.")  # Explica los controles antes de entrar en la interfaz bloqueante de OpenCV.
    print("Para varios campos, seleccione una ROI por fila de números. Mantenga fuera unidades y rótulos si es posible.")  # Recomienda una geometría útil para la posterior detección y lectura de dígitos.
    while True:  # Repite la selección porque el número de displays o campos depende de los instrumentos presentes.
        rectangulo = cv2.selectROI(ventana, imagen, showCrosshair=True, fromCenter=False)  # selectROI recibe nombre e imagen; showCrosshair dibuja guías y fromCenter=False usa una esquina como inicio del arrastre.
        x, y, ancho, alto = (int(valor) for valor in rectangulo)  # OpenCV devuelve (x,y,w,h); la conversión garantiza tipos int serializables en JSON.
        if ancho == 0 or alto == 0:  # Al cancelar selectROI devuelve una región de área cero; área A=w*h y A=0 representa ausencia de selección.
            break  # Finaliza la recopilación conservando las regiones que ya se confirmaron.
        nombre = f"roi_{len(rois) + 1:02d}"  # Numera automáticamente las regiones como roi_01, roi_02, etc., sin pedir nombres adicionales.
        roi = {"nombre": nombre, "x": x, "y": y, "w": ancho, "h": alto, "archivo": f"{nombre}.png"}  # Forma una entrada conforme al contrato del manifiesto.
        validar_rois(imagen, rois + [roi])  # Verifica también la nueva selección antes de aceptarla, sin alterar la lista anterior si falla.
        rois.append(roi)  # Añade únicamente rectángulos válidos y confirmados por el usuario.
        print(f"Confirmada {nombre}: x={x}, y={y}, w={ancho}, h={alto} píxeles.")  # Presenta los valores exactos que se guardarán para el procesamiento posterior.
    cv2.destroyWindow(ventana)  # destroyWindow recibe el nombre de una ventana y cierra únicamente la interfaz de selección al terminar.
    return rois  # Devuelve todas las selecciones confirmadas; una lista vacía indica cancelación total.


def mostrar_recortes(recortes):  # Presenta cada recorte en una ventana independiente y espera una tecla o el cierre de una ventana.
    nombres_ventanas = []  # Conserva los identificadores necesarios para comprobar si el usuario cerró las ventanas con la X.
    for nombre, recorte in recortes.items():  # Recorre la colección que ya fue validada y guardada.
        ventana = f"Recorte - {nombre}"  # Identifica visualmente cada display con el mismo nombre utilizado en el manifiesto.
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)  # namedWindow recibe nombre y bandera; WINDOW_NORMAL permite redimensionar la ventana sin cambiar los píxeles guardados.
        cv2.imshow(ventana, recorte)  # imshow recibe nombre de ventana y matriz; OpenCV presenta los tres canales en su orden BGR habitual.
        nombres_ventanas.append(ventana)  # Registra cada ventana creada para poder detectar su cierre.
    print("Recortes guardados. Pulse cualquier tecla en una ventana de OpenCV o cierre una ventana para finalizar.")  # Explica cómo salir del modo de revisión visual.
    while True:  # Mantiene vivas las ventanas mientras OpenCV procesa los eventos del teclado y del sistema.
        tecla = cv2.waitKey(30)  # waitKey recibe una espera de 30 ms, atiende eventos de HighGUI y devuelve -1 cuando no se pulsó una tecla.
        if tecla != -1:  # Cualquier código de tecla distinto de -1 representa una solicitud de terminar la revisión.
            break  # Cierra la visualización sin modificar los archivos que ya se guardaron.
        cerrada = False  # Acumula si alguna ventana fue cerrada por el usuario con la X.
        for ventana in nombres_ventanas:  # Consulta cada ventana, ya que no existe una única ventana principal en esta revisión.
            try:  # Algunos backends lanzan un error en lugar de devolver una visibilidad negativa para ventanas destruidas.
                if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:  # getWindowProperty recibe nombre y propiedad; WND_PROP_VISIBLE informa si la ventana sigue visible.
                    cerrada = True  # Registra el cierre para salir del bucle de espera.
                    break  # Evita consultar el resto de las ventanas una vez recibida la señal de salida.
            except cv2.error:  # Considera una ventana desaparecida como una solicitud de cierre, sin convertirla en un fallo del guardado.
                cerrada = True  # Conserva una terminación normal cuando el backend ya eliminó el identificador de la ventana.
                break  # Termina la comprobación de visibilidad de este ciclo.
        if cerrada:  # Comprueba la señal acumulada después de revisar las ventanas.
            break  # Finaliza la revisión cuando se cierra cualquier ventana de recorte.


def crear_argumentos():  # Define la interfaz de consola para cámara, imagen guardada y funcionamiento sin interfaz gráfica.
    parser = argparse.ArgumentParser(description="Paso 3: fotografiar, seleccionar displays y guardar ROIs para OpenCV.")  # ArgumentParser recibe una descripción y genera automáticamente --help.
    parser.add_argument("--camara", type=int, default=0, help="Índice de cámara; por defecto 0. Se ignora cuando se usa --imagen.")  # Un índice entero selecciona el dispositivo usado para la fotografía.
    parser.add_argument("--api", choices=("auto", "dshow", "msmf"), default="auto", help="API de captura en Windows; auto prueba alternativas.")  # Limita la API a opciones implementadas para evitar nombres no reconocidos.
    parser.add_argument("--imagen", type=Path, help="Fotografía existente: usa este archivo sin abrir ninguna cámara.")  # Proporciona una fuente reproducible para calibración y pruebas sin equipo conectado.
    parser.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent / "outputs", help="Directorio de captura.png, rois.json y recortes; por defecto outputs junto al script.")  # Mantiene todos los entregables en una carpeta estable independiente del directorio desde el que se ejecute.
    parser.add_argument("--sin-ventana", action="store_true", help="Desactiva toda interfaz gráfica; requiere al menos una opción --roi.")  # Permite ejecutar pruebas y procesamiento en equipos sin escritorio disponible.
    parser.add_argument("--roi", action="append", default=[], metavar="NOMBRE:X,Y,W,H", help="ROI explícita; repetir para varios displays. Si existe, sustituye la selección con ratón.")  # append acumula las regiones proporcionadas en lugar de sobrescribir la anterior.
    return parser  # Devuelve el parser para que main interprete los argumentos de una ejecución real o de pruebas.


def main(argv=None):  # Recibe opcionalmente una lista de argumentos; devuelve 0 al completar o cancelar, y 1 ante un fallo.
    parser = crear_argumentos()  # Centraliza la definición de opciones de consola.
    args = parser.parse_args(argv)  # parse_args recibe la lista opcional; None significa leer las opciones desde sys.argv.
    if args.sin_ventana and not args.roi:  # Sin ratón ni rectángulos explícitos no existe una forma de seleccionar displays.
        parser.error("--sin-ventana requiere al menos una --roi NOMBRE:X,Y,W,H.")  # argparse explica el uso incorrecto y termina con código 2 antes de abrir cámara o escribir archivos.
    try:  # Agrupa la captura, selección, validación y persistencia para proporcionar diagnósticos claros.
        rois = [analizar_roi(texto) for texto in args.roi]  # Analiza primero los rectángulos explícitos para detectar errores de formato antes de acceder al equipo.
        if args.imagen is not None:  # La fotografía guardada tiene prioridad absoluta frente a cualquier opción de cámara.
            imagen = cargar_imagen(args.imagen)  # Decodifica la fuente sin llamar a VideoCapture ni acceder al dispositivo físico.
            print(f"Fuente: fotografía {args.imagen.expanduser().resolve()}")  # Indica qué archivo produjo los recortes de esta ejecución.
        else:  # Solo utiliza la cámara cuando no se proporcionó una imagen de entrada.
            imagen = capturar_fotografia(args.camara, args.api)  # Adquiere una fotografía y libera inmediatamente el dispositivo para los pasos posteriores.
            print(f"Fuente: cámara {args.camara}; API solicitada {args.api}.")  # Registra la configuración utilizada durante la adquisición.
        alto_imagen, ancho_imagen = imagen.shape[:2]  # Obtiene H filas y W columnas de la matriz que realmente se recibió.
        print(f"Fotografía: ancho={ancho_imagen}, alto={alto_imagen} píxeles; canales BGR={imagen.shape[2]}.")  # Informa las dimensiones reales, necesarias para interpretar las coordenadas de cada ROI.
        if not rois:  # Si no se especificaron coordenadas por consola se utiliza la selección manual autorizada por este paso.
            rois = seleccionar_rois(imagen)  # Recopila regiones con selectROI hasta que el usuario cancela.
        if not rois:  # La cancelación de la primera selección significa que no existe ningún display confirmado.
            print("Selección cancelada sin ROIs: no se guardó ni reemplazó ningún archivo.")  # Explica que una cancelación total conserva los resultados existentes.
            return 0  # Cancelar es una decisión normal del usuario, no un error de adquisición.
        recortes = recortar_rois(imagen, rois)  # Valida todos los rectángulos y extrae sus matrices independientes antes de iniciar la escritura.
        salida = args.salida.expanduser().resolve()  # Resuelve el directorio de salida a una ruta absoluta para facilitar su localización.
        salida.mkdir(parents=True, exist_ok=True)  # Crea la carpeta solo después de confirmar al menos una ROI válida; conserva carpetas ya existentes.
        guardar_imagen(salida / "captura.png", imagen)  # Escribe la fotografía de referencia completa para relacionar los recortes con sus coordenadas originales.
        for roi in rois:  # Guarda un PNG por cada región confirmada y validada.
            guardar_imagen(salida / roi["archivo"], recortes[roi["nombre"]])  # Utiliza exclusivamente nombres seguros generados o comprobados anteriormente.
        manifiesto = {"version": 1, "imagen": "captura.png", "resolucion": [ancho_imagen, alto_imagen], "rois": rois}  # Ordena la resolución como [W,H] y almacena coordenadas en la fotografía original.
        temporal_json = salida / "rois.temporal.json"  # Define un archivo temporal distinto del manifiesto definitivo.
        temporal_json.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # Serializa enteros y textos con sangría; UTF-8 conserva los caracteres de los mensajes o futuras etiquetas.
        temporal_json.replace(salida / "rois.json")  # Publica el manifiesto únicamente una vez que todas las imágenes y el JSON temporal se escribieron correctamente.
        print(f"Guardadas {len(rois)} ROIs. Manifiesto: {salida / 'rois.json'}")  # Confirma la cantidad de displays y la ubicación que usarán los pasos 4 y 5.
        for roi in rois:  # Enumera los archivos reales para que el usuario pueda revisar cada recorte.
            print(f"  {roi['nombre']}: {salida / roi['archivo']}; (x,y,w,h)=({roi['x']},{roi['y']},{roi['w']},{roi['h']})")  # Muestra la relación entre etiqueta, imagen y coordenadas originales.
        if not args.sin_ventana:  # La revisión visual se omite completamente en modo de pruebas sin escritorio.
            mostrar_recortes(recortes)  # Abre las ventanas de recortes y procesa eventos hasta que el usuario termina la revisión.
        return 0  # Comunica éxito al sistema operativo después del guardado y de la revisión opcional.
    except KeyboardInterrupt:  # Ctrl+C solicita terminar de forma controlada también durante la captura o la selección con el ratón.
        print("Paso 3 interrumpido por el usuario.", file=sys.stderr)  # Comunica la interrupción sin presentar una traza de error como si fuese un fallo de OpenCV.
        return 130  # El código 130 identifica una interrupción por teclado; finally libera la interfaz gráfica antes de salir.
    except (ValueError, OSError, RuntimeError, cv2.error) as error:  # Atiende fallos de parámetros, archivos, cámara o GUI sin ocultar la causa original.
        print(f"ERROR en el paso 3: {error}", file=sys.stderr)  # Explica el problema en la consola para que pueda corregirse antes de repetir la ejecución.
        return 1  # Comunica que no se completó normalmente este paso.
    finally:  # Garantiza que la interfaz gráfica se cierre también cuando una selección o visualización produce un error.
        if not args.sin_ventana:  # El modo sin ventanas no ejecuta ninguna llamada HighGUI, compatible con entornos headless.
            try:  # Algunas compilaciones de OpenCV no incluyen soporte de ventanas; la limpieza no debe ocultar el error original.
                cv2.destroyAllWindows()  # destroyAllWindows, sin argumentos, cierra todas las ventanas de OpenCV creadas por este proceso.
            except cv2.error:  # Ignora únicamente un fallo de limpieza de la interfaz ya terminada.
                pass  # Conserva el código de salida y el diagnóstico emitido por la operación principal.


if __name__ == "__main__":  # Ejecuta el programa solo al invocarlo directamente y permite importar las funciones en verificaciones automatizadas.
    raise SystemExit(main())  # main produce el código de salida; SystemExit lo comunica a PowerShell o al proceso que llamó al script.
