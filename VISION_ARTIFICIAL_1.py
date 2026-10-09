# -*- coding: utf-8 -*-  # Permite guardar correctamente los comentarios y mensajes en español con tildes.
# PASO 1: abrir una cámara, obtener exactamente un frame y mostrarlo sin aplicar filtros.
# Ejecución desde la raíz del proyecto: python VISION_ARTIFICIAL_1.py
# Dependencia, si falta: python -m pip install opencv-python
# Se necesita una distribución de OpenCV con ventanas gráficas; opencv-python-headless no las incluye.
# Documentación de captura: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
# Documentación de ventanas: https://docs.opencv.org/4.x/d7/dfc/group__highgui.html
# Este paso usa únicamente cv2; las matrices que devuelve OpenCV en Python son arreglos de NumPy.
try:  # Intenta cargar OpenCV antes de acceder a cualquier dispositivo, para detectar una dependencia ausente.
    import cv2  # Importa la única biblioteca usada para adquirir y mostrar la imagen; no utiliza OCR ni modelos externos.
except ImportError as error:  # Captura un fallo de importación y conserva su detalle para facilitar el diagnóstico.
    print("ERROR: no se pudo importar OpenCV (cv2).")  # Identifica claramente el requisito que impide ejecutar el paso 1.
    print("Instala OpenCV en este mismo Python: python -m pip install opencv-python")  # Indica cómo instalar la versión con ventanas.
    print(f"Detalle de la importación: {error}")  # Muestra la causa original, incluida una posible dependencia de OpenCV ausente.
    raise SystemExit(1) from error  # Finaliza con código distinto de cero porque todavía no es posible intentar la captura.
# CONFIGURACIÓN: estos valores pueden editarse sin modificar la lógica de adquisición.
INDICE_CAMARA = 0  # VideoCapture interpreta 0 como la primera cámara; prueba 1 o 2 si el equipo deseado tiene otro índice.
API_CAMARA = cv2.CAP_ANY  # CAP_ANY deja que OpenCV seleccione la interfaz disponible; en Windows también puede probarse cv2.CAP_DSHOW.
NOMBRE_VENTANA = "VISION_ARTIFICIAL_1 - Captura unica"  # Define un identificador común para crear, mostrar y consultar la misma ventana.
# MODELO MATEMÁTICO DE LA IMAGEN: una captura en color suele tener forma (H, W, C).
# H es el número de filas, W el número de columnas y C el número de canales; normalmente C = 3.
# En una captura BGR de 8 bits, frame[y, x, 0], frame[y, x, 1] y frame[y, x, 2] representan azul, verde y rojo.
# Para uint8, cada canal toma un valor entero entre 0 y 255: 2 elevado a 8 produce 256 niveles posibles.
# Los índices de píxel comienzan en cero: 0 <= y < H y 0 <= x < W; la cantidad de píxeles es H * W.
# En este paso se consulta la matriz original: no se binariza, no se desenfoca y no se calculan contornos.
def main():  # Agrupa la ejecución; no recibe argumentos y devuelve 0 si termina correctamente o un código de error.
    camara = None  # Inicializa la referencia para que la limpieza sea segura incluso si VideoCapture falla al construirse.
    ventana_creada = False  # Registra si existe una ventana de este script para ejecutar su limpieza cuando corresponda.
    try:  # Protege la apertura, la captura y la visualización, garantizando que al finalizar se liberen los recursos.
        print(f"OpenCV: {cv2.__version__}")  # Informa qué versión de la biblioteca está ejecutando realmente este intérprete.
        print(f"Intentando abrir la cámara con índice {INDICE_CAMARA}...")  # Hace visible el dispositivo que se está intentando utilizar.
        camara = cv2.VideoCapture(INDICE_CAMARA, API_CAMARA)  # Recibe index (número de cámara) y apiPreference (interfaz); intenta abrir el dispositivo.
        if not camara.isOpened():  # isOpened() no recibe argumentos y devuelve si se inicializó la captura; comprueba la apertura antes de leer.
            print(f"ERROR: no se pudo abrir la cámara con índice {INDICE_CAMARA}.")  # Comunica el fallo sin intentar mostrar una imagen inexistente.
            print("Comprueba la conexión, el índice y los permisos de cámara para aplicaciones de escritorio.")  # Sugiere revisar las causas habituales.
            print("Cierra otras aplicaciones que utilicen la cámara; en Windows puedes probar API_CAMARA = cv2.CAP_DSHOW.")  # Ofrece una alternativa manual de interfaz.
            return 1  # Indica un fallo de apertura; el bloque finally liberará cualquier recurso asociado al intento.
        captura_correcta, frame = camara.read()  # read() no recibe argumentos aquí; adquiere y decodifica UN frame y devuelve (éxito, matriz).
        if not captura_correcta or frame is None or frame.size == 0:  # Comprueba el indicador, la existencia y el número de elementos para descartar una captura vacía.
            print("ERROR: la cámara se abrió, pero no entregó un frame válido.")  # Distingue el fallo de lectura del fallo de apertura.
            print("Revisa la conexión y el controlador de la cámara, y vuelve a ejecutar este paso.")  # Permite repetir la prueba sin ocultar el fallo con otras lecturas.
            return 2  # Indica que la apertura funcionó, pero la única lectura solicitada falló.
        alto, ancho = frame.shape[:2]  # shape contiene (H, W, C); [:2] toma las posiciones 0 y 1, sin incluir la 2, para obtener H y W reales.
        canales = frame.shape[2] if frame.ndim == 3 else 1  # Si hay tres ejes toma C del tercer eje; una matriz bidimensional contiene un único canal.
        cantidad_pixeles = alto * ancho  # Multiplica H filas por W columnas: H * W cuenta posiciones espaciales, sin multiplicar por los canales.
        ancho_reportado = camara.get(cv2.CAP_PROP_FRAME_WIDTH)  # get(propId) recibe el identificador del ancho y consulta los píxeles que informa el controlador.
        alto_reportado = camara.get(cv2.CAP_PROP_FRAME_HEIGHT)  # get(propId) recibe el identificador del alto y consulta las filas que informa el controlador.
        fps_reportados = camara.get(cv2.CAP_PROP_FPS)  # get(propId) recibe el identificador de FPS; consulta la tasa reportada, sin medirla con este único frame.
        backend = camara.getBackendName()  # getBackendName() no recibe argumentos y devuelve el nombre de la interfaz usada, por ejemplo MSMF o DSHOW.
        resolucion_reportada = f"{ancho_reportado:g} x {alto_reportado:g} píxeles" if ancho_reportado > 0 and alto_reportado > 0 else "no disponible"  # Solo describe una resolución reportada si ambas dimensiones son positivas.
        texto_fps = f"{fps_reportados:g} cuadros/s" if fps_reportados > 0 else "no disponible"  # Un valor no positivo no permite inferir una tasa válida; no se presenta como una medición.
        print("CAPTURA CORRECTA: se obtuvo un único frame.")  # Confirma que la adquisición requerida para este paso terminó correctamente.
        print(f"Índice de cámara: {INDICE_CAMARA}")  # Documenta qué dispositivo lógico produjo esta captura.
        print(f"Interfaz de captura: {backend}")  # Informa la interfaz seleccionada para ayudar a reproducir y diagnosticar la adquisición.
        print(f"Dimensiones reales (ancho x alto): {ancho} x {alto} píxeles")  # Presenta W y H en el orden habitual de resolución, usando la matriz recibida.
        print(f"Forma de la matriz (filas, columnas[, canales]): {frame.shape}")  # Muestra la forma original; el eje de canales existe si la imagen tiene tres dimensiones.
        print(f"Número de canales: {canales}")  # Informa cuántos valores de canal se almacenan por posición espacial.
        print(f"Tipo de dato de cada canal: {frame.dtype}")  # Informa la representación numérica real, normalmente uint8, sin asumirla de antemano.
        print(f"Cantidad de píxeles: {cantidad_pixeles}")  # Muestra el resultado H * W; un píxel en color contiene varios canales y sigue siendo una posición.
        print(f"Resolución reportada por el controlador: {resolucion_reportada}")  # Distingue los metadatos del dispositivo de las dimensiones reales del frame.
        print(f"FPS reportados por el controlador: {texto_fps}")  # Describe una propiedad informada; no es una tasa medida ni una especificación máxima de la cámara.
        print("Las propiedades reportadas dependen del controlador; cero puede indicar que una propiedad no está disponible.")  # Explica el límite de estos metadatos de captura.
        camara.release()  # release() no recibe argumentos; cierra la adquisición después de consultar las propiedades y conserva el frame ya copiado en memoria.
        camara = None  # Marca que la cámara ya se liberó, para que finally no necesite volver a cerrarla.
        cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)  # Recibe winname (nombre) y flags (modo); WINDOW_NORMAL permite ajustar el tamaño de la ventana.
        ventana_creada = True  # Confirma que la creación tuvo éxito y habilita el cierre de ventanas al terminar.
        cv2.imshow(NOMBRE_VENTANA, frame)  # Recibe winname (ventana de destino) y mat (matriz de imagen); muestra el frame sin alterar sus valores.
        print("Pulsa cualquier tecla con la ventana enfocada, o ciérrala con la X, para terminar.")  # Explica las dos formas de salir mientras la fotografía permanece visible.
        while True:  # Mantiene activa la interfaz de la imagen fija; este bucle no contiene read() y no captura más frames.
            tecla = cv2.waitKey(50)  # Recibe delay=50 milisegundos; procesa eventos y devuelve un código de tecla, o -1 si no hubo pulsación.
            if tecla != -1:  # Cualquier código distinto de -1 indica que se recibió una tecla; no se aplica ninguna operación de máscara binaria.
                break  # Sale del bucle de eventos y permite ejecutar la limpieza de la ventana.
            if cv2.getWindowProperty(NOMBRE_VENTANA, cv2.WND_PROP_VISIBLE) < 1:  # Recibe winname y prop_id; un valor inferior a 1 indica que la ventana dejó de estar visible.
                break  # Termina también cuando el usuario cierra la ventana con la X.
        print("Paso 1 finalizado.")  # Confirma una finalización normal después de mostrar la captura.
        return 0  # Devuelve éxito porque se abrió la cámara, se capturó un frame y se mostró la imagen.
    except cv2.error as error:  # Captura errores de OpenCV durante la adquisición o la creación y gestión de las ventanas.
        print("ERROR de OpenCV al capturar o mostrar la imagen.")  # Describe claramente qué parte del paso falló.
        print(f"Detalle de OpenCV: {error}")  # Conserva el mensaje técnico para diferenciar problemas de dispositivo, controlador o interfaz gráfica.
        print("Si el error menciona imshow o namedWindow, verifica que OpenCV tenga soporte gráfico y ejecuta el script en una sesión de escritorio.")  # Ayuda a identificar una instalación sin ventanas.
        return 3  # Señala un fallo de OpenCV; finally se ejecuta igualmente para limpiar los recursos.
    except KeyboardInterrupt:  # Captura Ctrl+C en la consola para permitir que el usuario interrumpa la espera y libere los recursos.
        print("Ejecución interrumpida por el usuario.")  # Distingue una interrupción manual de un fallo de la cámara.
        return 130  # Utiliza un código convencional para indicar una interrupción de teclado.
    finally:  # Se ejecuta tanto al salir normalmente como después de los errores o retornos anteriores.
        if camara is not None:  # Comprueba si todavía existe una captura que no se liberó antes de mostrar la imagen.
            camara.release()  # release() no recibe argumentos; cierra la cámara incluso si falló la apertura o la lectura.
        if ventana_creada:  # Evita llamar a la limpieza gráfica si nunca se logró crear una ventana.
            try:  # Protege el cierre si la ventana ya se cerró manualmente o la interfaz gráfica falló.
                cv2.destroyAllWindows()  # No recibe argumentos; destruye las ventanas HighGUI de este proceso y libera sus recursos.
            except cv2.error:  # Si la limpieza gráfica falla, conserva el resultado o error principal que ya se comunicó.
                pass  # No reemplaza el diagnóstico de captura o visualización por un segundo error de limpieza.
if __name__ == "__main__":  # Ejecuta main() solo al iniciar este archivo directamente; importarlo no abre automáticamente la cámara.
    raise SystemExit(main())  # Llama a main() sin argumentos y comunica su código de salida al sistema operativo.
