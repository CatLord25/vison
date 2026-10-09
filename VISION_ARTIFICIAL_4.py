# -*- coding: utf-8 -*-  # Guarda correctamente las explicaciones y los mensajes en español.
# PASO 4: localizar símbolos mediante contornos y proyección de columnas; todavía no asigna valores a las cifras.
# Ejecutar después de guardar calibración y ROIs: python VISION_ARTIFICIAL_4.py
# Se presupone una ROI por fila de cifras, con segmentos blancos sobre fondo negro y unidades fuera del recorte.
import argparse  # Define opciones para los umbrales geométricos, archivos de entrada y modo sin ventana.
import json  # Lee calibración y regiones, y guarda las cajas que consumirá el lector del paso 5.
from pathlib import Path  # Mantiene las rutas predeterminadas junto al proyecto, aunque la consola tenga otro directorio.
try:  # Verifica las dependencias de imagen antes de intentar procesar archivos.
    import cv2  # Usa exclusivamente OpenCV para contornos, máscaras y anotaciones de imagen.
    import numpy as np  # Usa NumPy como apoyo para proyecciones y arreglos de índices.
except ImportError as error:  # Explica por qué no puede iniciarse el procesamiento si falta un paquete.
    raise SystemExit(f"ERROR: instala opencv-python y numpy en este Python. Detalle: {error}")  # Termina sin generar detecciones ficticias.
from VISION_ARTIFICIAL_2 import aplicar_filtros  # Reutiliza los filtros calibrados, evitando implementar otra binarización con parámetros diferentes.
from VISION_ARTIFICIAL_3 import cargar_imagen, guardar_imagen, validar_rois  # Reutiliza lectura PNG Unicode, escritura y validación geométrica del paso 3.
BASE = Path(__file__).resolve().parent  # Obtiene la raíz física donde están los cinco scripts.
def intervalos_activos(proyeccion, hueco):  # Recibe un vector de conteos y un máximo de posiciones vacías; devuelve intervalos [inicio,fin) agrupados.
    posiciones = np.flatnonzero(proyeccion > 0)  # Obtiene los índices cuya suma de píxeles blancos es positiva, sin incluir columnas o filas vacías.
    if posiciones.size == 0:  # Una proyección completamente nula no contiene símbolos.
        return []  # Devuelve ausencia explícita y no una caja del tamaño de la imagen.
    intervalos = []  # Acumula intervalos con límites superiores exclusivos para utilizarlos directamente en slicing.
    inicio = anterior = int(posiciones[0])  # Inicializa el primer intervalo en la primera posición con primer plano.
    for posicion in posiciones[1:]:  # Recorre las posiciones activas restantes en orden creciente.
        actual = int(posicion)  # Convierte el escalar NumPy en entero Python para serialización JSON posterior.
        if actual - anterior - 1 > hueco:  # Calcula la cantidad de posiciones vacías intermedias; separa grupos cuando supera el máximo permitido.
            intervalos.append((inicio, anterior + 1))  # Cierra [inicio,anterior+1), incluyendo la última posición activa sin añadir un píxel extra al recorte.
            inicio = actual  # Comienza el siguiente grupo en la posición actual.
        anterior = actual  # Actualiza el extremo activo independientemente de si se dividió el grupo.
    intervalos.append((inicio, anterior + 1))  # Cierra también el último intervalo, que no tiene una posición posterior para disparar la separación.
    return intervalos  # Devuelve límites independientes de la orientación de la proyección.
def detectar_elementos(binaria, hueco=2, altura_min=0.30, area_min=2):  # Recibe máscara 0/255 y parámetros geométricos; devuelve cajas locales con tipo, sin reconocer cifras.
    if not isinstance(binaria, np.ndarray) or binaria.ndim != 2 or binaria.dtype != np.uint8 or binaria.size == 0:  # Exige HxW uint8 no vacío, formato requerido por findContours.
        raise ValueError("La máscara debe ser una matriz bidimensional uint8 no vacía.")  # Describe el contrato de imagen de esta función pública.
    if np.any((binaria != 0) & (binaria != 255)):  # Comprueba que las intensidades pertenezcan al conjunto binario {0,255}.
        raise ValueError("La máscara debe contener exclusivamente 0 y 255.")  # Evita interpretar sombras de gris como componentes encendidos.
    if hueco < 0 or area_min < 1 or not 0 < altura_min <= 1:  # Valida un hueco no negativo, un área mínima positiva y una proporción de altura válida.
        raise ValueError("Se requieren hueco>=0, area_min>=1 y altura_min en (0,1].")  # Explica los dominios matemáticos de las tres opciones.
    alto_roi, ancho_roi = binaria.shape  # Define H y W para validar alturas y convertir coordenadas locales.
    contornos, jerarquia = cv2.findContours(binaria.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Recibe imagen,modo,método; extrae solo contornos exteriores y comprime puntos colineales sin cambiar sus extremos.
    mascara = np.zeros_like(binaria)  # Crea una máscara negra del mismo tamaño donde se conservarán componentes suficientemente grandes.
    componentes = []  # Conserva cajas de componentes físicos, que pueden ser barras separadas de un mismo dígito.
    for contorno in contornos:  # Examina cada contorno exterior sin considerar los huecos interiores de una cifra como símbolos nuevos.
        x, y, w, h = cv2.boundingRect(contorno)  # boundingRect(array) recibe puntos y devuelve el mínimo rectángulo alineado a ejes: x,y,w,h.
        blancos = cv2.countNonZero(binaria[y:y + h, x:x + w])  # countNonZero(src) cuenta primer plano en la caja; un trazo fino puede tener contourArea=0 y aun contener píxeles válidos.
        if blancos < area_min:  # Compara píxeles de primer plano, no el área w*h del rectángulo, para descartar ruido diminuto.
            continue  # Excluye el componente según el umbral explícito de ruido.
        componentes.append({"x": x, "y": y, "w": w, "h": h})  # Conserva la geometría para separar puntos y evaluar posibles filas múltiples.
        cv2.drawContours(mascara, [contorno], -1, 255, cv2.FILLED)  # Recibe imagen,lista,índice,color,grosor; índice -1 dibuja todos y FILLED rellena su región exterior.
    limpia = cv2.bitwise_and(binaria, mascara)  # Intersección bit a bit: 255 AND 255=255 y cualquier AND 0=0; conserva los huecos originales que el dibujo relleno cubriría.
    if not componentes:  # Una imagen sin componentes por encima del ruido no contiene símbolos utilizables.
        return []  # Comunica ausencia de elementos al paso 5 para que su valor quede nulo.
    filas_activas = np.flatnonzero(np.any(limpia > 0, axis=1))  # Reduce columnas con OR lógico y obtiene las filas que contienen al menos un píxel blanco.
    y_superior = int(filas_activas[0])  # Define el límite superior del contenido, excluyendo márgenes vacíos de la ROI.
    alto_activo = int(filas_activas[-1]) - y_superior + 1  # Cuenta filas inclusivas entre el primer y último píxel: último-primero+1.
    altos_largos = [c["h"] for c in componentes if c["h"] >= 0.10 * alto_activo]  # Usa componentes altos para estimar el espacio vertical entre barras, evitando que puntos pequeños dominen la escala.
    hueco_vertical = max(2, min(round(0.60 * float(np.median(altos_largos))), round(0.18 * alto_activo))) if altos_largos else 2  # Tolera huecos internos hasta 60% de una barra, con límite de 18% del contenido total para no unir dos filas de unos continuos.
    bandas = intervalos_activos(np.count_nonzero(limpia, axis=1), hueco_vertical)  # Suma ocupaciones por fila y agrupa bandas cercanas para detectar filas de display claramente separadas.
    varias_filas = len(bandas) > 1  # Dos bandas separadas a esta escala no deben concatenarse como un único número.
    polaridad_sospechosa = cv2.countNonZero(limpia) / limpia.size > 0.75  # Más de 75% de primer plano suele indicar fondo invertido o una región sólida incompatible con segmentos separados.
    elementos = []  # Acumula cajas finales de dígitos y símbolos, preservando los desconocidos.
    sin_puntos = limpia.copy()  # Separa el análisis de columnas para que un decimal no se fusione con una cifra vecina.
    for componente in componentes:  # Evalúa los puntos sobre componentes individuales antes de agrupar las barras de los dígitos.
        x, y, w, h = (componente[k] for k in ("x", "y", "w", "h"))  # Extrae el rectángulo local del posible punto decimal.
        es_punto = w <= 0.20 * alto_activo and h <= 0.20 * alto_activo and 0.40 <= w / h <= 1.80 and y + h / 2 >= y_superior + 0.70 * alto_activo  # Exige tamaño pequeño, forma compacta y centro en la zona inferior de la fila.
        if es_punto:  # Los puntos que cumplen la geometría se conservan como símbolos independientes.
            elementos.append({**componente, "tipo": "desconocido" if varias_filas or polaridad_sospechosa else "decimal"})  # No acepta como decimal una máscara ambigua o de varias filas.
            sin_puntos[y:y + h, x:x + w] = 0  # Elimina solamente el rectángulo de este punto de la máscara auxiliar de agrupación horizontal.
    proyeccion_x = np.count_nonzero(sin_puntos, axis=0)  # P(x)=Σ_y 1[I(y,x)>0] cuenta píxeles blancos por columna y conserva la posición horizontal de cada símbolo.
    for x0, x1 in intervalos_activos(proyeccion_x, hueco):  # Agrupa columnas activas con pocos huecos para unir barras desconectadas sin dilatar la imagen.
        puntos = cv2.findNonZero(sin_puntos[:, x0:x1])  # findNonZero(src) devuelve coordenadas (x,y) de píxeles distintos de cero dentro de este intervalo.
        dx, y, w, h = cv2.boundingRect(puntos)  # Obtiene una caja ajustada sobre todos los segmentos del grupo, incluyendo sus huecos interiores.
        x = x0 + dx  # Convierte la coordenada horizontal relativa del intervalo a coordenada local de la ROI original.
        centro_y = y + h / 2  # Calcula el centro vertical geométrico del rectángulo para localizar una barra de signo en la mitad del display.
        if varias_filas or polaridad_sospechosa:  # Rechaza agrupaciones que mezclan filas o que parecen fondo blanco invertido.
            tipo = "desconocido"  # Deja evidencia en el paso 5 en vez de presentar un número combinado o inferido.
        elif h >= altura_min * alto_roi:  # Una cifra debe alcanzar la fracción solicitada del alto de la ROI, aunque sea estrecha como el uno.
            tipo = "digito"  # Marca la caja para evaluar sus siete segmentos en el paso siguiente, sin asignar todavía un valor.
        elif w / h >= 2 and abs(centro_y - (y_superior + alto_activo / 2)) <= 0.22 * alto_activo:  # Busca una barra horizontal al menos dos veces más ancha que alta y próxima al centro de la fila.
            tipo = "signo"  # Identifica un candidato a signo menos; su posición sintáctica se verificará durante la lectura.
        else:  # Una geometría que no cumple las reglas de cifra, decimal o signo no se elimina silenciosamente.
            tipo = "desconocido"  # Obliga a revisar el componente o recalibrar su ROI/filtros.
        elementos.append({"x": x, "y": y, "w": w, "h": h, "tipo": tipo})  # Guarda coordenadas locales enteras y la clasificación geométrica.
    return sorted(elementos, key=lambda elemento: (elemento["x"], elemento["y"]))  # Ordena los símbolos de izquierda a derecha, con desempate vertical determinista.
def leer_json(ruta):  # Recibe una ruta y devuelve su objeto JSON si el paso anterior generó el archivo.
    if not ruta.is_file():  # Detecta la falta de calibración o ROIs antes de procesar fotografías.
        raise ValueError(f"Falta {ruta}. Ejecuta primero los pasos 2 y 3 y guarda sus resultados.")  # Indica cómo resolver un requisito pendiente.
    return json.loads(ruta.read_text(encoding="utf-8"))  # Analiza el documento sin ejecutar su contenido como instrucciones.
def main(argv=None):  # Recibe opcionalmente argumentos de prueba y devuelve un código de salida para el paso 4.
    parser = argparse.ArgumentParser(description="Paso 4: contornos y cajas de símbolos de displays de siete segmentos.")  # Presenta una ayuda de consola con --help.
    parser.add_argument("--salida", type=Path, default=BASE / "outputs")  # Elige el directorio de manifiestos e imágenes, por defecto outputs junto al proyecto.
    parser.add_argument("--calibracion", type=Path)  # Permite cargar filtros de una ubicación diferente sin copiar su archivo.
    parser.add_argument("--rois", type=Path)  # Permite cargar coordenadas y fotografía desde otra configuración.
    parser.add_argument("--sin-ventana", action="store_true")  # Desactiva la interfaz gráfica durante pruebas o procesamiento por consola.
    parser.add_argument("--hueco", type=int, default=2)  # Establece la cantidad máxima de columnas vacías dentro de un grupo de segmentos.
    parser.add_argument("--altura-min", type=float, default=0.30)  # Establece la altura mínima de cifra como proporción respecto del alto de su ROI.
    parser.add_argument("--area-min", type=int, default=2)  # Filtra componentes con menos de esta cantidad de píxeles blancos en su caja.
    args = parser.parse_args(argv)  # Interpreta argumentos suministrados o la línea de comandos actual.
    ventanas = []  # Conserva los nombres de ventanas creadas para poder esperar eventos y cerrarlas al terminar.
    try:  # Protege la lectura de entradas, la detección y el guardado con un diagnóstico único de error.
        salida = args.salida.expanduser().resolve()  # Convierte la carpeta de resultados en una ruta absoluta para informes y archivos.
        ruta_calibracion = (args.calibracion or salida / "calibracion.json").resolve()  # Resuelve filtros explícitos o los guardados por defecto en el paso 2.
        ruta_rois = (args.rois or salida / "rois.json").resolve()  # Resuelve coordenadas explícitas o las guardadas por defecto en el paso 3.
        configuracion = leer_json(ruta_calibracion)  # Obtiene el diccionario de filtros reproducibles.
        manifiesto = leer_json(ruta_rois)  # Obtiene fotografía, resolución y coordenadas seleccionadas manualmente.
        imagen = cargar_imagen(ruta_rois.parent / manifiesto["imagen"])  # Carga la fotografía completa original, evitando depender de recortes PNG que pudieran haber quedado desactualizados.
        if [imagen.shape[1], imagen.shape[0]] != manifiesto["resolucion"]:  # Verifica W,H de la foto contra la selección, sin escalar ni truncar coordenadas.
            raise ValueError("La fotografía y el manifiesto de ROIs tienen distinta resolución; repite el paso 3.")  # Rechaza regiones cuyo origen geométrico cambió.
        validar_rois(imagen, manifiesto["rois"])  # Reutiliza comprobaciones de límites, enteros y nombres seguros antes de escribir imágenes.
        if not manifiesto["rois"]:  # Evita presentar como éxito una configuración que no contiene displays seleccionados.
            raise ValueError("El manifiesto no contiene ROIs; repite el paso 3.")  # Informa la selección necesaria antes de ejecutar contornos.
        resultados = []  # Conserva todas las regiones y cajas que se escribirán en el manifiesto del paso 4.
        procesadas = []  # Acumula imágenes en memoria antes de publicar archivos, evitando guardar resultados de una validación incompleta.
        for roi in manifiesto["rois"]:  # Procesa cada display independientemente de los demás.
            x, y, w, h = (roi[c] for c in ("x", "y", "w", "h"))  # Obtiene sus coordenadas absolutas previamente validadas.
            recorte = imagen[y:y + h, x:x + w].copy()  # R[j,i,c]=I[y+j,x+i,c]; extrae exactamente h filas y w columnas de la foto original.
            binaria = aplicar_filtros(recorte, configuracion["filtros"])["binaria"]  # Aplica la calibración guardada y utiliza primer plano blanco; Canny queda como diagnóstico visual del paso 2.
            elementos = detectar_elementos(binaria, args.hueco, args.altura_min, args.area_min)  # Localiza símbolos con los umbrales geométricos explícitos del usuario.
            anotada = recorte.copy()  # Conserva los píxeles originales y dibuja cajas sobre una copia independiente.
            for indice, elemento in enumerate(elementos):  # Enumera los símbolos en orden horizontal para relacionarlos con el JSON.
                ex, ey, ew, eh = (elemento[c] for c in ("x", "y", "w", "h"))  # Obtiene coordenadas locales dentro del recorte del display.
                color = (0, 0, 255) if elemento["tipo"] == "desconocido" else (0, 180, 0)  # Destaca en rojo elementos que necesitan revisión y en verde cajas geométricamente aceptadas.
                cv2.rectangle(anotada, (ex, ey), (ex + ew - 1, ey + eh - 1), color, 1)  # Recibe imagen,esquinas,color,grosor; el extremo inclusivo x+w-1 coincide con la última columna del recorte.
                cv2.putText(anotada, f"{indice}:{elemento['tipo']}", (ex, max(12, ey - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)  # Recibe imagen,texto,origen,fuente,escala,color,grosor; añade tipo y orden sin alterar la máscara binaria.
            nombre_binaria = f"{roi['nombre']}_binaria.png"  # Define un archivo de máscara independiente por display.
            nombre_anotada = f"{roi['nombre']}_contornos.png"  # Define un archivo de revisión visual por display.
            resultados.append({**roi, "binaria": nombre_binaria, "anotada": nombre_anotada, "elementos": elementos})  # Conserva coordenadas absolutas de la ROI y cajas locales de sus símbolos.
            procesadas.append((nombre_binaria, binaria, nombre_anotada, anotada))  # Retiene matrices y nombres para guardar después de validar todos los displays.
            print(f"{roi['nombre']}: {len(elementos)} elementos; tipos={[e['tipo'] for e in elementos]}")  # Informa la segmentación para detectar cifras fusionadas, símbolos ausentes o desconocidos.
        salida.mkdir(parents=True, exist_ok=True)  # Crea la carpeta una vez que las entradas y todas las ROIs se procesaron correctamente.
        for nombre_binaria, binaria, nombre_anotada, anotada in procesadas:  # Escribe los pares de imágenes ya procesadas.
            guardar_imagen(salida / nombre_binaria, binaria)  # Codifica PNG sin pérdida para preservar los valores binarios que usará el paso 5.
            guardar_imagen(salida / nombre_anotada, anotada)  # Guarda la copia anotada para inspección humana.
            if not args.sin_ventana:  # Crea vistas solo si se solicitó inspección gráfica.
                ventana = f"Paso 4 - {nombre_anotada}"  # Usa un nombre único por ROI para mostrar sus cajas.
                cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)  # namedWindow(winname,flags) permite redimensionar cada vista del recorte.
                cv2.imshow(ventana, anotada)  # imshow(winname,mat) muestra cajas sobre el display original para revisar agrupaciones.
                ventanas.append(ventana)  # Registra la vista que necesita procesar eventos y cerrarse.
        documento = {"version": 1, "fuente_rois": str(ruta_rois), "fuente_calibracion": str(ruta_calibracion), "resolucion": manifiesto["resolucion"], "deteccion": {"hueco": args.hueco, "altura_min": args.altura_min, "area_min": args.area_min}, "rois": resultados}  # Guarda también parámetros geométricos y procedencia para reproducir las detecciones.
        temporal = salida / "contornos.json.tmp"  # Evita truncar el manifiesto principal si su escritura se interrumpe.
        temporal.write_text(json.dumps(documento, ensure_ascii=False, indent=2), encoding="utf-8")  # Serializa cajas y procedencia conservando nombres y caracteres españoles.
        temporal.replace(salida / "contornos.json")  # Publica el nuevo manifiesto después de escribir todas las imágenes que referencia.
        print(f"Detecciones guardadas en {salida / 'contornos.json'}")  # Confirma la ubicación del resultado que necesita el paso 5.
        if ventanas:  # Espera la revisión humana cuando existen vistas abiertas.
            print("Pulsa una tecla o cierra una ventana para terminar la revisión.")  # Explica las formas de salir sin perder los resultados ya guardados.
            while cv2.waitKey(50) == -1:  # waitKey(delay) procesa eventos y devuelve -1 cuando no se recibió una tecla; no reprocesa las imágenes.
                try:  # Tolera que un backend deje de aceptar consultas después de destruir manualmente una ventana.
                    if any(cv2.getWindowProperty(v, cv2.WND_PROP_VISIBLE) < 1 for v in ventanas):  # getWindowProperty(winname,prop_id) comprueba si alguna vista dejó de ser visible.
                        break  # Finaliza la revisión al cerrar cualquiera de las vistas.
                except cv2.error:  # Una ventana destruida puede comunicar su ausencia mediante una excepción.
                    break  # Considera ese cierre una salida manual normal de la revisión.
        return 0  # La detección terminó; encontrar desconocidos se comunica en el manifiesto y lo evaluará el paso 5.
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:  # Maneja archivos, manifiestos, argumentos y procesamiento inválidos con un mensaje claro.
        print(f"ERROR del paso 4: {error}")  # Identifica por qué no pudo completarse la detección.
        return 1  # Señala un error operativo sin producir un manifiesto de éxito aparente.
    except KeyboardInterrupt:  # Permite cancelar la revisión o el procesamiento desde consola mediante Ctrl+C.
        print("Paso 4 interrumpido por el usuario.")  # Distingue una cancelación manual de un fallo de imagen.
        return 130  # Comunica interrupción de teclado con código convencional.
    finally:  # Cierra los recursos gráficos incluso si el procesamiento o la revisión falla.
        if ventanas:  # Evita llamadas HighGUI durante ejecuciones sin ventanas.
            cv2.destroyAllWindows()  # destroyAllWindows() sin argumentos cierra las ventanas creadas por este proceso.
if __name__ == "__main__":  # La importación de detectar_elementos desde el paso 5 no crea ventanas ni inicia main().
    raise SystemExit(main())  # Ejecuta el archivo directamente y devuelve su código al sistema operativo.
