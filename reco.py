import os
import cv2
import face_recognition
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from sklearn.metrics import classification_report

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Reconocimiento Facial")
root.geometry("800x600")

# Directorios de almacenamiento
capturas_dir = "D:/Desktop/reco/datos"
if not os.path.exists(capturas_dir):
    os.makedirs(capturas_dir)

# Variables para almacenar encodings
rostros_encodings = []
etiquetas = []

# Etiqueta para mostrar la cámara
label_cam = tk.Label(root)
label_cam.pack()

# Captura de imágenes
def capturar_imagenes():
    etiqueta = simpledialog.askstring("Etiqueta", "Ingresa el nombre para la captura:")
    if not etiqueta:
        messagebox.showwarning("Advertencia", "Debes ingresar una etiqueta.")
        return

    cam = cv2.VideoCapture(0)
    contador = 0
    ruta_capturas = os.path.join(capturas_dir, etiqueta)
    os.makedirs(ruta_capturas, exist_ok=True)

    def actualizar_frame():
        nonlocal contador
        ret, frame = cam.read()
        if not ret or contador >= 100:
            cam.release()
            label_cam.config(image="")
            cv2.destroyAllWindows()
            messagebox.showinfo("Captura Completa", f"100 imágenes capturadas para {etiqueta}.")
            return

        # Guardar la imagen capturada
        img_path = os.path.join(ruta_capturas, f"{etiqueta}_{contador}.jpg")
        cv2.imwrite(img_path, frame)
        contador += 1

        # Convertir el frame de OpenCV a formato ImageTk para mostrarlo en la GUI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label_cam.imgtk = imgtk
        label_cam.config(image=imgtk)

        # Llamar a la función de actualización cada 100 ms
        root.after(100, actualizar_frame)

    actualizar_frame()

# Entrenamiento del modelo
def entrenar_modelo():
    global rostros_encodings, etiquetas
    rostros_encodings = []
    etiquetas = []

    # Procesar cada carpeta en capturas_dir
    for etiqueta in os.listdir(capturas_dir):
        ruta_etiqueta = os.path.join(capturas_dir, etiqueta)
        if not os.path.isdir(ruta_etiqueta):
            continue

        # Procesar cada imagen en la carpeta
        for imagen_nombre in os.listdir(ruta_etiqueta):
            imagen_path = os.path.join(ruta_etiqueta, imagen_nombre)
            imagen = face_recognition.load_image_file(imagen_path)
            encodings = face_recognition.face_encodings(imagen)
            if encodings:
                rostros_encodings.append(encodings[0])
                etiquetas.append(etiqueta)

    messagebox.showinfo("Entrenamiento Completo", "El modelo ha sido entrenado con las imágenes capturadas.")

# Reconocimiento facial en tiempo real
def reconocimiento_facial():
    if not rostros_encodings:
        messagebox.showerror("Error", "Primero debes entrenar el modelo.")
        return

    cam = cv2.VideoCapture(0)

    def actualizar_reconocimiento():
        ret, frame = cam.read()
        if not ret:
            cam.release()
            label_cam.config(image="")
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rostros = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, rostros)

        for (top, right, bottom, left), encoding in zip(rostros, encodings):
            coincidencias = face_recognition.compare_faces(rostros_encodings, encoding)
            etiqueta = "Desconocido"

            # Asignar la etiqueta si hay coincidencia
            if True in coincidencias:
                idx = coincidencias.index(True)
                etiqueta = etiquetas[idx]

            # Dibujar el rectángulo alrededor del rostro y la etiqueta
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, etiqueta, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Convertir el frame a formato ImageTk para mostrarlo en la GUI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label_cam.imgtk = imgtk
        label_cam.config(image=imgtk)

        # Llamar a la función de actualización cada 10 ms
        root.after(10, actualizar_reconocimiento)

    actualizar_reconocimiento()

# Reconocimiento facial en un video local
def reconocimiento_video():
    if not rostros_encodings:
        messagebox.showerror("Error", "Primero debes entrenar el modelo.")
        return

    # Seleccionar archivo de video
    filepath = filedialog.askopenfilename(
        title="Seleccionar Video",
        filetypes=[("Archivos de Video", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    if not filepath:
        return

    # Crear una nueva ventana para el video
    ventana_video = tk.Toplevel(root)
    ventana_video.title("Reconocimiento en Video")
    ventana_video.geometry("800x600")

    # Etiqueta para mostrar el video
    label_video = tk.Label(ventana_video)
    label_video.pack(fill="both", expand=True)

    cam = cv2.VideoCapture(filepath)
    fps = cam.get(cv2.CAP_PROP_FPS)
    intervalo = int(1000 / fps) if fps > 0 else 33

    def cerrar_video():
        """Libera la cámara y cierra la ventana de video."""
        cam.release()
        ventana_video.destroy()

    # Vincula el botón de cierre a la liberación de la cámara
    ventana_video.protocol("WM_DELETE_WINDOW", cerrar_video)

    def procesar_video():
        ret, frame = cam.read()
        if not ret:
            cerrar_video()
            messagebox.showinfo("Video Finalizado", "Se completó el procesamiento del video.")
            return

        # Convertir frame a RGB para face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rostros = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, rostros)

        # Detección y etiquetas
        for (top, right, bottom, left), encoding in zip(rostros, encodings):
            coincidencias = face_recognition.compare_faces(rostros_encodings, encoding)
            etiqueta = "Desconocido"

            if True in coincidencias:
                idx = coincidencias.index(True)
                etiqueta = etiquetas[idx]

            # Dibujar rectángulo y etiqueta
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, etiqueta, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Obtener dimensiones actuales de la ventana
        ventana_width = ventana_video.winfo_width()
        ventana_height = ventana_video.winfo_height()

        # Redimensionar el frame al tamaño de la ventana
        frame = cv2.resize(frame, (ventana_width, ventana_height))

        # Convertir frame a formato ImageTk
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label_video.imgtk = imgtk
        label_video.config(image=imgtk)

        # Procesar el siguiente fotograma
        ventana_video.after(intervalo, procesar_video)

    procesar_video()

def evaluar_modelo():
    global rostros_encodings, etiquetas

    if not rostros_encodings:
        messagebox.showerror("Error", "Primero debes entrenar el modelo.")
        return

    directorio = filedialog.askdirectory(title="Seleccionar directorio de prueba")
    if not directorio:
        return

    # Directorio de salida para guardar imágenes etiquetadas
    directorio_salida = os.path.join(directorio, "resultado_etiquetado")
    os.makedirs(directorio_salida, exist_ok=True)

    y_true = []
    y_pred = []

    for etiqueta in os.listdir(directorio):
        carpeta = os.path.join(directorio, etiqueta)
        if not os.path.isdir(carpeta):
            continue

        for imagen_nombre in os.listdir(carpeta):
            imagen_path = os.path.join(carpeta, imagen_nombre)

            try:
                imagen = face_recognition.load_image_file(imagen_path)
                rostros = face_recognition.face_locations(imagen)
                encodings = face_recognition.face_encodings(imagen, rostros)

                # Convertir la imagen a formato BGR para dibujar con OpenCV
                imagen_bgr = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

                if not encodings:
                    print(f"No se detectaron rostros en {imagen_nombre}")
                    continue

                for rostro, encoding in zip(rostros, encodings):
                    coincidencias = face_recognition.compare_faces(rostros_encodings, encoding)
                    prediccion = "Desconocido"

                    if True in coincidencias:
                        idx = coincidencias.index(True)
                        prediccion = etiquetas[idx]

                    # Añadir resultados al conjunto de métricas
                    y_true.append(etiqueta)
                    y_pred.append(prediccion)

                    # Dibujar el cuadro y la etiqueta en la imagen
                    top, right, bottom, left = rostro
                    cv2.rectangle(imagen_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(imagen_bgr, prediccion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Guardar la imagen etiquetada
                salida_path = os.path.join(directorio_salida, f"etiquetada_{imagen_nombre}")
                cv2.imwrite(salida_path, imagen_bgr)

            except Exception as e:
                print(f"Error al procesar {imagen_nombre}: {e}")

    # Validar que las listas no estén vacías antes de generar métricas
    if not y_true or not y_pred:
        messagebox.showerror("Error", "No se pudieron calcular métricas: conjunto de prueba vacío o sin detecciones.")
        return

    # Generar reporte de métricas
    reporte = classification_report(y_true, y_pred, zero_division=0)
    print("Reporte de métricas:\n", reporte)
    messagebox.showinfo("Métricas", f"Reporte generado. Consulta la consola para más detalles.")
    messagebox.showinfo("Etiquetas", f"Las imágenes etiquetadas se guardaron en: {directorio_salida}")



# Botones de la interfaz
btn_capturar = tk.Button(root, text="Capturar Imágenes", command=capturar_imagenes, width=20, height=2)
btn_capturar.pack(pady=10)

btn_entrenar = tk.Button(root, text="Entrenar Modelo", command=entrenar_modelo, width=20, height=2)
btn_entrenar.pack(pady=10)

btn_reconocer = tk.Button(root, text="Iniciar Reconocimiento", command=reconocimiento_facial, width=20, height=2)
btn_reconocer.pack(pady=10)

btn_video = tk.Button(root, text="Reconocer en Video", command=reconocimiento_video, width=20, height=2)
btn_video.pack(pady=10)

btn_evaluar = tk.Button(root, text="Evaluar Modelo", command=evaluar_modelo, width=20, height=2)
btn_evaluar.pack(pady=10)

# Ejecutar la interfaz
root.mainloop()