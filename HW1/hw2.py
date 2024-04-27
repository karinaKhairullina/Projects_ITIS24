import cv2
from ultralytics import YOLO

# Инициализация модели
model = YOLO('yolov8n.pt')

# Открыть видео
video_path = 'video.mp4'
video = cv2.VideoCapture(video_path)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Получить результаты распознавания машин (классы: 2, 5, 7)
    results = model.predict(frame, classes=[2, 5, 7])

    # Анализ результатов
    if len(results) > 0 and 'boxes' in results[0]:
        boxes = results[0]['boxes']
        for box in boxes:
            x1, y1, x2, y2, _, class_id = map(int, box)
            class_name = model.names[class_id]
            # Отрисовать прямоугольник вокруг объекта
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Нанести надпись с классом объекта
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Отображение кадра с прямоугольниками
    cv2.imshow('Video', frame)

    # Остановка видео
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  закрыть окно
video.release()
cv2.destroyAllWindows()