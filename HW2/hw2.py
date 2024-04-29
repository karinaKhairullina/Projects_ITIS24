import cv2
from ultralytics import YOLO

# Инициализация модели
model = YOLO('yolov8s.pt')

# Открыть видео
video_path = 'output2.avi'
video = cv2.VideoCapture(video_path)

# Параметры для линии и подсчета трафика
line_position = 500  # Позиция линии
traffic_count = {'встречное': 0, 'попутное': 0}  # Счетчик трафика

# Словарь для отслеживания машин
tracked_cars = {}

while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame)

    # Отрисовка разделительной полосы
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id in [2, 5, 7]:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Определение центра объекта
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Проверка, пересекла ли машина линию и не встречалась ли ранее
                if center_y > line_position and (class_id, center_x, center_y) not in tracked_cars:
                    # Отрисовка прямоугольника
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # Подсчет трафика
                    traffic_count['встречное'] += 1

                    # Добавление машины в отслеживание
                    tracked_cars[(class_id, center_x, center_y)] = True

                elif center_y <= line_position and (class_id, center_x, center_y) not in tracked_cars:
                    # Отрисовка прямоугольника
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Подсчет трафика
                    traffic_count['попутное'] += 1

                    # Добавление машины в отслеживание
                    tracked_cars[(class_id, center_x, center_y)] = True

    # Отображение количества машин на встречном и попутном движении на экране
    cv2.putText(frame, f"Oncoming: {traffic_count['встречное']}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Opposite: {traffic_count['попутное']}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Отображение кадра с прямоугольниками
    cv2.imshow('Video', frame)

    # Остановка видео при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрыть видео
video.release()
cv2.destroyAllWindows()