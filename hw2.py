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

# Список для отслеживания прямоугольников на текущем кадре
current_rectangles = []

# Переменные для отслеживания прямоугольников на предыдущем кадре
previous_rectangles = []
previous_frame_number = None

# Функция для вычисления расстояния между двумя точками
def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame)

    # Отрисовка разделительной полосы
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)

    # Очистка списка прямоугольников на текущем кадре
    current_rectangles.clear()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id in [2, 5, 7]:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Определение центра объекта
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Отрисовка прямоугольника
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Добавление прямоугольника в список текущих прямоугольников
                current_rectangles.append(((x1, y1), (x2, y2)))

    # Проверка наличия прямоугольников на предыдущем кадре
    if previous_frame_number is not None:
        # Переменная для хранения прямоугольников, которые находятся на том же месте
        same_position_rectangles = []

        # Перебор прямоугольников на предыдущем кадре
        for prev_rect in previous_rectangles:
            # Перебор прямоугольников на текущем кадре
            for curr_rect in current_rectangles:
                # Вычисление расстояния между центрами прямоугольников
                prev_center = ((prev_rect[0][0] + prev_rect[1][0]) // 2, (prev_rect[0][1] + prev_rect[1][1]) // 2)
                curr_center = ((curr_rect[0][0] + curr_rect[1][0]) // 2, (curr_rect[0][1] + curr_rect[1][1]) // 2)
                dist = distance(prev_center, curr_center)

                # Если расстояние между центрами прямоугольников меньше порогового значения,
                # считаем, что прямоугольники находятся на том же месте
                if dist < 20:
                    same_position_rectangles.append(prev_rect)

        # Отрисовка прямоугольников на том же месте
        for rect in same_position_rectangles:
            cv2.rectangle(frame, rect[0], rect[1], (255, 0, 0), 2)

    # Обновление списка прямоугольников на предыдущем кадре
    previous_rectangles = current_rectangles.copy()
    previous_frame_number = video.get(cv2.CAP_PROP_POS_FRAMES)

    # Подсчет количества прямоугольников и отображение на видео
    oncoming_count = sum(1 for rect in current_rectangles if rect[0][1] > line_position)
    opposite_count = sum(1 for rect in current_rectangles if rect[0][1] <= line_position)
    cv2.putText(frame, f"Oncoming: {oncoming_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Opposite: {opposite_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Отображение кадра
    cv2.imshow('Video', frame)

    # Остановка видео при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрыть видео
video.release()
cv2.destroyAllWindows()
