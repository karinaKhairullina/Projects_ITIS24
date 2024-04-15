import cv2
import numpy as np

video_capture = cv2.VideoCapture('video.mp4')

# Создание объектов для вычитания фона и определения контуров
background_subtractor = cv2.createBackgroundSubtractorMOG2()
kernel = np.ones((5, 5), np.uint8)

# Пороговое значение площади контура
AREA_THRESHOLD = 500

# Переменные для отслеживания объектов на двух полосах
lane1_cars = []
lane2_cars = []

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Метод вычитания фона
    fg_mask = background_subtractor.apply(frame)

    # Улучшение контуров
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Нахождение контуров
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отрисовываем контуры и определяем направление движения объектов
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > AREA_THRESHOLD:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Определение центра контура
            center_x = x + w // 2

            # Положение центра контура относительно середины фрейма
            if center_x < frame.shape[1] // 2:
                # не добавлена ли уже эта машина в lane1_cars
                is_new_car = True
                for car in lane1_cars:
                    if abs(center_x - car[0]) < w / 2:
                        is_new_car = False
                        break
                if is_new_car:
                    lane1_cars.append((center_x, y))
            else:
                #  не добавлена ли уже эта машина в lane2_cars
                is_new_car = True
                for car in lane2_cars:
                    if abs(center_x - car[0]) < w / 2:
                        is_new_car = False
                        break
                if is_new_car:
                    lane2_cars.append((center_x, y))

    # Пересекают ли машины середину фрейма
    lane1_count = len(lane1_cars)
    lane2_count = len(lane2_cars)

    # Cчетчики на кадр
    cv2.putText(frame, f"Lane 1: {lane1_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Lane 2: {lane2_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Отображение видео
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
