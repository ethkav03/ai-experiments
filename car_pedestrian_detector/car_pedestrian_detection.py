import cv2

def main():
    video = cv2.VideoCapture("AdobeStock_304561508_Video_HD_Preview.mov")
    car_detector_model = 'haarcascade_car.xml'
    pedestrian_detector_model = 'haarcascade_fullbody.xml'
    # import model
    car_detector = cv2.CascadeClassifier(car_detector_model)
    pedestrian_detector = cv2.CascadeClassifier(pedestrian_detector_model)

    while True:
        (read, frame) = video.read()
        if read:
            # convert image to grayscale
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break

        # detect cars
        cars = car_detector.detectMultiScale(grayscale)

        # detect pedestrians
        pedestrians = pedestrian_detector.detectMultiScale(grayscale)

        # frame cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # frame pedestrians
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # display
        cv2.imshow("Car & Pedestrian Detector", frame)
        key = cv2.waitKey(1)

        if key == 81 or key == 113:
            break

    video.release()


if __name__ == "__main__":
    main()
