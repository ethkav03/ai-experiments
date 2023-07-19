import cv2
import random

def main():
    # pre-trained data
    trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # faces in video
    video_faces(trained_face_data)

def video_faces(trained_face_data):
    #video
    webcam = cv2.VideoCapture(0)
    _, frame = webcam.read()

    # get size of frame
    height, width = frame.shape[:2]

    # generate a random square on frame
    randx = random.randint(20, width - 20)
    randy = random.randint(20, height - 20)
    cv2.rectangle(frame, (randx, randy), (20, 20), (0, 255, 0), 2)

    score = 0

    while True:

        # read current frame
        _, frame = webcam.read()
        height, width = frame.shape[:2]

        # flip frame to mirror
        flipped = cv2.flip(frame, 1)

        # convert to grayscale
        grayscale = cv2.cvtColor(flipped, cv2.COLOR_BGRA2GRAY)

        # detect face
        face_coords = trained_face_data.detectMultiScale(grayscale)
        # print(face_coords)

        cv2.rectangle(flipped, (randx, randy), (randx + 20, randy + 20), (0, 255, 0), 2)
        cv2.putText(flipped, str(score), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        # rectangle around faces
        for (x, y, w, h) in face_coords:
            cv2.rectangle(flipped, (x, y), (x + w, y + h), (0, 0, 255), 2)        

        for (x, y, w, h) in face_coords:
            # increase score and generate a new random square on frame if player hits square
            if randx + 10 > x and randx + 10 < x + w and randy + 10 > y and randy + 10 < y + h:
                score += 1
                randx = random.randint(20, width - 20)
                randy = random.randint(20, height - 20)

        cv2.imshow('Face Detector', flipped)
        key = cv2.waitKey(1)

        # terminate
        if key == 81 or key == 113:
            break

    # clear webcam
    webcam.release()


if __name__ == "__main__":
    main()