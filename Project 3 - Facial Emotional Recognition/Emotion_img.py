from facial_emotion_recognition import EmotionRecognition
import cv2

er = EmotionRecognition(device='cpu')

# Correct file path with double backslashes
image_path = 'C:\\Users\\sen_s\\OneDrive\\Documents\\AI\\Project 3 - Facial Emotional Recognition\\download (1).jpg'

frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not load the image. Check the file path.")
else:
    # Recognize emotion in the image
    frame = er.recognise_emotion(frame, return_type='BGR')

    # Display the result
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
