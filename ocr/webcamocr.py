import cv2 
import pytesseract 
import threading 
import torch 
import sounddevice as sd 
import numpy as np 
import time 
from transformers import pipeline


asr = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=-1)

east_net = cv2.dnn.readNet("/Users/kathleenalvescosta/Desktop/school/independentstudy/frozen_east_text_detection.pb")

ocr_result = ""
lock = threading.Lock()
ocr_active = False
running = True
last_ocr_time = 0

def detect_text_regions_east(image, net, conf_threshold=0.5):
    h, w = image.shape[:2]
    new_w, new_h = (320, 320)
    r_w = w / float(new_w)
    r_h = h / float(new_h)

    blob = cv2.dnn.blobFromImage(image, 1.0, (new_w, new_h),
                                 (123.68, 116.78, 103.94), True, False)

    net.setInput(blob)
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    (num_rows, num_cols) = scores.shape[2:4]
    boxes = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]

        for x in range(num_cols):
            score = scores_data[x]
            if score < conf_threshold:
                continue

            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = angles[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x0[x] + x2[x]
            w = x1[x] + x3[x]

            end_x = int(offset_x + cos * x1[x] + sin * x2[x])
            end_y = int(offset_y - sin * x1[x] + cos * x2[x])
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            boxes.append((int(start_x * r_w), int(start_y * r_h),
                          int(end_x * r_w), int(end_y * r_h)))
            confidences.append(float(score))

    indices = cv2.dnn.NMSBoxes(
        [(x, y, ex - x, ey - y) for x, y, ex, ey in boxes],
        confidences, conf_threshold, 0.4
    )

    final_boxes = []
    for i in indices:
        i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
        final_boxes.append(boxes[i])
    return final_boxes

def ocr_thread(gray_frame, rois):
    global ocr_result
    text_lines = []
    for x1, y1, x2, y2 in rois:
        roi = gray_frame[y1:y2, x1:x2]
        data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT, config='--oem 3 --psm 7')
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            if text and conf > 70:
                text_lines.append((x1, y1, text))
    with lock:
        ocr_result = text_lines

def asr_thread():
    global ocr_active
    samplerate = 16000
    duration = 5

    while running:
        print("Listening...")
        audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()

        audio = np.squeeze(audio)

        try:
            transcription = asr({"array": audio, "sampling_rate": samplerate})["text"].lower()
            print(f"Heard: {transcription}")
            if any(phrase in transcription for phrase in ["start", "on", "what does this say"]):
                ocr_active = True
            elif any(phrase in transcription for phrase in ["stop", "off"]):
                ocr_active = False
        except Exception as e:
            print(f"Error in ASR: {e}")

threading.Thread(target=asr_thread, daemon=True).start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display_frame = frame.copy()

    if ocr_active and time.time() - last_ocr_time > 2:
        last_ocr_time = time.time()
        boxes = detect_text_regions_east(frame, east_net, conf_threshold=0.3)
        rois = [(x1, y1, x2, y2) for (x1, y1, x2, y2) in boxes]
        if rois:
            threading.Thread(target=ocr_thread, args=(gray.copy(), rois), daemon=True).start()

    with lock:
        for x, y, text in ocr_result:
            cv2.rectangle(display_frame, (x, y), (x + 150, y + 30), (0, 255, 0), 2)
            cv2.putText(display_frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    status_text = "OCR: ON" if ocr_active else "OCR: OFF"
    cv2.putText(display_frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255 if not ocr_active else 0), 2)

    cv2.imshow('Webcam Assistant', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

running = False
cap.release()
cv2.destroyAllWindows()
