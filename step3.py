import cv2
import os
import urllib.request

# Haar 얼굴 검출기 XML 파일 준비 (없으면 한 번만 내려받음)
CASCADE_FILE = "haarcascade_frontalface_default.xml"
CASCADE_URL = ("https://raw.githubusercontent.com/opencv/opencv/4.x/"
               "data/haarcascades/haarcascade_frontalface_default.xml")

if not os.path.exists(CASCADE_FILE):
    print("얼굴 검출기 파일을 내려받는 중...")
    urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILE)

face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
if face_cascade.empty():
    print("얼굴 검출기를 불러올 수 없습니다.")
    raise SystemExit

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    raise SystemExit

WIN = "Sheriff Robot View"

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # ✅ 중심점 계산
        cx = x + w // 2
        cy = y + h // 2

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ✅ 십자선: 가로선 하나, 세로선 하나
        cv2.line(img, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
        cv2.line(img, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)

        # ✅ 글자 위치: 얼굴이 화면 맨 위에 있으면 네모 아래에 쓰기
        ty = y - 10 if y > 30 else y + h + 25
        cv2.putText(img, "TARGET LOCKED", (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow(WIN, img)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()