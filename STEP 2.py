import cv2
import os
import urllib.request  # ✅

# ✅ Haar 얼굴 검출기 XML 파일 준비 (없으면 한 번만 내려받음)
CASCADE_FILE = "haarcascade_frontalface_default.xml"
CASCADE_URL = ("https://raw.githubusercontent.com/opencv/opencv/4.x/"
               "data/haarcascades/haarcascade_frontalface_default.xml")

if not os.path.exists(CASCADE_FILE):
    print("얼굴 검출기 파일을 내려받는 중...")
    urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILE)

# ✅ 얼굴 검출기 불러오기
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

    # ✅ 흑백으로 바꾸기 (검출기는 밝기 차이만 봄)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ✅ 얼굴 찾기
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

    # ✅ 찾은 얼굴마다 초록 네모 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow(WIN, img)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()