
import cv2
import os
import urllib.request

# 얼굴 검출기 준비
CASCADE_FILE = "haarcascade_frontalface_default.xml"
CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/"
    "data/haarcascades/haarcascade_frontalface_default.xml"
)

if not os.path.exists(CASCADE_FILE):
    print("얼굴 검출기 다운로드 중...")
    urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILE)

face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

if face_cascade.empty():
    print("얼굴 검출기를 불러올 수 없습니다.")
    raise SystemExit

# 카메라 실행
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    raise SystemExit

WIN = "Camera XYZ Coordinates"

while True:
    success, img = cap.read()

    if not success:
        break

    # 거울처럼 좌우 반전
    img = cv2.flip(img, 1)

    H, W = img.shape[:2]

    # 화면 중앙을 좌표 원점으로 설정
    origin_x = W // 2
    origin_y = H // 2

    # 화면 중앙 십자선
    cv2.line(
        img, (origin_x - 15, origin_y),
        (origin_x + 15, origin_y), (255, 255, 255), 2
    )
    cv2.line(
        img, (origin_x, origin_y - 15),
        (origin_x, origin_y + 15), (255, 255, 255), 2
    )

    # 얼굴 감지
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # 기본 좌표값
    x_coord = 0
    y_coord = 0
    z_coord = 0.0

    if len(faces) > 0:
        # 가장 큰 얼굴 선택
        face = max(faces, key=lambda f: f[2] * f[3])

        x, y, w, h = face

        # 얼굴 중심 좌표
        cx = x + w // 2
        cy = y + h // 2

        # X, Y: 화면 중앙 기준 픽셀 좌표
        x_coord = cx - origin_x
        y_coord = origin_y - cy

        # Z: 얼굴 크기를 이용한 상대 거리 지표
        # 실제 미터 단위 거리는 아닙니다.
        z_coord = 1000.0 / max(w, 1)

        # 얼굴 테두리와 중심점
        cv2.rectangle(
            img, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )
        cv2.circle(
            img, (cx, cy), 6, (0, 255, 255), -1
        )

        # 화면 중앙에서 얼굴 중심까지 연결
        cv2.line(
            img, (origin_x, origin_y),
            (cx, cy), (0, 255, 255), 2
        )

        # 얼굴 중심 좌표 표시
        cv2.putText(
            img, "FACE CENTER",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 2
        )

    # 좌표 정보 표시 배경
    cv2.rectangle(
        img, (10, 10), (min(W - 10, 720), 95),
        (25, 25, 25), -1
    )

    # X 좌표
    cv2.putText(
        img, f"X: {x_coord:+d} px",
        (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 255), 2
    )

    # Y 좌표
    cv2.putText(
        img, f"Y: {y_coord:+d} px",
        (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 0), 2
    )

    # Z 상대 거리 지표
    z_text = (
        f"Z: {z_coord:.2f} (relative)"
        if len(faces) > 0
        else "Z: -- (no face)"
    )

    cv2.putText(
        img, z_text,
        (250, 65), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 150, 0), 2
    )

    # 축 설명
    cv2.putText(
        img, "X: Left/Right | Y: Up/Down",
        (10, H - 35), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1
    )
    cv2.putText(
        img, "Z: Relative depth indicator",
        (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1
    )

    # 카메라 화면 출력
    cv2.imshow(WIN, img)

    # Q 또는 ESC로 종료
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

    # 창을 X 버튼으로 닫은 경우
    if cv2.getWindowProperty(
        WIN, cv2.WND_PROP_VISIBLE
    ) < 1:
        break

cap.release()
cv2.destroyAllWindows()
