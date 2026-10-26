import cv2

cap = cv2.VideoCapture(0)          # 0번 카메라(기본 웹캠) 연결

if not cap.isOpened():             # 카메라가 안 열리면 알려 주고 종료
    print("카메라를 열 수 없습니다.")
    raise SystemExit

WIN = "Sheriff Robot View"         # 창 이름

while True:
    success, img = cap.read()      # 사진 한 장(프레임) 찍기
    if not success:                # 사진을 못 찍었으면 반복 종료
        break

    img = cv2.flip(img, 1)         # 좌우 반전 (거울처럼)
    cv2.imshow(WIN, img)           # 화면에 보여 주기

    key = cv2.waitKey(1) & 0xFF    # 1ms 동안 키 입력 기다리기
    if key in (ord('q'), 27):      # q 또는 ESC를 누르면 종료
        break
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:   # X 버튼으로 창을 닫으면 종료
        break

cap.release()                      # 카메라 놓아주기
cv2.destroyAllWindows()            # 창 모두 닫기