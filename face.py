import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np
import pyautogui
import time

# VideoCapture オブジェクトを取得します
DEVICE_ID = 0
capture = cv2.VideoCapture(DEVICE_ID)
predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 遷移を管理するための変数
start_time_eye = None
pattern = 1
last_pattern_change = time.time()
pattern_delay = 1.0  # 1秒ごとにパターン変更
wait_for_reset_yaw = False
wait_for_reset_pitch = False
reset_time_yaw = None
reset_time_pitch = None
last_yaw_press = None
last_pitch_press = None

def check_eye_closed(shape):
    # 両目が閉じているかの確認
    Right_eye_upper_y = shape[43][1]
    Right_eye_lower_y = shape[47][1]
    Left_eye_upper_y = shape[37][1]
    Left_eye_lower_y = shape[41][1]
    print(Right_eye_upper_y - Right_eye_lower_y, Left_eye_upper_y - Left_eye_lower_y)

    return Right_eye_upper_y - Right_eye_lower_y > -10 and Left_eye_upper_y - Left_eye_lower_y > -10

def process_pattern_1(yaw, pitch):
    # パターン1：連続で方向キーを入力
    if 10 < yaw < 50:
        pyautogui.press('left')
    elif -50 < yaw < -10:
        pyautogui.press('right')

    if 20 < pitch < 50:
        pyautogui.press('up')
    elif -50 < pitch < -15:
        pyautogui.press('down')

def process_pattern_2(yaw, pitch):
    # パターン2：一度だけ方向キーを入力し、顔を元の位置に戻すと次の入力ができる
    global wait_for_reset_yaw, wait_for_reset_pitch, last_yaw_press, last_pitch_press

    if 10 < yaw < 50 and not wait_for_reset_yaw:
        pyautogui.press('left')
        wait_for_reset_yaw = True
        last_yaw_press = time.time()
    elif -50 < yaw < -10 and not wait_for_reset_yaw:
        pyautogui.press('right')
        wait_for_reset_yaw = True
        last_yaw_press = time.time()

    if 20 < pitch < 50 and not wait_for_reset_pitch:
        pyautogui.press('up')
        wait_for_reset_pitch = True
        last_pitch_press = time.time()
    elif -50 < pitch < -15 and not wait_for_reset_pitch:
        pyautogui.press('down')
        wait_for_reset_pitch = True
        last_pitch_press = time.time()

    # 顔が中央に戻った場合にリセット
    if abs(yaw) < 10 and wait_for_reset_yaw and time.time() - last_yaw_press > 1:
        wait_for_reset_yaw = False
    if abs(pitch) < 10 and wait_for_reset_pitch and time.time() - last_pitch_press > 1:
        wait_for_reset_pitch = False

def process_pattern_3(yaw, pitch):
    # パターン3：顔の向きに合わせてマウスを動かす
    x,y = pyautogui.position()
    w,h = pyautogui.size()
    if 10 < pitch < 50:
        #print("上")
        if not y-100<0:
            pyautogui.moveRel(0, -100)
        else:
            pyautogui.moveTo(x,1)
    if -50 < pitch < -10:
        #print("下")
        if not y+100>h:
            pyautogui.moveRel(0, 100)
        else:
            pyautogui.moveTo(x,h-1)
    if 10 < yaw < 50:
        #print("左")
        if not x-100<0:
            pyautogui.moveRel(-100, 0)
        else:
            pyautogui.moveTo(1,y)
    if -50 < yaw < -10:
        #print("右")
        if not x+100>w:
            pyautogui.moveRel(100, 0)
        else:
            pyautogui.moveTo(w-1,y)

while True:
    ret, frame = capture.read()
    frame = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    image_points = None

    # Fail-Safeを無効にする
    pyautogui.FAILSAFE = False #マウスが画面外に行ってもエラーを出さない

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        image_points = np.array([
                tuple(shape[30]),  # 鼻頭
                tuple(shape[21]),
                tuple(shape[22]),
                tuple(shape[39]),
                tuple(shape[42]),
                tuple(shape[31]),
                tuple(shape[35]),
                tuple(shape[48]),
                tuple(shape[54]),
                tuple(shape[57]),
                tuple(shape[8]),
                ], dtype='double')

    if len(rects) > 0:
        model_points = np.array([
                (0.0, 0.0, 0.0),
                (-30.0, -125.0, -30.0),
                (30.0, -125.0, -30.0),
                (-60.0, -70.0, -60.0),
                (60.0, -70.0, -60.0),
                (-40.0, 40.0, -50.0),
                (40.0, 40.0, -50.0),
                (-70.0, 130.0, -100.0),
                (70.0, 130.0, -100.0),
                (0.0, 158.0, -10.0),
                (0.0, 250.0, -50.0)
                ])

        size = frame.shape
        focal_length = size[1]
        center = (size[1] // 2, size[0] // 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype='double')
        dist_coeffs = np.zeros((4, 1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
        mat = np.hstack((rotation_matrix, translation_vector))
        (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
        yaw = eulerAngles[1].item()
        pitch = eulerAngles[0].item()
        roll = eulerAngles[2].item()

        # 両目が閉じられているか確認
        if check_eye_closed(shape):
            if start_time_eye is None:
                start_time_eye = time.time()
            elif time.time() - start_time_eye > 1:
                # 1秒以上閉じていたらパターン変更
                if time.time() - last_pattern_change > pattern_delay:
                    pattern = (pattern % 3) + 1
                    last_pattern_change = time.time()
                    print(f"Pattern changed to: {pattern}")
        else:
            start_time_eye = None

        # 現在のパターンに応じて処理
        if pattern == 1:
            process_pattern_1(yaw, pitch)
        elif pattern == 2:
            process_pattern_2(yaw, pitch)
        elif pattern == 3:
            process_pattern_3(yaw, pitch)

        cv2.putText(frame, f'yaw : {int(yaw)}', (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'pitch : {int(pitch)}', (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'roll : {int(roll)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
