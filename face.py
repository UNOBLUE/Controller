import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np
import pyautogui
import time
import pygame

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

# Pygameの初期化
pygame.mixer.init()
# 効果音ファイルの読み込み
sound = pygame.mixer.Sound("modechange.mp3")
sound.set_volume(0.2)

def calculate_ear(eye):
    # 目のアスペクト比を計算する関数
    A = np.linalg.norm(eye[1] - eye[5])  # 縦の距離1
    B = np.linalg.norm(eye[2] - eye[4])  # 縦の距離2
    C = np.linalg.norm(eye[0] - eye[3])  # 横の距離

    ear = (A + B) / (2.0 * C)
    return ear

def check_eye_closed(shape):
    # 目のランドマークのインデックス
    left_eye = shape[36:42]   # 左目
    right_eye = shape[42:48]  # 右目

    # 左右の目のアスペクト比を計算
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)

    # 両目のEARの平均値を取得
    ear = (left_ear + right_ear) / 2.0

    # EARが一定値以下の場合、目が閉じていると判定（一般的に0.2~0.25がしきい値として使用される）
    return ear < 0.2


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
    # パターン3：顔の向きに合わせてマウスを滑らかに動かす（pitchとyawの大きさに応じて）
    x, y = pyautogui.position()  # 現在のマウスの位置
    w, h = pyautogui.size()  # 画面のサイズ

    # 基本の移動速度（小さい動きから大きい動きに拡張するためのベース値）
    base_speed = 5  # これを基準として移動距離を調整
    duration = 0.05  # 移動時間（秒）

    # pitch（上下方向）の角度に応じて移動距離を調整
    # 例えば、pitchが50度に近いほど移動が大きくなる
    vertical_speed = base_speed * abs(pitch)  # pitchの最大角度50度を基準に調整

    # 上方向
    if 10 < pitch < 50:
        if y - vertical_speed > 0:
            pyautogui.moveRel(0, -vertical_speed, duration=duration)  # 上に移動
        else:
            pyautogui.moveTo(x, 1, duration=duration)  # 画面上端に固定

    # 下方向
    if -50 < pitch < -10:
        if y + vertical_speed < h:
            pyautogui.moveRel(0, vertical_speed, duration=duration)  # 下に移動
        else:
            pyautogui.moveTo(x, h - 1, duration=duration)  # 画面下端に固定

    # yaw（左右方向）の角度に応じて移動距離を調整
    # 例えば、yawが50度に近いほど移動が大きくなる
    horizontal_speed = base_speed * abs(yaw)  # yawの最大角度50度を基準に調整

    # 左方向
    if 10 < yaw < 50:
        if x - horizontal_speed > 0:
            pyautogui.moveRel(-horizontal_speed, 0, duration=duration)  # 左に移動
        else:
            pyautogui.moveTo(1, y, duration=duration)  # 画面左端に固定

    # 右方向
    if -50 < yaw < -10:
        if x + horizontal_speed < w:
            pyautogui.moveRel(horizontal_speed, 0, duration=duration)  # 右に移動
        else:
            pyautogui.moveTo(w - 1, y, duration=duration)  # 画面右端に固定



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

        change_permission = True  # パターンが変更されたかを記録するフラグ

        # 両目が閉じられているか確認
        if check_eye_closed(shape):
            if start_time_eye is None:
                start_time_eye = time.time()  # 目が閉じられた時間を記録
            elif time.time() - start_time_eye > 1 and change_permission:
            # 1秒以上閉じていて、まだパターン変更が行われていない場合
                if time.time() - last_pattern_change > pattern_delay:
                    pattern = (pattern % 3) + 1  # パターンを1, 2, 3の順に切り替え
                    last_pattern_change = time.time()
                    change_permission = False  # パターンが変更されたことを記録
                    print(f"Pattern changed to: {pattern}")
                    sound.play()
        else:
            start_time_eye = None  # 目が開いた場合、閉じていた時間をリセット
            change_permission = True  # 目が開いたので次のパターン変更を許可


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
pygame.mixer.quit()
cv2.destroyAllWindows()
