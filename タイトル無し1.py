import cv2 #OpenCV:画像処理系ライブラリ

import dlib #機械学習系ライブラリ

import imutils #OpenCVの補助

from imutils import face_utils

import numpy as np

import pyautogui

import serial


# VideoCapture オブジェクトを取得します

DEVICE_ID = 0 #ID 0は標準web cam

capture = cv2.VideoCapture(DEVICE_ID)#dlibの学習済みデータの読み込み

predictor_path = "shape_predictor_68_face_landmarks.dat"



detector = dlib.get_frontal_face_detector() #顔検出器の呼び出し。ただ顔だけを検出する。

predictor = dlib.shape_predictor(predictor_path) #顔から目鼻などランドマークを出力する


ser = serial.Serial('COM3', 9600) 
not_used = ser.readline()
val_decoded=1000

while(True): #カメラから連続で画像を取得する

    val_arduino = ser.readline()
    ret, frame = capture.read() #カメラからキャプチャしてframeに１コマ分の画像データを入れる



    frame = imutils.resize(frame, width=1000) #frameの画像の表示サイズを整える

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #gray scaleに変換する

    rects = detector(gray, 0) #grayから顔を検出

    image_points = None



    for rect in rects:

        shape = predictor(gray, rect)

        shape = face_utils.shape_to_np(shape)



        for (x, y) in shape: #顔全体の68箇所のランドマークをプロット

            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)



        image_points = np.array([

                tuple(shape[30]),#鼻頭

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

                ],dtype='double')



    if len(rects) > 0:

        model_points = np.array([

                (0.0,0.0,0.0), # 30

                (-30.0,-125.0,-30.0), # 21

                (30.0,-125.0,-30.0), # 22

                (-60.0,-70.0,-60.0), # 39

                (60.0,-70.0,-60.0), # 42

                (-40.0,40.0,-50.0), # 31

                (40.0,40.0,-50.0), # 35

                (-70.0,130.0,-100.0), # 48

                (70.0,130.0,-100.0), # 54

                (0.0,158.0,-10.0), # 57

                (0.0,250.0,-50.0) # 8

                ])



        size = frame.shape



        focal_length = size[1]

        center = (size[1] // 2, size[0] // 2) #顔の中心座標



        camera_matrix = np.array([

            [focal_length, 0, center[0]],

            [0, focal_length, center[1]],

            [0, 0, 1]

        ], dtype='double')



        dist_coeffs = np.zeros((4, 1))



        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,

                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        #回転行列とヤコビアン

        (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)

        mat = np.hstack((rotation_matrix, translation_vector))



        #yaw,pitch,rollの取り出し

        (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)

        yaw = eulerAngles[1]

        pitch = eulerAngles[0]

        roll = eulerAngles[2]



        # #頭頂部を左右に向けるパターン　精度があまり安定しないが、画面を見続けやすい点で優れている

        # if 10 < roll < 50:

        #     print("左")

        # elif -50 < roll < -10:

        #     print("右")

        #顔の正面を左右に向けるパターン 精度はこっちのほうが高そうだが、画面が見にくい

        x,y = pyautogui.position()
        w,h = pyautogui.size()
        if (int(val_arduino)-int(val_decoded))>200:
            pyautogui.click()
            pyautogui.write("q")
            #pass
        if 10 < pitch < 50:

            print("上")
            if not y-100<0:
                pyautogui.moveRel(0, -100)
            else:
                pyautogui.moveTo(x,1)
        elif -50 < pitch < -10:

            print("下")
            if not y+100>h:
                pyautogui.moveRel(0, 100)
            else:
                pyautogui.moveTo(x,h-1)
        elif 10 < yaw < 50:

            print("左")
            if not x-100<0:
                pyautogui.moveRel(-100, 0)
            else:
                pyautogui.moveTo(1,y)
        elif -50 < yaw < -10:

            print("右")
            if not x+100>w:
                pyautogui.moveRel(100, 0)
            else:
                pyautogui.moveTo(w-1,y)

        







        cv2.putText(frame, 'yaw : ' + str(int(yaw)), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.putText(frame, 'pitch : ' + str(int(pitch)), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.putText(frame, 'roll : ' + str(int(roll)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)



        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,

                                                         translation_vector, camera_matrix, dist_coeffs)

        #計算に使用した点のプロット/顔方向のベクトルの表qrightqrightqrightqrightqrightqrightqrightrightqrightqright

        for p in image_points:

            cv2.drawMarker(frame, (int(p[0]), int(p[1])),  (0.0, 1.409845, 255),markerType=cv2.MARKER_CROSS, thickness=1)



        p1 = (int(image_points[0][0]), int(image_points[0][1]))

        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))



        cv2.arrowedLine(frame, p1, p2, (255, 0, 0), 2)



    cv2.imshow('frame',frame) # 画像を表示する

    val_decoded = val_arduino
    print(val_decoded)

    if cv2.waitKey(1) & 0xFF == ord('q'): #qを押すとbreakしてwhileから抜ける

        break





capture.release() #video captureを終了する

cv2.destroyAllWindows() #windowを閉じる


ser.close()