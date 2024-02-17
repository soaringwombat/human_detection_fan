# ライブラリのインポート
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import threading
import cv2 as cv
import numpy as np
import imutils
import time
import keyboard

# PID制御の変数
M = 0.00        # 操作量
M1 =  0.00      # 前回の操作量
goal = 50.00    # 目標値
e = 0.00        # 偏差
e1 = 0.00       # 前回の偏差
e2 = 0.00       # 前々回の偏差
Kp = 0.01       # 比例定数
Ki = 0.1        # 積分定数
Kd = 0.1        # 微分定数
t = 100         # 制御周期

# GPIOピンの設定
shakeMotorCW = 12
shakeMotorCCW = 13
shakeEncoderA = 16
shakeEncoderB = 20
modeOutputAPin = 5
modeOutputBPin = 6

# PWMの変数
freq = 100

# モード変更
modeOutputA = 0
modeOutputB = 0

# 首振りの変数
targetAngle = 0
nowAngle = 0

# 人の有無
personFlag = False

# モデルのオブジェクトを指定
labels = ["background", "aeroplane", "bicycle", "bird", 
"boat","bottle", "bus", "car", "cat", "chair", "cow", 
"diningtable","dog", "horse", "motorbike", "person", "pottedplant", 
"sheep","sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))
detectionFilterValue = 0.8  # 0.0~1.0でフィルターの閾値を設定

# デバッグ用
def event_callback(pinname):
    global nowAngle, flag, shakeEncoderB
    if(GPIO.input(shakeEncoderB) == 1):
        nowAngle += 1
        print("CW: ", nowAngle)
    else:
        nowAngle -= 1
        print("CCW: ", nowAngle)

# GPIOの初期設定
GPIO.setmode(GPIO.BCM)
GPIO.setup(shakeMotorCW, GPIO.OUT)
GPIO.setup(shakeMotorCCW, GPIO.OUT)
GPIO.setup(shakeEncoderA, GPIO.IN)
GPIO.setup(shakeEncoderB, GPIO.IN)
GPIO.setup(modeOutputAPin, GPIO.OUT)
GPIO.setup(modeOutputBPin, GPIO.OUT)
GPIO.add_event_detect(shakeEncoderA, GPIO.RISING, callback=event_callback)

# PWMの設定
shakeMotorCwPWM = GPIO.PWM(shakeMotorCW, freq)
shakeMotorCcwPWM = GPIO.PWM(shakeMotorCCW, freq)

# カメラの初期設定
def CameraSettings():
    global labels, detectionFilterValue, picam2, nn

    # Caffeモデルを読み込む
    print('[Status] モデルを読み込んでいます...')
    nn = cv.dnn.readNetFromCaffe('model/SSD_MobileNet_prototxt', 'model/SSD_MobileNet.caffemodel')

    # カメラをpicamera2で読み込み
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (3280, 2464)}))
    picam2.start()

# カメラのモジュール
def CameraModule():
    global targetLabels, detectionFilterValue, targetAngle, nn, personFlag, picam2
    # ビデオをループする
    while True:
        frame = picam2.capture_array()  # カメラからフレームを取得する
        frame = imutils.resize(frame, width=400)  # フレームをリサイズする
        frame = cv.cvtColor(frame, cv.COLOR_RGBA2RGB)  # 色をRGBに変換する
        (h, w) = frame.shape[:2]  # フレームのサイズを取得する

        # フレームをBlobに変換する
        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        
        # BlobをCNNモデルに入力して推論を行う
        nn.setInput(blob)
        detections = nn.forward()

        # 検出されたオブジェクトに対してループ処理を行う
        for i in np.arange(0,detections.shape[2]):
            # 予測の信頼度を抽出する
            confidence = detections[0, 0, i, 2]

            # 信頼度が低い予測をフィルタリングする
            if confidence > detectionFilterValue:
                if labels[int(detections[0, 0, i, 1])] == "person":
                    # 人がいるときにフラグを立てる
                    personFlag = True
                    # オブジェクトの境界ボックスの座標を計算する
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, x2 = box.astype("int")[0], box.astype("int")[2]
                    center = int((x1+x2)/2)
                    
                    #角度を図で表示する
                    #中心下からcenterの位置まで線を引く
                    a1 = (int(w/2), h)
                    a2 = (center, int(h/2))
                    cv.line(frame, a1, a2, (0, 0, 255), 3)
                    cv.line(frame,(0, int(h/2)),(w, int(h/2)),(0, 0, 255), 2)
                    cv.line(frame,(center, 0),(center, h),(0, 0, 255), 2)
                    #角度を求めて表示,整数で表示
                    targetAngle = -int(np.arctan2(a2[1]-a1[1], a2[0]-a1[0]) * 180 / np.pi) - 90
                    cv.putText(frame, str(targetAngle), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), thickness=2)
                    break

        # フレームを表示する
        cv.imshow("Frame", frame)

        # キーが押されたら終了する
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# プロペラのモータ制御モジュール
def PropellerModule():
    global personFlag, modeOutputA, modeOutputB
    cnt = 0
    while True:
        if cnt%4 == 0:
            modeOutputA = 0
            modeOutputB = 0
        elif cnt%4 == 1:
            modeOutputA = 0
            modeOutputB = 1
        elif cnt%4 == 2:
            modeOutputA = 1
            modeOutputB = 0
        else:
            modeOutputA = 1
            modeOutputB = 1
        
        GPIO.output(modeOutputAPin, modeOutputA)
        GPIO.output(modeOutputBPin, modeOutputB)

        if keyboard.is_pressed('q'):
            break

        cnt = (cnt + 1) % 4
        time.sleep(1)
    
# 首振りのモータ制御モジュール
def ShakingModule():
    global targetAngle, nowAngle, M, M1, e, e1, e2, Kp, Ki, Kd
    # integral = 0  # 積分値を初期化
    while True:
        e2 = e1
        e1 = e
        e = targetAngle - nowAngle  # 偏差（e） = 目的値（goal） - 前回の操作量
        # integral += e  # 積分値を更新
        P = Kp * e  # P制御
        # I = Ki * integral  # I制御
        D = Kd * (e - e1)  # D制御
        G = P + D  # 制御量を計算
        # G = P + I + D  # 制御量を計算
        # if(targetAngle == nowAngle):
        #     integral = 0
        if(G > 0):
            shakeMotorCwPWM.start(G)
            shakeMotorCcwPWM.stop()
        else:
            shakeMotorCwPWM.stop()
            shakeMotorCcwPWM.start(-G)
            
        if keyboard.is_pressed('a'):
            shakeMotorCwPWM.start(50)
            shakeMotorCcwPWM.stop()
        elif keyboard.is_pressed('d'):
            shakeMotorCwPWM.stop()
            shakeMotorCcwPWM.start(50)
        elif keyboard.is_pressed('q'):
            break
        else:
            shakeMotorCwPWM.stop()
            shakeMotorCcwPWM.stop()

# カメラの初期設定
CameraSettings()

# スレッドの作成
cameraModule = threading.Thread(target=CameraModule)
propellerModule = threading.Thread(target=PropellerModule)
shakingModule = threading.Thread(target=ShakingModule)

# スレッドの開始
cameraModule.start()
propellerModule.start()
shakingModule.start()

cameraModule.join()
propellerModule.join()
shakingModule.join()

# 終了処理
shakeMotorCwPWM.stop()
shakeMotorCcwPWM.stop()
picam2.stop()
cv.destroyAllWindows()
GPIO.cleanup()
