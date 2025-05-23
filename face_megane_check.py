# cd /Users/x20016xx/userbility/face_recognition
# source ./bin/activate
# python3.9 face_megane_check.py

#以下pip3.9 音声認識に関するライブラリのインストール
# pip3.9 install speechrecognition
# pip3.9 install sounddevice
# pip3.9 install torch 
# pip3.9 install git+https://github.com/r9y9/nnmnkwii

# pip3.9 install ttslearn 
# brew install portaudio
# CPPFLAGS="-I$(brew --prefix)/include" LDFLAGS="-L$(brew --prefix)/lib" pip3.9 install pyaudio

#現段階の課題点
#音声認識中はカメラが止まること　（解決はほぼ不可　リアルタイムでの対話のサンプルがあればできるかも）
#コンボボックスでのメガネ切り替えができない点　音声でのみ切り替えられる
#　理由：音声でメガネを変えた場合　コンボボックスの表示を自動的に切り替えられないため
#　妥協案：音声のみで切り替えて　画面上には切り替えられるメガネのリストを表示
#　妥協案2：コンボボックスを取りやめて　ボタンでの選択式にする→リアルタイムである必要がないため　フラグを切り替えられたら良い

# 現在は妥協案2を採用しました

#変更点
#カメラの空白を画像に差し替えた
#音声認識の追加
#カメラ起動と画像認識を同時に開始するようにした
#画像保存できるようにした
#GUIの調整はしていないが色々追加した
#音声の種類を切り替えられるボタンは消しても良いと考えている
#現段階の課題点

#未実装
#メガネを戻る操作　→ 一個前の画像ナンバーを記憶して　戻してというとそのナンバーを用いて切り替え


# coding: UTF-8

# GUI構築のライブラリ
import PySimpleGUI as sg # PySimpleGUIをインポート

# 共通ライブラリ
import numpy as np  # numpyのインポート
import os  # OS依存の操作（Pathやフォルダ操作など）用ライブラリのインポート
import random

# 画像処理ライブラリ
import cv2  # OpenCV（python版）のインポート
import dlib  #Dlibのインクルード

#音声合成ライブラリ
import speech_recognition as sr #音声認識用
import sounddevice as sd  # 録音・再生系のライブラリ
import torch  # 深層学習のライブラリ
from ttslearn.pretrained import create_tts_engine  # 音声合成ライブラリ

#深層学習用
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ---- 大域変数 ----
landmark = 'なし'
tmp = 0
cam_event = 0
reco_txt = ''
display_size = (600, 360)  # ディスプレイサイズ
isOpened = 0  # カメラがオープンになっているかどうかのフラグ
isRun = 0  # 認識On/Offフラグ（Onの時に認識実行）
image_flag = 'AUCF-22A-181_86'
image_list = (['AUCF-22A-181_86',
	    	'AMTF-17A-047_97',
			'AMCF-22A-229_127',
			'AUMF-22A-171_94'])

IMAGE_PATH = "./megane/AUCF-22A-181_86.png"  # 画像パス
IMAGE_PATH2 = "./megane/AMTF-17A-047_97.png"  # 画像パス
IMAGE_PATH3 = "./megane/AMCF-22A-229_127.png"  # 画像パス
IMAGE_PATH4 = "./megane/AUMF-22A-171_94.png"  # 画像パス

# ---- GUI初期設定 ----
# Windowのサイズ (横, 縦) 単位ピクセル
WINDOW_SIZE = (600, 710)

# デザインテーマとフォントの設定
sg.theme('DarkGreen5')
FONT = "Any 16"

# フレーム設定
#音声認識した文を表示
FRAME_RECOG = sg.Frame(
    layout=[
        [
            sg.Text(
                "ここに音声認識結果が表示されます",
                font=("Ricty", 22),
                text_color="#000000",
                background_color="#eee8d5",
                size=(40, 1),
                key="-RECOG_TEXT-",
            ),
        ],
    ],
    title="音声認識結果",
    font=("Ricty", 20),
    element_justification="center",
)

#音声認識後の動作を表示
FRAME_RESULT = sg.Frame(
    layout=[
        [
            sg.Text(
                "ここに動作結果が表示されます",
                font=("Ricty", 22),
                text_color="#000000",
                background_color="#eee8d5",
                size=(40, 1),
                key="-RESULT_TEXT-",
            ),
        ],
    ],
    title="動作結果",
    font=("Ricty", 20),
    element_justification="center",
)

# 音声合成用
# 音声合成エンジン構築
PWG_ENGINE = create_tts_engine("multspk_tacotron2_hifipwg_jvs24k", device=DEVICE)

# 話者ID
DEFAULT_SPK = "jvs010"
SPK_ID = PWG_ENGINE.spk2id[DEFAULT_SPK]

# 話者IDから話者名へ変換する辞書
SPK2ID = {"jvs010": "話者01"}

# 話者選択用のフレーム
FONT = "Any 16"
FRAME_SPK = sg.Frame(
    layout=[
        [
            sg.Button("-メガネ01-", key="AUCF-22A-181_86", font=FONT, size=(12, 1)),
            sg.Button("-メガネ02-", key="AMTF-17A-047_97", font=FONT, size=(12, 1)),
            sg.Button("-メガネ03-", key="AMCF-22A-229_127", font=FONT, size=(12, 1)),
            sg.Button("-メガネ04-", key="AUMF-22A-171_94", font=FONT, size=(12, 1)),
        ],
        [
            sg.Text(
                "現在のメガネ：{}".format(image_flag),
                font=("Ricty", 15),
                key="IMG_NAME",
            )
        ],
    ],
    title="メガネ選択",
    font=("Ricty", 16),
    element_justification="center",
    title_location=sg.TITLE_LOCATION_TOP,
    key="-CHGIMG-",
)

# ウィンドウの部品とレイアウト
LAYOUT = [
    [sg.Text('1:', font=FONT), sg.Button('カメラを起動', font=FONT, key='camera'), sg.Button("終了", font=FONT, key="-EXIT-")],
    [sg.Image(filename='./no_camera.png', size=display_size, key='-input_image-')],
    [sg.Text('2:任意のメガネを選択', font=FONT)],
    [FRAME_SPK],
    [	
		sg.Input('./Save_img/output.jpg', key='outputFile'),
	   	sg.Button('画像保存', font=FONT, key='save'),
		sg.Button('メガネ表示切替', font=FONT, key='megane_bt'),
		sg.Button("音声認識", font=FONT, key="-SYNTH-"),
	],
    [FRAME_RECOG],
    [FRAME_RESULT],
]

# ウィンドウの生成
WINDOW = sg.Window("メガネ試着アプリ", LAYOUT, finalize=True, size=WINDOW_SIZE)


# ---------  画像処理関数群  ----------
# アイコンを読み込む関数
def load_icon(path, distance):
	icon = cv2.imread(path, -1)
	icon_height, _  = icon.shape[:2]
	icon = img_resize(icon, float(distance * 1.5/icon_height))
	icon_h, icon_w  = icon.shape[:2]
	
	return icon, icon_w, icon_h

# 画像をリサイズする関数
def img_resize(img, scale):
	h, w  = img.shape[:2]
	img = cv2.resize(img, (int(w*scale*0.45), int(h*scale*0.45)))
	return img

# 距離と顔の中心座標を計算
def calc_distance_and_pos(img, parts, isLandmarkON):
	# 確認(33が顔の中心位置)
	cnt = 0
	pos = None
	p1 = None
	distance = 0.0
	
	for i in parts:
		if (cnt == 0):
			# 顔の幅を測る時の始点
			p1 = i
		if (cnt == 14):
			# 顔の幅を計算
			distance = ((p1.x-i.x)**2 + (p1.y-i.y)**2)**0.5
		if (cnt == 29):
			pos = i # 顔の中心位置
		if (isLandmarkON == 'あり'):
			# 画像に点とテキストをプロット
			cv2.putText(img, str(cnt), (i.x, i.y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), thickness=1, lineType=cv2.LINE_8)
			cv2.circle(img, (i.x, i.y), 1, (255, 0, 0), -1)
		cnt = cnt + 1
	
	return distance, pos

# 画像を保存する関数
def save_image(img):
	date = datetime.now().strftime("%Y%m%d_%H%M%S")
	path = "./" + date + ".png"
	cv2.imwrite(path, img) # ファイル保存

# 画像を合成する関数
def merge_images(bg, fg_alpha, s_x, s_y):
	alpha = fg_alpha[:,:,3]  # アルファチャンネルだけ抜き出す(要は2値のマスク画像)
	alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) # grayをBGRに
	alpha = alpha / 255.0    # 0.0〜1.0の値に変換
	
	fg = fg_alpha[:,:,:3]
	
	f_h, f_w, _ = fg.shape # アルファ画像の高さと幅を取得
	b_h, b_w, _ = bg.shape # 背景画像の高さを幅を取得
	
	# 画像の大きさと開始座標を表示
	print("f_w:{} f_h:{} b_w:{} b_h:{} s({}, {})".format(f_w, f_h, b_w, b_h, s_x, s_y))
	
	bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] * (1.0 - alpha)).astype('uint8') # アルファ以外の部分を黒で合成
	bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] + (fg * alpha)).astype('uint8')  # 合成
	
	return bg
	
# 画像リサイズ関数（高さが指定した値になるようにリサイズ (アスペクト比を固定)）
def scale_to_height(img, height):
	h, w = img.shape[:2]
	width = round(w * (height / h))
	dst = cv2.resize(img, dsize=(width, height))
	
	return dst

# ---- 顔認識エンジンセット ----
detector = dlib.get_frontal_face_detector()  #Dlibに用意されている検出器をセット
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 学習済みファイル読み込み

# 音声認識＆音声合成
def speak_text(text):
    """音声合成をする関数"""

    # テキスト音声合成
    wav, sr = PWG_ENGINE.tts(text, spk_id=SPK_ID)

    # 音割れ防止
    wav = (wav / np.abs(wav).max()) * (np.iinfo(np.int16).max / 2 - 1)

    # 再生
    sd.play(wav.astype(np.int16), sr)
    sd.sleep(int(1000 * len(wav) / sr))
    
# 音声認識による分岐
#音声認識と内容による分岐
def speak():
	global reco_txt, image_flag, cam_event, isRun, tmp

	# マイクからの音声入力
	r = sr.Recognizer()
	with sr.Microphone() as source:
		audio = r.listen(source)  # 音声取得

	try:
        # 日本語でGoogle音声認識
        # textには音声認識結果が文字列として入っている
		text = r.recognize_google(audio, language="ja-JP")
	except sr.UnknownValueError:
		WINDOW["-RECOG_TEXT-"].Update("Google音声認識は音声を理解できませんでした。")
		speak_text("再度音声を入力してください")
	except sr.RequestError as e:
		WINDOW["-RECOG_TEXT-"].Update("Google音声認識からの結果を" "要求できませんでした;" " {0}".format(e))
	else:
        # 音声認識結果を表示
		WINDOW["-RECOG_TEXT-"].Update(format(text))
        #speak_text(text)

		#コマンド終了
		#ありがとうと終了でプログラムの終了
		if "ありがとう" in text or "終了" in text:
			print("プログラムを終了します。")
			speak_text("ありがとうございました。")
			speak_text("プログラムを終了します。")
			finalize()

		#起動して　カメラの入った言葉で　カメラが起動します
		if "起動して" in text or "カメラ" in text:
			cam_event = 1
			#カメラがONか判定
			if isOpened:
				print("カメラは起動されています")
				speak_text("カメラは起動されています")
				WINDOW["-RESULT_TEXT-"].Update("カメラは起動されています")
				#イベントのフラグを折る
				cam_event = 0

			else:
				print("カメラを起動します")
				speak_text("カメラを起動します")
				WINDOW["-RESULT_TEXT-"].Update("カメラを起動しました")


		#カメラが起動していない場合上記以外のコマンドは全て弾かれる
		elif isOpened == 0:
			print("カメラを起動してください")
			speak_text("カメラを起動してください"), WINDOW["-RESULT_TEXT-"].Update("カメラを起動してください")
	    
		#カメラが起動している場合
		elif isOpened == 1:
            #メガネを変える
			#1番メガネ
			if "1番" in text or "一番" in text:
				#メガネをかけている時（画像認識中）
				if isRun:
					print("メガネを変えます")
					speak_text("メガネを変えます"), WINDOW["-RESULT_TEXT-"].Update("メガネを" + image_flag +"に変えました")
					image_flag = image_list[0]
					tmp = 0
				else:
					print("メガネをかけてください")
					speak_text("メガネをかけてください"), WINDOW["-RESULT_TEXT-"].Update("メガネをかけてください")
			#2番メガネ
			elif "2番" in text or "二番" in text:
				#メガネをかけている時（画像認識中）
				if isRun:
					print("メガネを変えます")
					speak_text("メガネを変えます"), WINDOW["-RESULT_TEXT-"].Update("メガネを" + image_flag +"に変えました")
					image_flag = image_list[1]
					tmp = 1
				else:
					print("メガネをかけてください")
					speak_text("メガネをかけてください"), WINDOW["-RESULT_TEXT-"].Update("メガネをかけてください")
			#3番メガネ
			elif "3番" in text or "三番" in text:
				#メガネをかけている時（画像認識中）
				if isRun:
					print("メガネを変えます")
					speak_text("メガネを変えます"), WINDOW["-RESULT_TEXT-"].Update("メガネを" + image_flag +"に変えました")
					image_flag = image_list[2]
					tmp = 2
				else:
					print("メガネをかけてください")
					speak_text("メガネをかけてください"), WINDOW["-RESULT_TEXT-"].Update("メガネをかけてください")
			#4番メガネ
			elif "4番" in text or "四番" in text:
				#メガネをかけている時（画像認識中）
				if isRun:
					print("メガネを変えます")
					speak_text("メガネを変えます"), WINDOW["-RESULT_TEXT-"].Update("メガネを" + image_flag +"に変えました")
					image_flag = image_list[3]
					tmp = 3
				else:
					print("メガネをかけてください")
					speak_text("メガネをかけてください"), WINDOW["-RESULT_TEXT-"].Update("メガネをかけてください")

			#メガネをランダムで変更
			elif "ランダム"in text or "変えて" in text:
				#メガネをかけている時（画像認識中）
				if isRun:
					#現在の画像ナンバーと同じ場合はもう一度
					while True:
						ram = random.randrange(len(image_list))
						print(tmp, ram)
						if(ram != tmp):
							tmp = ram
							break
						else:
							continue

					image_flag = image_list[tmp]
					
					print("ランダムでメガネを変える")
					print("メガネをランダムで変えます")
					speak_text("メガネをランダムで変更します"), WINDOW["-RESULT_TEXT-"].Update("メガネをランダムに変えました")

				else:
					print("メガネをかけてください")
					speak_text("メガネをかけてください"), WINDOW["-RESULT_TEXT-"].Update("メガネをかけてください")

			#メガネのオン/オフ
			elif "外して" in text or "消して" in text:
				#メガネをかけている時（画像認識中）
				if isRun:
					print("メガネを外します")
					speak_text("メガネを外します"), WINDOW["-RESULT_TEXT-"].Update("メガネを外しました")
					isRun = 0
				#メガネをかけていない時
				else:
					print("メガネを付けていません")
					speak_text("メガネを付けていません"), WINDOW["-RESULT_TEXT-"].Update("メガネをかけていません")

			elif "つけて" in text or "かけて" in text:
				#メガネをかけていない時
				if isRun == 0:
					print("メガネをかけます")
					speak_text("メガネをかけます"), WINDOW["-RESULT_TEXT-"].Update("メガネををかけました")
					isRun = 1
				#メガネをかけている時（画像認識中）
				else:
					print("メガネを既にかけています")
					speak_text("メガネを既にかけています"), WINDOW["-RESULT_TEXT-"].Update("メガネを既にかけています")

			elif "保存" in text or "撮って" in text:
				print("Write image -> " + values['outputFile'])
				cv2.imwrite(values['outputFile'], outimg)  # OpenCVの画像書き出し関数を利用
				speak_text("写真を撮りました"), WINDOW["-RESULT_TEXT-"].Update("画像を" + values['outputFile'] +"に保存しました")

			#上記コマンドに当てはまらない場合
			else:
				print(text)
				reco_txt = text
				speak_text(text + "に関するコマンドがありません")
				WINDOW["-RESULT_TEXT-"].Update(text + "に関するコマンドがありません")
				# result_txt = text

#終了コマンド
def finalize():
    # Windowを閉じる
    WINDOW.close()
    
# イベントループ
while True:
	event, values = WINDOW.read(timeout=10)
	
	# windowを閉じるか 終了ボタンを押したら終了
	if event in (sg.WIN_CLOSED, "-EXIT-"):
		finalize()
		break
	
	if event == 'read': #「読み込み」ボタンが押されたときの処理
		# 画像の読み込み処理
		print("Read image = " + values['inputFilePath'])
		orig_img = cv2.imread(values['inputFilePath'])  # OpenCVの画像読み込み関数を利用
		# 画像が大きいのでサイズを1/2にする．shapeに画像のサイズ(row, column)が入っている
		#		height = orig_img.shape[0]  #shape[0]は行数（画像の縦幅）
		#		width = orig_img.shape[1]  #shape[1]は列数（画像の横幅）
		#		orig_img = cv2.resize(orig_img , (int(width/2), int(height/2)))  #OpenCVのresize関数
		# 表示用に画像を固定サイズに変更（大きい画像を入力した時に認識ボタンなどが埋もれないように）
		disp_img = scale_to_height(orig_img, display_size[1])
		# 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
		imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
		# ウィンドウへ表示
		WINDOW['-input_image-'].update(data=imgbytes)
	
	if event == 'camera':  #「カメラ」ボタンが押された時の処理
		print("Camera Open")
		WINDOW["-RESULT_TEXT-"].Update("カメラを起動し、" + image_flag + "を表示しています")
		cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する
		isOpened, orig_img = cap.read()
		if isOpened:  # 正常にフレームを読み込めたら
			fps = cap.get(cv2.CAP_PROP_FPS)
			width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			orig_img = cv2.resize(orig_img , (int(width/2), int(height/2)))
			height = orig_img.shape[0]  #shape[0]は行数（画像の縦幅）
			width = orig_img.shape[1]  #shape[1]は列数（画像の横幅）

			print("Frame size = " + str(orig_img.shape))
			# 表示用に画像を固定サイズに変更（大きい画像を入力した時に認識ボタンなどが埋もれないように）
			disp_img = scale_to_height(orig_img, display_size[1])
			# 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
			imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
			# ウィンドウへ表示
			WINDOW['-input_image-'].update(data=imgbytes)
		else:
			print("Cannot capture a frame image")

		#認識開始処理
		isRun = 1
		if isRun:
			print("Start recognition")

	if cam_event == 1:  #「カメラ」ボタンが押された時の処理
		print("Camera Open")
		cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する
		isOpened, orig_img = cap.read()
		if isOpened:  # 正常にフレームを読み込めたら
			fps = cap.get(cv2.CAP_PROP_FPS)
			width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			orig_img = cv2.resize(orig_img , (int(width/2), int(height/2)))
			height = orig_img.shape[0]  #shape[0]は行数（画像の縦幅）
			width = orig_img.shape[1]  #shape[1]は列数（画像の横幅）

			print("Frame size = " + str(orig_img.shape))
			# 表示用に画像を固定サイズに変更（大きい画像を入力した時に認識ボタンなどが埋もれないように）
			disp_img = scale_to_height(orig_img, display_size[1])
			# 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
			imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
			# ウィンドウへ表示
			WINDOW['-input_image-'].update(data=imgbytes)
		else:
			print("Cannot capture a frame image")

		#認識開始処理
		isRun = 1
		if isRun:
			print("Start recognition")

		cam_event = 0

	if event == 'megane_bt':
		#認識開始処理 ON/OFF
		isRun = (isRun + 1) % 2
		if isRun:
			print("Start recognition")
			WINDOW["-RESULT_TEXT-"].Update("メガネを表示します")
		else:
			print("Finish recognition")
			WINDOW["-RESULT_TEXT-"].Update("メガネを非表示します")

	if event in (
            "AUCF-22A-181_86",
	    	"AMTF-17A-047_97",
			"AMCF-22A-229_127",
			"AUMF-22A-171_94",
		):
			image_flag = event
			WINDOW["-RESULT_TEXT-"].Update("メガネを" + image_flag +"に変えました")
			

	if isOpened == 1:
		# ---- フレーム読み込み ----
		ret, orig_img = cap.read()
		if ret:  # 正常にフレームを読み込めたら
			orig_img = cv2.resize(orig_img , (int(orig_img.shape[1]/2), int(orig_img.shape[0]/2)))
			height = orig_img.shape[0]  #shape[0]は行数（画像の縦幅）
			width = orig_img.shape[1]  #shape[1]は列数（画像の横幅）
			if isRun:  # 認識実行中なら
				# ---- 顔認識実行 ----
				img = orig_img  #処理結果表示用の変数 img を用意して，orig_imgをコピー
				faces = detector(orig_img[:, :, ::-1])  #画像中の全てを探索して「顔らしい箇所」を検出
				if len(faces) > 0:  # 顔を見つけたら以下を処理する
					for face in faces:  #全ての顔 faces から一つの face を取り出して

						parts = predictor(orig_img, face).parts()  #顔パーツ推定
						distance, pos = calc_distance_and_pos( orig_img, parts, landmark )
						
						if (image_flag == 'AUCF-22A-181_86'):
							icon, icon_w, icon_h = load_icon(IMAGE_PATH, distance)
							if (pos != None) and (distance > 0.0):
								# icon画像の中心をランドマーク33番（鼻）に対応づける
								x = pos.x - int(icon_w/2)
								y = pos.y - int((icon_h*2)/3)
								if (0 <= y) and (y <= (height-int(icon_h))) and (0 <= x) and (x <= (width-int(icon_w))):
									# 画面の範囲内だったら画像を合成
									img = merge_images(img, icon, x, y)
									outimg = img
									WINDOW["IMG_NAME"].Update("現在のメガネ：" + image_flag)
						
						if (image_flag == 'AMTF-17A-047_97'):
							icon, icon_w, icon_h = load_icon(IMAGE_PATH2, distance)
							if (pos != None) and (distance > 0.0):
								# icon画像の中心をランドマーク33番（鼻）に対応づける
								x = pos.x - int(icon_w/2)
								y = pos.y - int((icon_h*2)/3)
								if (0 <= y) and (y <= (height-int(icon_h))) and (0 <= x) and (x <= (width-int(icon_w))):
									# 画面の範囲内だったら画像を合成
									img = merge_images(img, icon, x, y)
									outimg = img
									WINDOW["IMG_NAME"].Update("現在のメガネ：" + image_flag)

						if (image_flag == 'AMCF-22A-229_127'):
							icon, icon_w, icon_h = load_icon(IMAGE_PATH3, distance)
							if (pos != None) and (distance > 0.0):
								# icon画像の中心をランドマーク33番（鼻）に対応づける
								x = pos.x - int(icon_w/2)
								y = pos.y - int((icon_h*3)/4)
								if (0 <= y) and (y <= (height-int(icon_h))) and (0 <= x) and (x <= (width-int(icon_w))):
									# 画面の範囲内だったら画像を合成
									img = merge_images(img, icon, x, y)
									outimg = img
									WINDOW["IMG_NAME"].Update("現在のメガネ：" + image_flag)

						if (image_flag == 'AUMF-22A-171_94'):
							icon, icon_w, icon_h = load_icon(IMAGE_PATH4, distance)
							if (pos != None) and (distance > 0.0):
								# icon画像の中心をランドマーク33番（鼻）に対応づける
								x = pos.x - int(icon_w/2)
								y = pos.y - int((icon_h*2)/3)
								if (0 <= y) and (y <= (height-int(icon_h))) and (0 <= x) and (x <= (width-int(icon_w))):
									# 画面の範囲内だったら画像を合成
									img = merge_images(img, icon, x, y)
									outimg = img
									WINDOW["IMG_NAME"].Update("現在のメガネ：" + image_flag)

				# 表示用に画像を固定サイズに変更
				disp_img = scale_to_height(orig_img, display_size[1])
				# 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
				imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
				# ウィンドウへ表示
				WINDOW['-input_image-'].update(data=imgbytes)
			else:  # 認識フラグOffなら入力フレームをそのまま表示
				# 表示用に画像を固定サイズに変更
				disp_img = scale_to_height(orig_img, display_size[1])
				# 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
				imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
				# ウィンドウへ表示
				WINDOW['-input_image-'].update(data=imgbytes)
			
	if event == 'save': #「画像保存」ボタンが押されたときの処理
		if isOpened == 1:
			print("Write image -> " + values['outputFile'])
			cv2.imwrite(values['outputFile'], outimg)  # OpenCVの画像書き出し関数を利用
			WINDOW["-RESULT_TEXT-"].Update("画像を" + values['outputFile'] +"に保存しました")

	if event in ("-SYNTH-"):
		#WINDOW["-RESULT_TEXT-"].Update("話しかけてください")
		speak()

	


