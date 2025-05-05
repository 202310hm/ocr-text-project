from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
import os
import subprocess

# Tesseractの設定
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
tessdata_dir = '/opt/homebrew/share/tessdata'

# popplerのパスを確認
try:
    poppler_path = subprocess.check_output(['which', 'pdftoppm']).decode().strip()
    poppler_path = os.path.dirname(poppler_path)
except subprocess.CalledProcessError:
    poppler_path = '/opt/homebrew/bin'  # macOSでのデフォルトパス

# PDFファイルのパス（絶対パスを使用）
pdf_path = os.path.abspath('pdf_folder/ND-820_catalog.pdf')

# PDFファイルの存在確認
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDFファイルが見つかりません: {pdf_path}")

# 出力用フォルダ
output_folder = Path('ocr_texts')
output_folder.mkdir(exist_ok=True)

try:
    # PDFを画像に変換（高解像度設定 + popplerパス指定）
    images = convert_from_path(
        pdf_path,
        dpi=600,
        poppler_path=poppler_path
    )
except Exception as e:
    print(f"PDFの変換中にエラーが発生しました: {str(e)}")
    print(f"popplerのパス: {poppler_path}")
    print("以下のコマンドを実行してpopplerを再インストールしてください：")
    print("brew reinstall poppler")
    raise

def preprocess_image(image):
    # PIL ImageをOpenCV形式に変換
    img = np.array(image)
    
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # ノイズ除去（軽度な設定）
    denoised = cv2.fastNlMeansDenoising(gray, h=3, templateWindowSize=7, searchWindowSize=21)
    
    # コントラスト強調（適度な設定）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 二値化（大津の方法）
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

# 各ページをOCR処理してテキスト出力
for i, image in enumerate(images):
    # 画像の前処理
    processed_image = preprocess_image(image)

    # 前処理後の画像を保存
    cv2.imwrite(str(output_folder / f'debug_page_{i+1}.png'), processed_image)
    
    # OCR実行（日本語 + 高精度モデル + 最適化された設定）
    text = pytesseract.image_to_string(
        processed_image,
        lang='jpn+eng',
        config=f'--tessdata-dir {tessdata_dir} --psm 6 --oem 1'
    )

    # テキストファイルとして保存
    output_file = output_folder / f'page_{i+1}.txt'
    with output_file.open('w', encoding='utf-8') as f:
        f.write(text)
