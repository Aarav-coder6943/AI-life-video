import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import os
from moviepy.editor import VideoFileClip, AudioFileClip

# ── CONFIG ─────────────────────────────────────────────
W, H = 1280, 720
FPS = 30
RAW_VIDEO = "raw_video.mp4"
FINAL_VIDEO = "final_output.mp4"
MUSIC = "music.mp3"

writer = cv2.VideoWriter(RAW_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

# ── COLORS ─────────────────────────────────────────────
WHITE = (255,255,255)
GOLD = (255,200,80)
BLUE = (100,160,255)
RED = (255,80,80)
GREEN = (80,220,130)

# ── FONT ───────────────────────────────────────────────
def get_font(size):
    return ImageFont.truetype("DejaVuSans-Bold.ttf", size)

FONT_BIG = get_font(70)
FONT_MED = get_font(40)
FONT_SMALL = get_font(28)

# ── UTIL ───────────────────────────────────────────────
def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def ease(t):
    return t*t*(3-2*t)

def zoom_frame(img, scale):
    w, h = img.size
    new_w, new_h = int(w/scale), int(h/scale)
    img = img.crop(((w-new_w)//2,(h-new_h)//2,(w+new_w)//2,(h+new_h)//2))
    return img.resize((w,h), Image.LANCZOS)

def write(img):
    writer.write(pil_to_cv(img))

def text_center(draw, text, y, font, color):
    bbox = draw.textbbox((0,0), text, font=font)
    x = (W - (bbox[2]-bbox[0]))//2
    draw.text((x,y), text, font=font, fill=color)

# ── SCENE 1: BEGINNING ─────────────────────────────────
def scene_beginning():
    for f in range(FPS*4):
        t = f/(FPS*4)
        img = Image.new("RGB",(W,H),(0,0,0))
        draw = ImageDraw.Draw(img)

        alpha = int(ease(t)*255)
        text_center(draw,"It started normally.",H//2-20,FONT_MED,(alpha,alpha,alpha))

        img = zoom_frame(img, 1+0.2*t)
        write(img)

# ── SCENE 2: GRIND LOOP ────────────────────────────────
def scene_grind():
    words = ["School","Assignments","Exams","Repeat"]
    for f in range(FPS*5):
        img = Image.new("RGB",(W,H),(10,10,20))
        draw = ImageDraw.Draw(img)

        for i,w in enumerate(words):
            y = int((i*120 - f*4)%H)
            draw.text((W//2-100,y),w,font=FONT_MED,fill=WHITE)

        text_center(draw,"It never stops.",H//2,FONT_MED,RED)

        write(img)

# ── SCENE 3: OVERTHINKING ──────────────────────────────
def scene_overthinking():
    thoughts = ["What if I'm not enough?","Why can't I stop?","I'm trying."]
    for f in range(FPS*5):
        t = f/30
        img = Image.new("RGB",(W,H),(10,0,0))
        draw = ImageDraw.Draw(img)

        for i,txt in enumerate(thoughts):
            x = int(W//2 + 200*math.sin(t+i))
            y = int(H//2 + 150*math.cos(t+i))
            draw.text((x,y),txt,font=FONT_SMALL,fill=RED)

        text_center(draw,"My mind is loud.",H//2,FONT_MED,WHITE)
        write(img)

# ── SCENE 4: GUITAR BREAKTHROUGH ───────────────────────
def scene_guitar():
    for f in range(FPS*4):
        t = f/(FPS*4)
        img = Image.new("RGB",(W,H),(5,0,10))
        draw = ImageDraw.Draw(img)

        if t>0.3:
            text_center(draw,"Then one day...",H//2-80,FONT_MED,WHITE)
        if t>0.5:
            text_center(draw,"Everything clicked.",H//2,FONT_BIG,GOLD)

        write(img)

# ── SCENE 5: CODING ────────────────────────────────────
def scene_code():
    lines = ["def life():","  try: improve()","  except: try_again()"]
    for f in range(FPS*4):
        img = Image.new("RGB",(W,H),(0,20,0))
        draw = ImageDraw.Draw(img)

        for i,l in enumerate(lines):
            draw.text((100,200+i*60),l,font=FONT_MED,fill=GREEN)

        text_center(draw,"I build things.",80,FONT_MED,GREEN)
        write(img)

# ── SCENE 6: IDENTITY ──────────────────────────────────
def scene_identity():
    for f in range(FPS*5):
        t = f/(FPS*5)
        img = Image.new("RGB",(W,H),(0,0,20))
        draw = ImageDraw.Draw(img)

        text_center(draw,"Who am I?",H//2-50,FONT_MED,WHITE)

        if t>0.5:
            text_center(draw,"Still figuring it out.",H//2+20,FONT_SMALL,GOLD)

        img = zoom_frame(img,1+0.3*t)
        write(img)

# ── SCENE 7: FINAL ─────────────────────────────────────
def scene_final():
    for f in range(FPS*5):
        t = f/(FPS*5)
        img = Image.new("RGB",(W,H),(0,0,0))
        draw = ImageDraw.Draw(img)

        if t>0.3:
            text_center(draw,"Still trying.",H//2,FONT_BIG,GOLD)
        if t>0.6:
            text_center(draw,"And that's enough.",H//2+100,FONT_SMALL,WHITE)

        write(img)

# ── RENDER VIDEO ───────────────────────────────────────
print("Rendering...")
scene_beginning()
scene_grind()
scene_overthinking()
scene_guitar()
scene_code()
scene_identity()
scene_final()

writer.release()

# ── ADD MUSIC ──────────────────────────────────────────
video = VideoFileClip(RAW_VIDEO)
audio = AudioFileClip(MUSIC).subclip(0, video.duration)

final = video.set_audio(audio)
final.write_videofile(FINAL_VIDEO)

print("DONE:", FINAL_VIDEO)
