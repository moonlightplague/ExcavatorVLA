from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json, subprocess
out=Path(__file__).resolve().parent
folder=next((out/'raw').glob('run*/episode_*/images'))
font_file='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font=lambda sz:ImageFont.truetype(font_file,sz,index=2)
title,small=font(30),font(21)
size=(960,640)
cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x640','-r','20','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'bed_unload.mp4')]
proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
def make(head,sub):
 img=Image.new('RGB',size,'#172423')
 d=ImageDraw.Draw(img)
 d.text((24,14),head,font=title,fill='white')
 d.text((24,65),sub,font=small,fill='#BED5D1')
 return img
def send(img,n):
 raw=img.tobytes()
 for _ in range(n):proc.stdin.write(raw)
for i in range(141,190):
 img=make('\u7b2c\u4e00\u6597  |  \u94f2\u6597\u6253\u5f00\uff0c\u77f3\u5757\u843d\u5165\u8f66\u53a2','\u5de6\u4fa7\u4e3a\u8f66\u5916\u89c6\u89d2\uff0c\u53f3\u4fa7\u4e3a\u673a\u81c2\u89c6\u89d2   \u00b7   \u4eff\u771f\u539f\u901f')
 for k,cam in enumerate(('1','0')):
  frame=Image.open(folder/cam/('%06d.ppm'%i)).resize((470,470),Image.Resampling.LANCZOS)
  img.paste(frame,(8+474*k,116))
 send(img,2)
for i in range(199,237):
 img=make('\u540c\u4e00\u6b21\u5378\u6599  |  \u8f66\u53a2\u5185\u4fa7\u8fd1\u666f','\u56de\u653e\u901f\u5ea6 0.5\u00d7   \u00b7   \u53ef\u89c1\u5730\u677f\u4fdd\u6301\u539f\u4f4d\uff0c\u7c97\u9897\u7c92 MPM \u63a5\u89e6\u9762\u62ac\u9ad8 7 \u5398\u7c73')
 frame=Image.open(out/'closeups'/('%06d.png'%i))
 img.paste(frame.resize((714,536),Image.Resampling.LANCZOS),(123,103))
 send(img,4)
proc.stdin.close()
if proc.wait():raise SystemExit('ffmpeg failed')

