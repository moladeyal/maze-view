import json, cv2, numpy as np, math
from PIL import Image, ImageDraw, ImageFont
A=json.load(open('annotations.json'))
W=json.load(open('walls_raw.json')); O=json.load(open('openings.json')); P=json.load(open('patches.json'))
# ---- clean image with title fixed
im=cv2.imread('maze_clean.jpg'); H,Wd=im.shape[:2]
g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
bg=cv2.GaussianBlur(cv2.dilate(g,np.ones((15,15))),(0,0),25); d=1-g/bg
reg=np.zeros((H,Wd),np.uint8); reg[15:116,1262:]=1; reg[116:245,1352:]=1
mask=((d>0.03)&(reg>0)).astype(np.uint8)*255
mask=cv2.dilate(mask,np.ones((9,9),np.uint8))
# keep the A tunnel outline / corridor walls untouched
mask[:, :1262]=0
im=cv2.inpaint(im,mask,6,cv2.INPAINT_TELEA)
title={"text":"מבוך זאב","box":[1372,40,1692,118]}
x0,y0,x1,y1=title['box']
pil=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)); dr=ImageDraw.Draw(pil)
dr.rounded_rectangle((x0,y0,x1,y1),radius=10,fill=(250,250,245),outline=(40,40,40),width=3)
f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',54)
txt=title['text']
tb=dr.textbbox((0,0),txt,font=f,direction='rtl')
dr.text(((x0+x1)/2-(tb[2]+tb[0])/2,(y0+y1)/2-(tb[3]+tb[1])/2),txt,font=f,fill=(20,20,20),direction='rtl')
im=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
cv2.imwrite('maze_clean_t.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,90])
# ---- structured annotations
A['title']=title
ug={ "U-A":{"under":"A","ends":["A-left-1","A-right"],"polygon":[[540,145],[700,148],[900,152],[1100,142],[1255,129],[1247,153],[1100,170],[900,172],[700,172],[557,176]]},
     "U-B1":{"under":"B1","ends":["B1-left","B1-right"],"polygon":[[703,304],[740,306],[790,311],[835,316],[862,318],[876,319],[864,353],[835,351],[790,347],[740,342],[699,338]]},
     "U-BOX":{"under":"BOX","ends":["BOX-top","BOX-bottom"],"polygon":[[670,822],[702,817],[751,966],[691,974]]}}
A['underground_tunnels']=[{"id":k,"border":"dotted","placeholder_path":True,**v} for k,v in ug.items()]
ends={e for v in ug.values() for e in v['ends']}
A['openings']=[{"id":o['id'],"structure":o['structure'],"kind":"underground_tunnel_end" if o['id'] in ends else "room_entrance","outline_removed":o['cut']} for o in O]
AP={a['id']:p for a,p in zip(json.load(open('arrows.json')),json.load(open('arrow_paths.json')))}
for l in A['trampoline_links']:
    l['arrow_path']=AP.get(l['from']+'-'+l['to'])
A['pointer_arrows']=[{"id":k,"path":AP[k],"note":"label pointer drawn by the child; NOT a corridor wall"} for k in AP if k not in {l['from']+'-'+l['to'] for l in A['trampoline_links']}]
WATER=[[874,413],[847,432],[830,452],[823,462],[822,490],[819,549],[813,603],[822,655],[839,693],[855,709],[907,717],[933,707],[953,685],[973,653],[976,612],[972,562],[962,500],[952,457],[937,428],[920,418],[902,413]]
A['rooms']=[{"id":"R1","label":"water room (wave shapes)","type":"water","requires":"goggles","polygon":WATER,"note":"closed with straight lines across the corridor entrances"}]
A['items']=[{"id":"goggles","type":"swimming_goggles","icon":"G","center":[707,878],"located_in":"BOX","enables":"R1"}]
A['structure_names']={"BOX":"goggles room (rectangular room)","A":"tunnel 'קודם קחו את המשקפת'","B1":"small 'מקפצה' tunnel","R1":"water room"}
A['walls']=W
A['notes']={"walls":"thin centre-lines of the pencil corridor borders (open polylines [[x,y],...]), auto-traced ~2px tolerance; small gaps patched by hand",
 "underground_tunnels":"polygon path under the room is a straight placeholder between the two tunnel ends, not drawn by the child",
 "coords":"all pixels of maze.jpg / maze_clean.jpg (1710x1378)"}
json.dump(A,open('annotations.json','w'),ensure_ascii=False,separators=(',',':'))

def draw_water_and_goggles(img, S):
    ov=img.copy()
    cv2.fillPoly(ov,[(np.array(WATER)*S).astype(np.int32)],(255,200,120))
    img[:]=cv2.addWeighted(ov,0.45,img,0.55,0)
    cv2.polylines(img,[(np.array(WATER)*S).astype(np.int32)],True,(230,150,40),4,cv2.LINE_AA)
def draw_G(img,S):
    c=tuple((np.array(A['items'][0]['center'])*S).astype(int).tolist())
    cv2.circle(img,c,26*S//2*2,(60,200,60),-1,cv2.LINE_AA); cv2.circle(img,c,26*S//2*2,(20,90,20),5,cv2.LINE_AA)
    (tw,th),_=cv2.getTextSize('G',cv2.FONT_HERSHEY_DUPLEX,2.0,5)
    cv2.putText(img,'G',(c[0]-tw//2,c[1]+th//2),cv2.FONT_HERSHEY_DUPLEX,2.0,(255,255,255),5,cv2.LINE_AA)
# ---- overlay render
S=2; big=cv2.resize(im,None,fx=S,fy=S,interpolation=cv2.INTER_CUBIC); big=(big*0.8+255*0.2).astype(np.uint8)
Pn=lambda l:(np.array(l)*S).astype(np.int32)
draw_water_and_goggles(big,S)
for l in W: cv2.polylines(big,[Pn(l)],False,(190,60,0),2,cv2.LINE_AA)
def dotted(poly,col,step=14,r=3):
    pts=poly+[poly[0]]
    for a,b in zip(pts,pts[1:]):
        a=np.array(a,float)*S;b=np.array(b,float)*S;L=np.linalg.norm(b-a)
        for t in np.arange(0,L,step): c=a+(b-a)*t/L; cv2.circle(big,(int(c[0]),int(c[1])),r,col,-1,cv2.LINE_AA)
for v in ug.values(): dotted(v['polygon'],(0,140,255))
for o in O:
    kind='t' if o['id'] in ends else 'r'
    cv2.polylines(big,[Pn(o['cut'])],False,(120,120,120) if kind=='t' else (0,170,0),5 if kind=='t' else 4,cv2.LINE_AA)
Tm={t['id']:t for t in A['trampolines']}
for l in A['trampoline_links']:
    a,b=Tm[l['from']],Tm[l['to']]
    p=np.array(a['center'],float)*S;q=np.array(b['center'],float)*S
    u=(q-p)/np.linalg.norm(q-p); gap=(a['radius']+6)*S
    p2=p+u*gap; q2=q-u*gap
    cv2.arrowedLine(big,tuple(p2.astype(int)),tuple(q2.astype(int)),(0,0,230),7,cv2.LINE_AA,tipLength=0.07)
for t in A['trampolines']: cv2.circle(big,tuple((np.array(t['center'])*S).tolist()),t['radius']*S,(0,0,230),5,cv2.LINE_AA)
draw_G(big,S)
big=cv2.resize(big,(2000,int(big.shape[0]*2000/big.shape[1])),interpolation=cv2.INTER_AREA)
cv2.imwrite('maze_walls_s.jpg',big,[cv2.IMWRITE_JPEG_QUALITY,88])

# ---- annotations-only image
b2=cv2.resize(im,None,fx=S,fy=S,interpolation=cv2.INTER_CUBIC)
draw_water_and_goggles(b2,S)
for t in A['tunnels']:
    pts=(np.array(t['points'])*S).astype(np.int32)
    o=b2.copy(); cv2.fillPoly(o,[pts],(230,90,20)); b2[:]=cv2.addWeighted(o,0.22,b2,0.78,0)
    cv2.polylines(b2,[pts],True,(230,90,20),5,cv2.LINE_AA)
for v in ug.values(): dotted(v['polygon'],(0,140,255)) if False else None
for l in A['trampoline_links']:
    a,b=Tm[l['from']],Tm[l['to']]
    p=np.array(a['center'],float)*S;q=np.array(b['center'],float)*S
    u=(q-p)/np.linalg.norm(q-p); gap=(a['radius']+6)*S
    cv2.arrowedLine(b2,tuple((p+u*gap).astype(int)),tuple((q-u*gap).astype(int)),(0,0,230),9,cv2.LINE_AA,tipLength=0.07)
for t in A['trampolines']:
    c=tuple((np.array(t['center'])*S).astype(int).tolist())
    o=b2.copy(); cv2.circle(o,c,t['radius']*S,(0,0,230),-1); b2[:]=cv2.addWeighted(o,0.3,b2,0.7,0)
    cv2.circle(b2,c,t['radius']*S,(0,0,230),8,cv2.LINE_AA)
draw_G(b2,S)
b2=cv2.resize(b2,(1710,1378),interpolation=cv2.INTER_AREA)
cv2.imwrite('maze_marked.jpg',b2,[cv2.IMWRITE_JPEG_QUALITY,90])
