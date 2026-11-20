import cv2, numpy as np, json
from skimage.morphology import skeletonize
im=cv2.imread('maze_clean.jpg'); H,W=im.shape[:2]
g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
bg=cv2.GaussianBlur(cv2.dilate(g,np.ones((15,15))),(0,0),25)
d=1-g/bg
m=(d>0.14).astype(np.uint8)
m[:, :275]=((d>0.22)[:, :275]).astype(np.uint8)
m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3)))
k,lab,st,_=cv2.connectedComponentsWithStats(m,connectivity=8)
for i in range(1,k):
    if st[i,4]<150: m[lab==i]=0
A=json.load(open('annotations.json'))
ex=np.zeros((H,W),np.uint8)
def rect(x0,y0,x1,y1): cv2.rectangle(ex,(x0,y0),(x1,y1),1,-1)
def poly_inset(pts,inset):
    t=np.zeros((H,W),np.uint8); cv2.fillPoly(t,[np.array(pts,np.int32)],1)
    t=cv2.erode(t,np.ones((2*inset+1,2*inset+1),np.uint8)); return t
rect(0,0,205,1378); rect(0,0,430,22)            # stickers/left
rect(1345,0,1710,310)                            # handwriting top right
rect(900,1095,1140,1345)                         # cookie sticker
rect(1245,1090,1480,1230)                        # tape right
rect(695,1180,895,1340)                          # tape left
for t in A['trampolines']:
    cv2.circle(ex,tuple(t["center"]),32,1,-1)
ex|=poly_inset(A['tunnels'][0]['points'],20)     # text in A
ex|=poly_inset(A['tunnels'][1]['points'],8)      # text in B1
rect(445,217,630,274)                            # minneha label bubble
rect(330,790,600,940)                            # lower-left label bubble (text+outline)
rect(205,40,245,100); rect(1290,50,1350,100); rect(555,45,600,95); rect(985,58,1010,70)
ex|=poly_inset([[640,835],[780,835],[780,965],[640,965]],10)  # dumbbell in box
# thin pointer arrows drawn by the kid
arrows=[[(380,575),(430,545),(520,525),(600,512),(690,525)],
 [(1130,555),(1200,590),(1260,620),(1315,650)],
 [(772,905),(775,800),(768,700),(752,640),(745,615)],
 [(812,378),(805,440),(795,500)],
 [(950,165),(935,250),(915,390)],
 [(485,905),(500,960),(560,960),(612,922)]]
for a in arrows: cv2.polylines(ex,[np.array(a,np.int32)],False,1,16)
ex=cv2.dilate(ex,np.ones((3,3),np.uint8))
m[ex>0]=0
cv2.imwrite('mask_walls.png',255-m*255)
sk=skeletonize(m>0).astype(np.uint8)
openings=json.load(open('openings.json'))
def shorten(p,cut):
    p=[np.array(q,float) for q in p]
    def trim(p):
        d=cut; p=p[:]
        while len(p)>1:
            v=p[1]-p[0]; L=np.linalg.norm(v)
            if L>d: p[0]=p[0]+v/L*d; return p
            d-=L; p.pop(0)
        return p
    return trim(trim(p)[::-1])[::-1]
er=np.zeros((H,W),np.uint8)
for o in openings:
    p=shorten(o['cut'],5)
    cv2.polylines(er,[np.round(np.array(p)).astype(np.int32)],False,1,7)
sk[er>0]=0
for o in openings:
    for seg in o.get('add',[]):
        cv2.polylines(sk,[np.array(seg,np.int32)],False,1,1)
np.save('sk.npy',sk); np.save('ex.npy',ex)
ov=im.copy(); ov[sk>0]=(0,0,255)
ov[(ex>0)&(sk==0)]=(ov[(ex>0)&(sk==0)]*0.6+np.array([0,80,0])).astype(np.uint8)
cv2.imwrite('sk_ov.png',ov)
