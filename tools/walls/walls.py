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
rect(1300,0,1710,148); rect(1352,148,1710,310)                            # handwriting top right
rect(900,1095,1140,1345)                         # cookie sticker
rect(1245,1090,1480,1230)                        # tape right
rect(695,1180,895,1340)                          # tape left
for t in A['trampolines']:
    cv2.circle(ex,tuple(t["center"]),58 if t["id"]=="T2" else 30,1,-1)
ex|=poly_inset(A['tunnels'][0]['points'],20)     # text in A
ex|=poly_inset(A['tunnels'][1]['points'],8)      # text in B1
rect(445,217,630,274)                            # minneha label bubble
rect(330,790,600,940)                            # lower-left label bubble (text+outline)
cv2.circle(ex,(750,614),22,1,-1); rect(1290,612,1328,670); rect(905,352,948,398); rect(630,215,665,255); rect(420,222,455,250); rect(205,40,245,100); rect(1290,50,1350,100); rect(555,45,600,95); rect(985,58,1010,70)
ex|=poly_inset([[640,835],[780,835],[780,965],[640,965]],10)  # dumbbell in box
# thin pointer arrows drawn by the kid: follow the actual pencil line between waypoints
from skimage.graph import route_through_array
cost=1.0/(0.03+np.clip(d,0,0.3))
arrows=json.load(open('arrows.json'))
arrow_paths=[]
for a in arrows:
    path=[]
    for p,q in zip(a['pts'],a['pts'][1:]):
        r,_=route_through_array(cost,(p[1],p[0]),(q[1],q[0]),fully_connected=True,geometric=True)
        path+=[(c,rr) for rr,c in r]
    arrow_paths.append(path)
    cv2.polylines(ex,[np.array(path,np.int32)],False,1,a.get('w',9))
    for e in (path[0],path[-1]): cv2.circle(ex,tuple(int(v) for v in e),12,1,-1)
json.dump([[list(map(int,p)) for p in pa[::6]] for pa in arrow_paths],open('arrow_paths.json','w'))
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
dele=np.zeros((H,W),np.uint8)
for (cx,cy,rx,ry) in json.load(open('deletions.json')):
    cv2.ellipse(dele,(cx,cy),(rx,ry),0,0,360,1,-1)
sk[dele>0]=0
for seg in json.load(open('patches.json')):
    cv2.polylines(sk,[np.array(seg,np.int32)],False,1,1)
for o in openings:
    for seg in o.get('add',[]):
        cv2.polylines(sk,[np.array(seg,np.int32)],False,1,1)
from heal import heal,prune
prot=[p for o in openings for p in (o['cut'][0],o['cut'][-1])]
sk=prune(sk)
sk,added,left=heal(sk,prot,[(200,0,440,330)])
json.dump({'added':[[list(map(int,a)),list(map(int,b))] for a,b in added],'left':left},open('heal_info.json','w'))
print('healed',len(added),'dangling',len(left))
np.save('sk.npy',sk); np.save('ex.npy',ex)
ov=im.copy(); ov[sk>0]=(0,0,255)
ov[(ex>0)&(sk==0)]=(ov[(ex>0)&(sk==0)]*0.6+np.array([0,80,0])).astype(np.uint8)
cv2.imwrite('sk_ov.png',ov)
