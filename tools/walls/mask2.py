import json,cv2,numpy as np,sys
A=json.load(open('annotations.json'))
H,W=1378,1710
wall=np.zeros((H,W),np.uint8)
for l in A['walls']: cv2.polylines(wall,[np.array(l,np.int32)],False,255,3)
dist=cv2.distanceTransform((wall==0).astype(np.uint8),cv2.DIST_L2,5)
R=5
for Dmax in (20,26,32):
    free=((dist>=R)).astype(np.uint8)
    n,lab,st,_=cv2.connectedComponentsWithStats(free,connectivity=4)
    outside=lab[2,2]
    ok=((dist>=R)&(dist<=Dmax)&(lab!=outside))
    print(Dmax,ok.sum())
    vis=cv2.cvtColor(cv2.imread('maze_clean_t.jpg'),cv2.COLOR_BGR2GRAY); vis=cv2.cvtColor((vis*0.5+120).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    vis[ok]=(vis[ok]*0.4+np.array([60,200,60])*0.6).astype(np.uint8)
    cv2.imwrite(f'walk{Dmax}.png',cv2.resize(vis,(1710,1378)))
