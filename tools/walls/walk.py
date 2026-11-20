import json,cv2,numpy as np
A=json.load(open('annotations.json'))
H,W=1378,1710
wall=np.zeros((H,W),np.uint8)
for l in A['walls']: cv2.polylines(wall,[np.array(l,np.int32)],False,255,3)
R=5
free=(cv2.dilate(wall,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*R+1,2*R+1)))==0).astype(np.uint8)
RM=int(__import__('sys').argv[1]) if len(__import__('sys').argv)>1 else 40
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*(RM+1)+1,2*(RM+1)+1))
wide=cv2.morphologyEx(free,cv2.MORPH_OPEN,k)
narrow=free&(1-wide)
n,lab,st,_=cv2.connectedComponentsWithStats(free,connectivity=4)
outside=lab[2,2]
rooms=(free&(lab!=outside)).astype(np.uint8)
ex=np.zeros((H,W),np.uint8)
for e in A.get('walkable_extra',[]):
    if 'polygon' in e: cv2.fillPoly(ex,[np.array(e['polygon'],np.int32)],1)
    if 'circle' in e: cv2.circle(ex,(e['circle'][0],e['circle'][1]),e['circle'][2],1,-1)
ex=(ex>0)&(cv2.dilate(wall,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*R+1,2*R+1)))==0)
walk=((narrow|rooms|ex)>0).astype(np.uint8)
start=(270,100)
n2,lab2,st2,_=cv2.connectedComponentsWithStats(walk,connectivity=4)
def comp(p): return lab2[p[1],p[0]]
T={t['id']:t for t in A['trampolines']}
reach={comp(start)}
print('start',comp(start),'walk at start',walk[start[1],start[0]])
ch=True
while ch:
    ch=False
    for l in A['trampoline_links']:
        a,b=comp(T[l['from']]['center']),comp(T[l['to']]['center'])
        print(l['from'],a,'->',l['to'],b) if False else None
        if a in reach and b not in reach and b>0: reach.add(b); ch=True
print('trampoline comps',{t:int(comp(T[t]['center'])) for t in T},'reach',[int(r) for r in reach])
print('cookie-ish',[int(comp(p)) for p in [(1020,1060),(1150,1200),(1190,1250),(1100,1330)]])
vis=cv2.imread('maze_clean_t.jpg'); vis=(vis*0.5+110).astype(np.uint8)
rm=np.isin(lab2,list(reach)); 
vis[walk>0]=(vis[walk>0]*0.5+np.array([80,80,230])*0.5).astype(np.uint8)   # unreachable (red)
vis[rm]=(vis[rm]*0.5+np.array([60,200,60])*0.5).astype(np.uint8)           # reachable (green)
cv2.circle(vis,start,6,(255,0,0),-1)
cv2.imwrite('reach.png',vis); np.save('walk.npy',walk)
