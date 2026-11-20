import json,cv2,numpy as np
A=json.load(open('annotations.json'))
H,W=1378,1710
wall=np.zeros((H,W),np.uint8)
for l in A['walls']: cv2.polylines(wall,[np.array(l,np.int32)],False,255,3)
R=5
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*R+1,2*R+1))
free=(cv2.dilate(wall,k)==0).astype(np.uint8)
n,lab,st,_=cv2.connectedComponentsWithStats(free,connectivity=4)
outside=lab[2,2]
def comp(p): return lab[p[1],p[0]]
start=(283,178)
print('start comp',comp(start),'area',st[comp(start),4],'outside',outside)
for t in A['trampolines']: print(t['id'],t['center'],'comp',comp(t['center']),'area',st[comp(t['center']),4])
print('cookie',comp((1020,1250)),'goal-ish pts',[comp(p) for p in [(1000,1290),(1180,1250),(1120,1180)]])
# reachability through trampoline links (bidirectional? arrows are directed)
T={t['id']:t for t in A['trampolines']}
reach={comp(start)}; ch=True
while ch:
    ch=False
    for l in A['trampoline_links']:
        a,b=comp(T[l['from']]['center']),comp(T[l['to']]['center'])
        if a in reach and b not in reach: reach.add(b); ch=True
print('reachable comps',reach)
