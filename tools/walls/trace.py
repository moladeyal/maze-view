import numpy as np, cv2, json
from skan import Skeleton, summarize
sk=np.load('sk.npy')
S=Skeleton(sk,keep_images=False)
df=summarize(S,separator='_')
lines=[]
for i in range(S.n_paths):
    pts=S.path_coordinates(i)[:,::-1]  # x,y
    # prune short terminal branches (type 1 = junction-endpoint)
    ln=len(pts)
    ctype=df.loc[i,'branch_type']
    if ctype==1 and ln<18: continue
    if ctype==0 and ln<25: continue  # isolated small
    ap=cv2.approxPolyDP(pts.astype(np.int32).reshape(-1,1,2),2.0,False).reshape(-1,2)
    lines.append(ap.tolist())
print(len(lines),sum(len(l) for l in lines))
json.dump(lines,open('walls_raw.json','w'))
im=cv2.imread('maze_clean.jpg')
ov=(im*0.55+255*0.45).astype(np.uint8)
for l in lines: cv2.polylines(ov,[np.array(l,np.int32)],False,(0,0,200),1,cv2.LINE_AA)
cv2.imwrite('walls_ov.png',ov)
