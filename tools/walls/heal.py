import numpy as np, cv2
from scipy import ndimage as ndi
from skan import Skeleton, summarize
def prune(sk,minlen=18,rounds=2):
    sk=sk.copy().astype(np.uint8)
    for _ in range(rounds):
        S=Skeleton(sk,keep_images=False); df=summarize(S,separator='_')
        for i in range(S.n_paths):
            if df.loc[i,'branch_type']==1 and len(S.path_coordinates(i))<minlen:
                pts=S.path_coordinates(i).astype(int)
                # drop the endpoint side but keep the junction pixel
                nb=ndi.convolve(sk,np.ones((3,3)),mode='constant')-1
                for y,x in pts:
                    if nb[y,x]<3: sk[y,x]=0
    return sk
def walk(sk,y,x,n=14):
    H,W=sk.shape; path=[(y,x)]; seen={(y,x)}; cur=(y,x)
    for _ in range(n):
        nxt=None
        for dy in(-1,0,1):
            for dx in(-1,0,1):
                q=(cur[0]+dy,cur[1]+dx)
                if (dy or dx) and 0<=q[0]<H and 0<=q[1]<W and sk[q] and q not in seen: nxt=q;break
            if nxt:break
        if not nxt:break
        seen.add(nxt);path.append(nxt);cur=nxt
    return path
def heal(sk,protect_pts,no_zones,prot_r=14,max_ee=45,max_et=28):
    sk=sk.copy().astype(np.uint8); H,W=sk.shape
    nb=ndi.convolve(sk,np.ones((3,3)),mode='constant')-1
    ends=[tuple(e) for e in np.argwhere((sk>0)&(nb==1))]
    info={}
    for (y,x) in ends:
        p=walk(sk,y,x,14)
        if len(p)<14: continue
        v=np.array([x-p[-1][1],y-p[-1][0]],float); v/=np.linalg.norm(v)+1e-9
        info[(y,x)]=(v,p)
    added=[]
    skp=sk>0
    for (y,x),(v,p) in info.items():
        if any((x-px)**2+(y-py)**2<prot_r**2 for px,py in protect_pts): continue
        if any(x0<=x<=x1 and y0<=y<=y1 for x0,y0,x1,y1 in no_zones): continue
        own=np.zeros((H,W),bool)
        for q in walk(sk,y,x,40): own[q]=True
        own=ndi.binary_dilation(own,iterations=5)
        best=None
        # 1) other dangling ends, facing us
        for (y2,x2),(v2,_) in info.items():
            if (y2,x2)==(y,x) or own[y2,x2]: continue
            w=np.array([x2-x,y2-y],float); L=np.linalg.norm(w)
            if L<2 or L>max_ee: continue
            c=float(v@w)/L; c2=float(v2@-w)/L
            if c<0.8 or c2<0.5: continue
            sc=L*(2-c)
            if best is None or sc<best[0]: best=(sc,(x2,y2))
        # 2) T-join onto a wall, tight alignment
        if best is None:
            y0,y1,x0,x1=max(0,y-max_et),min(H,y+max_et+1),max(0,x-max_et),min(W,x+max_et+1)
            ys,xs=np.nonzero(skp[y0:y1,x0:x1]); ys+=y0; xs+=x0
            for yy,xx in zip(ys,xs):
                if own[yy,xx]: continue
                w=np.array([xx-x,yy-y],float); L=np.linalg.norm(w)
                if L<2 or L>max_et: continue
                c=float(v@w)/L
                if c<0.93: continue
                sc=L*(2-c)
                if best is None or sc<best[0]: best=(sc,(xx,yy))
        if best: added.append(((x,y),best[1]))
    for a,b in added: cv2.line(sk,a,b,1,1)
    done={a for a,_ in added}
    left=[(int(x),int(y)) for (y,x) in info if (x,y) not in done]
    return sk,added,left
