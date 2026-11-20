import math, sys, statistics
from PIL import Image, ImageDraw, ImageOps
im = ImageOps.exif_transpose(Image.open('/tmp/maze.jpg')).convert('RGB')
S=2
g = im.convert('L'); gp = g.load(); px = im.load()
THR=90
def med(v,k=15):
    return [statistics.median(v[max(0,i-k):i+k+1]) for i in range(len(v))]
# top line: per display-x, first dark y (full res scan) in y 45..110
top={}
for x in range(300,1741):
    for y in range(56*S,112*S):
        if gp[x*S,y]<THR: top[x]=y/S; break
xs=sorted(top); ys=med([top[x] for x in xs])
topfit=dict(zip(xs,ys))
left={}
for y in range(300,1436):
    for x in range(120*S,260*S):
        if gp[x,y*S]<THR: left[y]=x/S; break
yy=sorted(left); xx=med([left[y] for y in yy])
leftfit=dict(zip(yy,xx))
print(len(topfit),len(leftfit), min(ys),max(ys),min(xx),max(xx))
A,B=-6,18  # strip relative to first-dark (display px)
SH=22
# erase top line
for x,y0 in topfit.items():
    for X in range(x*S,x*S+S):
        for Y in range(int((y0+A)*S),int((y0+B)*S)):
            px[X,Y]=px[X,int((y0-6)*S)]
for y,x0 in leftfit.items():
    for Y in range(y*S,y*S+S):
        for X in range(int((x0+A)*S),int((x0+B)*S)):
            px[X,Y]=px[int((x0-8)*S),Y]
for Y in range(1280*S,1440*S):
    c=px[72*S,Y]
    for X in range(25*S,72*S): px[X,Y]=c
box=(30,56,1740,1434)
im=im.crop(tuple(v*S for v in box)); ox,oy=box[0]*S,box[1]*S
im.convert('RGB').resize((im.width//2,im.height//2),Image.LANCZOS).save('/tmp/work/maze_clean.jpg',quality=90)
import json
A=json.load(open('/tmp/work/annotations.json'))
ov=Image.new('RGBA',im.size,(0,0,0,0)); d=ImageDraw.Draw(ov)
red=(230,20,20,255); blue=(20,90,230,255)
Z=lambda p:(p[0]*S,p[1]*S)
T={t['id']:t for t in A['trampolines']}
def arrow(p,q,col,w=12,gap=0,h=60):
    (x1,y1),(x2,y2)=p,q
    ang=math.atan2(y2-y1,x2-x1);c,s=math.cos(ang),math.sin(ang)
    sx,sy=x1+gap*c,y1+gap*s; ex,ey=x2-gap*c,y2-gap*s
    d.line((sx,sy,ex-h*.5*c,ey-h*.5*s),fill=col,width=w)
    d.polygon([(ex+8*c,ey+8*s),(ex-h*c+h*.55*s,ey-h*s-h*.55*c),(ex-h*c-h*.55*s,ey-h*s+h*.55*c)],fill=col)
for t in A['trampolines']:
    x,y=Z(t['center']);r=t['radius']*S
    d.ellipse((x-r,y-r,x+r,y+r),fill=(230,20,20,90),outline=red,width=10)
for l in A['trampoline_links']:
    a,b=T[l['from']],T[l['to']]
    arrow(Z(a['center']),Z(b['center']),red,gap=(a['radius']+6)*S)
U={t['id']:t for t in A['tunnels']}
for t in A['tunnels']:
    x,y=Z(t['center']);rx,ry=t['rx']*S,t['ry']*S
    d.ellipse((x-rx,y-ry,x+rx,y+ry),fill=(20,90,230,50),outline=blue,width=10)
for l in A['tunnel_links']:
    a,b=U[l['from']],U[l['to']]
    d.line((*Z(a['center']),*Z(b['center'])),fill=blue,width=10)
out=Image.alpha_composite(im.convert('RGBA'),ov).convert('RGB')
out=out.resize((out.width//2,out.height//2),Image.LANCZOS)
out.save(sys.argv[1],quality=90)
print(out.size)
