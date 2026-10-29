"""Classroom Utilization Analyzer: data generation, EDA, K-Means, anomaly detection, scoring, export."""
import json, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
rng = np.random.default_rng(42)

# ---------- 1. Synthetic data ----------
caps = [30,30,40,40,40,60,60,60,60,80,80,80,100,100,120,120,40,60,60,80,30,50,70,90]
rooms = pd.DataFrame({"Room":[f"{'ABC'[i//8]}{101+i%8}" for i in range(24)],"Capacity":caps})
depts = ["IT","Computer","Electronics","Mechanical","Civil"]
subs = {"IT":["DBMS","OS","CN","AI"],"Computer":["DSA","TOC","ML","SE"],"Electronics":["VLSI","DSP","EMT"],"Mechanical":["Thermo","FM","CAD"],"Civil":["Struct","Survey","Geo"]}
slots = [9,10,11,12,13,14,15,16]
slot_w = {9:.85,10:.95,11:.85,12:.35,13:.7,14:.75,15:.6,16:.4}      # demand by hour
day_w = {0:.9,1:1,2:.95,3:.85,4:.6}                                   # Friday is weak
room_pop = rng.uniform(.35,1,24)                                       # how often a room is booked
dept_w = {"IT":.95,"Computer":.9,"Electronics":.75,"Mechanical":.65,"Civil":.7}
dates = pd.bdate_range("2026-08-03", periods=30)
rows=[]
for d in dates:
    for ri,r in rooms.iterrows():
        for h in slots:
            if rng.random() > room_pop[ri]*slot_w[h]*1.15: continue
            dep = rng.choice(depts, p=[.3,.25,.15,.15,.15])
            size = int(np.clip(rng.normal(55,22),12,130))             # class size independent of room -> mismatch
            att = rng.binomial(size, np.clip(dept_w[dep]*day_w[d.weekday()]*rng.uniform(.6,1.05),.05,1))
            if rng.random()<.03: att = int(att*rng.uniform(0,.25))     # cancellations / injected anomalies
            rows.append([d,d.day_name(),h,r.Room,r.Capacity,f"{dep[:2].upper()}-{rng.choice(list('ABC'))}",att,f"Prof {rng.choice(list('ABCDEFGH'))}",dep,rng.choice(subs[dep])])
df = pd.DataFrame(rows,columns=["Date","Day","Hour","Room","Capacity","Class","StudentsPresent","Faculty","Department","Subject"])

# ---------- 2. Cleaning + features ----------
df = df.drop_duplicates(["Date","Hour","Room"]).dropna()
df["Date"]=pd.to_datetime(df["Date"]); df["Week"]=df["Date"].dt.isocalendar().week.astype(int)-df["Date"].dt.isocalendar().week.min()+1
df["Occupancy"]=(df.StudentsPresent/df.Capacity*100).round(1)
df["Category"]=pd.cut(df.Occupancy,[-1,30,60,85,100,1e9],labels=["Underutilized","Low/Moderate","Efficient","Highly utilized","Overcrowded"])
df.assign(Date=df.Date.dt.strftime("%Y-%m-%d")).to_csv("../data/classroom_usage.csv",index=False)

# ---------- 3. Room-level metrics ----------
total_slots = len(dates)*len(slots)
g = df.groupby("Room")
rs = g.agg(Capacity=("Capacity","first"),Sessions=("Occupancy","size"),AvgOcc=("Occupancy","mean"),MaxOcc=("Occupancy","max"),
           AvgAtt=("StudentsPresent","mean"),P90Att=("StudentsPresent",lambda s:s.quantile(.9)),Empty=("StudentsPresent",lambda s:(s<=0.1*s.max()).sum())).reset_index()
peak = df[df.Hour.isin([10,11])].groupby("Room").Occupancy.mean().rename("PeakOcc")
rs = rs.merge(peak,on="Room",how="left").fillna(0)
rs["Frequency"]=(rs.Sessions/total_slots*100)
rs["CapMatch"]=(100-(rs.AvgOcc-80).abs()*1.25).clip(0,100)
rs["Efficiency"]=(0.5*rs.AvgOcc.clip(upper=100)+0.3*rs.Frequency/rs.Frequency.max()*100+0.2*rs.CapMatch).round(1)

# ---------- 4. K-Means ----------
X = StandardScaler().fit_transform(rs[["AvgOcc","MaxOcc","Sessions","Empty","AvgAtt","PeakOcc"]])
km = KMeans(3,n_init=10,random_state=42).fit(X); rs["c"]=km.labels_
order = rs.groupby("c").AvgOcc.mean().sort_values().index.tolist()
names = {order[0]:"Underutilized",order[1]:"Moderate",order[2]:"Highly utilized"}
rs["Cluster"]=rs.c.map(names)

# ---------- 5. Anomalies ----------
df["RoomMean"]=df.groupby("Room").Occupancy.transform("mean"); df["Dev"]=df.Occupancy-df.RoomMean
iso = IsolationForest(contamination=.02,random_state=42).fit(df[["Occupancy","Dev","Hour"]])
df["Anomaly"]=iso.predict(df[["Occupancy","Dev","Hour"]])==-1
an = df[(df.Anomaly)&(df.Dev<-35)&(df.RoomMean>45)].sort_values("Dev").head(12)

# ---------- 6. Recommendations ----------
def rec(r):
    if r.P90Att < .6*r.Capacity:
        c = rooms[(rooms.Capacity>=r.P90Att*1.15)&(rooms.Capacity<r.Capacity)].Capacity.min()
        return f"Capacity mismatch: 90% of sessions seat {r.P90Att:.0f} or fewer. Move to a {int(c)}-seat room." if c==c else "Capacity mismatch: no smaller room available; consider shared use."
    if r.AvgOcc>=75: return "High utilization. Avoid adding more classes."
    if r.AvgOcc<30: return "Underutilized. Release hours for other departments."
    return "Healthy usage. Monitor."
rs["Recommendation"]=rs.apply(rec,axis=1)
rs["Status"]=pd.cut(rs.AvgOcc,[-1,30,60,85,100,1e9],labels=["Underutilized","Low/Moderate","Efficient","Highly utilized","Overcrowded"])
rs.drop(columns="c").round(1).to_csv("../data/room_summary.csv",index=False)

# ---------- 7. Export for dashboard ----------
days=["Monday","Tuesday","Wednesday","Thursday","Friday"]
heat=df.pivot_table(index="Day",columns="Hour",values="Occupancy",aggfunc="mean").reindex(days).round(1)
out=dict(
 kpi=dict(rooms=len(rs),sessions=len(df),avg=round(df.Occupancy.mean(),1),under=int((rs.Status=="Underutilized").sum()),
          over=int((df.Occupancy>100).sum()),mismatch=int(rs.Recommendation.str.startswith("Capacity").sum()),
          peak=int(df.groupby("Hour").Occupancy.mean().idxmax())),
 rooms=rs.drop(columns="c").round(1).to_dict("records"),
 heat=dict(days=days,hours=slots,vals=heat.values.tolist()),
 byDay=df.groupby("Day").Occupancy.mean().reindex(days).round(1).tolist(),
 byHour=df.groupby("Hour").Occupancy.mean().round(1).tolist(),
 byDept=df.groupby("Department").Occupancy.mean().round(1).sort_values(ascending=False).to_dict(),
 cats=df.Category.value_counts().to_dict(),
 anomalies=[dict(Date=r.Date.strftime("%d %b"),Day=r.Day[:3],Hour=int(r.Hour),Room=r.Room,Occ=r.Occupancy,Usual=round(r.RoomMean,1),Class=r.Class,Subject=r.Subject) for r in an.itertuples()])

# ---------- 8. Booking-assistant data ----------
wd={d:i for i,d in enumerate(days)}
avail={r:[[0]*8 for _ in range(5)] for r in rs.Room}
for (rm,dy,hr),n in df.groupby(["Room","Day","Hour"]).size().items(): avail[rm][wd[dy]][slots.index(hr)]=int(n)
out["avail"]=avail
out["subjAtt"]={s:dict(mean=round(g_.StudentsPresent.mean(),1),p75=round(g_.StudentsPresent.quantile(.75),1)) for s,g_ in df.groupby("Subject")}
out["deptSubs"]=subs
json.dump(out,open("../data/results.json","w"),default=int)
print(out["kpi"]); print(rs.Cluster.value_counts().to_dict()); print(len(an),"anomalies")
