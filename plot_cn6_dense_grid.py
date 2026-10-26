#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

SIGHTS=[("02",.2),("03",.3),("04",.4),("05",.5),("06",.6),("07",.7),("08",.8),("09",.9),("10",1.),("11",1.1),("12",1.2)]
LOSSES=[("00",0.),("10",.1),("20",.2),("30",.3),("40",.4),("50",.5),("60",.6),("70",.7),("80",.8),("90",.9),("99",.99)]
METHODS=["match","db_all","db_invisible"]
RADAR=[("04","30","s=.4"),("06","30","s=.6"),("08","30","s=.8"),("10","30","s=1.0"),("12","30","s=1.2"),("08","00","p=.0"),("08","50","p=.5"),("08","70","p=.7"),("08","90","p=.9")]
T95=2.7764451051977987

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--input-root",type=Path,default=Path("outputs/cn6_missing_information_sweep_500ep"))
    p.add_argument("--output-dir",type=Path,default=Path("outputs/cn6_missing_information_sweep_500ep/aggregate_dense_grid"))
    p.add_argument("--radar-only",action="store_true")
    return p.parse_args()

def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.labelsize":10,"xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8,"pdf.fonttype":42,"ps.fonttype":42,"savefig.dpi":400})

def condition(sc,lc): return f"sr{sc}_pl{lc}"
def avg(x): return float(np.mean(x))
def std(x): return float(np.std(x,ddof=1))
def ci(x): return T95*std(x)/math.sqrt(5)

def load_one(root,sc,sight,lc,loss,method,seed):
    c=condition(sc,lc); path=root/c/f"{method}_seed{seed}"/"evaluation_summary.json"
    if not path.is_file(): raise FileNotFoundError(path)
    d=json.loads(path.read_text(encoding="utf-8"))
    ok=(d.get("status")=="completed" and d.get("method")==method and int(d.get("training_seed",-1))==seed and int(d.get("checkpoint_episode",-1))==100000 and int(d.get("evaluation_episodes",-1))==500 and int(d.get("evaluation_seed",-1))==20270001 and math.isclose(float(d.get("sight_range")),sight) and math.isclose(float(d.get("packet_loss")),loss))
    if not ok: raise ValueError(f"Protocol mismatch: {path}")
    d.update(condition=c,summary_path=str(path)); return d

def load(root,radar_only):
    sl=dict(SIGHTS); pl=dict(LOSSES)
    points=[(sc,sl[sc],lc,pl[lc]) for sc,lc,_ in RADAR] if radar_only else [(sc,s,lc,l) for sc,s in SIGHTS for lc,l in LOSSES]
    rows=[load_one(root,sc,s,lc,l,m,k) for sc,s,lc,l in points for m in METHODS for k in range(5)]
    expected=(9 if radar_only else 121)*15
    if len(rows)!=expected: raise ValueError(f"Expected {expected}, got {len(rows)}")
    return rows

def vals(rows,c,m):
    found=sorted((r for r in rows if r["condition"]==c and r["method"]==m),key=lambda r:int(r["training_seed"]))
    if len(found)!=5: raise ValueError(f"Expected five seeds: {c}/{m}")
    return [float(r["mean_pol"]) for r in found]

def gaps(rows,c,other): return [a-b for a,b in zip(vals(rows,c,"db_invisible"),vals(rows,c,other))]

def ranks(x):
    order=sorted(range(len(x)),key=x.__getitem__); out=[0.]*len(x); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and math.isclose(x[order[j]],x[order[i]],abs_tol=1e-12): j+=1
        rank=(i+1+j)/2
        for k in range(i,j): out[order[k]]=rank
        i=j
    return out

def wilcoxon(diff):
    nz=[x for x in diff if not math.isclose(x,0,abs_tol=1e-12)]
    if not nz:return 0.,1.
    r=ranks([abs(x) for x in nz]); total=sum(r); pos=sum(a for a,x in zip(r,nz) if x>0); obs=min(pos,total-pos)
    extreme=0
    for bits in range(2**len(r)):
        candidate=sum(a for i,a in enumerate(r) if bits&(1<<i))
        extreme+=min(candidate,total-candidate)<=obs+1e-12
    return obs,extreme/(2**len(r))

def save(fig,out,name):
    for folder,suffix in [("figures_png","png"),("figures_pdf","pdf")]:
        d=out/folder; d.mkdir(parents=True,exist_ok=True)
        fig.savefig(d/f"{name}.{suffix}",bbox_inches="tight",facecolor="white")
    plt.close(fig)

def matrix(rows,other):
    return np.array([[avg(gaps(rows,condition(sc,lc),other)) for sc,_ in SIGHTS] for lc,_ in LOSSES])

def edges(x):
    x=np.asarray(x,float); mid=(x[:-1]+x[1:])/2
    return np.r_[x[0]-(mid[0]-x[0]),mid,x[-1]+(x[-1]-mid[-1])]

def heatmap(mat,out,name,label,max_abs=None):
    max_abs=max_abs or max(.01,float(np.max(np.abs(mat))))
    fig,ax=plt.subplots(figsize=(4.25,3.35))
    mesh=ax.pcolormesh(edges([x for _,x in SIGHTS]),edges([x for _,x in LOSSES]),mat,cmap="RdBu",norm=TwoSlopeNorm(vmin=-max_abs,vcenter=0,vmax=max_abs),shading="flat",edgecolors="white",linewidth=.35)
    ax.set_xticks([x for _,x in SIGHTS]); ax.set_yticks([x for _,x in LOSSES])
    ax.set_xticklabels([f"{x:.1f}" for _,x in SIGHTS]); ax.set_yticklabels([f"{x:.2f}" if x==.99 else f"{x:.1f}" for _,x in LOSSES])
    ax.set_xlabel("Sight range"); ax.set_ylabel("Packet loss")
    cb=fig.colorbar(mesh,ax=ax,pad=.025,fraction=.055); cb.set_label(label)
    fig.subplots_adjust(left=.14,right=.88,bottom=.16,top=.98); save(fig,out,name)

def combined(a,b,out):
    scale=max(.01,float(np.max(np.abs([a,b])))); norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale)
    fig,axes=plt.subplots(1,2,figsize=(7.25,3),sharex=True,sharey=True)
    for ax,mat,label in [(axes[0],a,"DB-Invisible − DB-All"),(axes[1],b,"DB-Invisible − MATCH")]:
        mesh=ax.pcolormesh(edges([x for _,x in SIGHTS]),edges([x for _,x in LOSSES]),mat,cmap="RdBu",norm=norm,shading="flat",edgecolors="white",linewidth=.3)
        ax.set_xlabel("Sight range"); ax.set_title(label,fontsize=9.5,pad=7)
        ax.set_xticks([.2,.4,.6,.8,1.,1.2]); ax.set_yticks([0,.2,.4,.6,.8,.99]); ax.set_yticklabels(["0.0","0.2","0.4","0.6","0.8","0.99"])
    axes[0].set_ylabel("Packet loss")
    cb=fig.colorbar(mesh,ax=axes,pad=.025,fraction=.035); cb.set_label("Mean paired ΔPOL")
    fig.subplots_adjust(left=.09,right=.88,bottom=.18,top=.88,wspace=.12); save(fig,out,"03_gap_heatmaps_shared_scale")

def radar(rows,out):
    records=[]; sl=dict(SIGHTS); pl=dict(LOSSES)
    for sc,lc,label in RADAR:
        c=condition(sc,lc)
        for other in ["db_all","match"]:
            d=gaps(rows,c,other); stat,p=wilcoxon(d)
            records.append({"condition":c,"axis_label":label,"sight_range":sl[sc],"packet_loss":pl[lc],"comparison":f"db_invisible_minus_{other}","mean_paired_pol_difference":avg(d),"ci95_half_width":ci(d),"wins":sum(x>0 for x in d),"losses":sum(x<0 for x in d),"wilcoxon_statistic":stat,"wilcoxon_exact_p":p})
    labels=[x[2] for x in RADAR]; angles=np.linspace(0,2*np.pi,9,endpoint=False); closed=np.r_[angles,angles[0]]
    fig,ax=plt.subplots(figsize=(4.3,4.05),subplot_kw={"projection":"polar"}); maximum=0
    for key,label,color in [("db_invisible_minus_db_all","DB-Invisible − DB-All","#4E79A7"),("db_invisible_minus_match","DB-Invisible − MATCH","#E15759")]:
        r=[x["mean_paired_pol_difference"] for x in records if x["comparison"]==key]; maximum=max(maximum,max(r)); rc=r+[r[0]]
        ax.plot(closed,rc,color=color,linewidth=1.8,label=label); ax.fill(closed,rc,color=color,alpha=.13)
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1); ax.set_xticks(angles); ax.set_xticklabels(labels)
    upper=max(.10,math.ceil(maximum*100)/100); ax.set_ylim(0,upper); ticks=np.arange(.02,upper+.001,.02)
    ax.set_yticks(ticks); ax.set_yticklabels([f"{x:.2f}" for x in ticks],fontsize=7.5); ax.set_rlabel_position(215)
    ax.grid(color="#C9C9C9",linewidth=.6); ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,1.18),ncol=2)
    fig.subplots_adjust(left=.1,right=.9,bottom=.07,top=.82); save(fig,out,"04_nine_condition_gap_radar")
    return records

def write_csv(path,rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def write_matrix(path,mat):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["packet_loss\\sight_range"]+[x for _,x in SIGHTS])
        for (_,loss),row in zip(LOSSES,mat): w.writerow([loss]+[f"{x:.9f}" for x in row])

def main():
    a=args(); a.output_dir.mkdir(parents=True,exist_ok=True); style(); rows=load(a.input_root,a.radar_only)
    radar_data=radar(rows,a.output_dir); write_csv(a.output_dir/"nine_condition_radar_values.csv",radar_data)
    if a.radar_only:
        print("Validated: 135/135 existing results"); print("Saved:",a.output_dir); return
    ma,mm=matrix(rows,"db_all"),matrix(rows,"match"); summary=[]; pair=[]
    for sc,s in SIGHTS:
        for lc,l in LOSSES:
            c=condition(sc,lc)
            for m in METHODS:
                x=vals(rows,c,m); summary.append({"condition":c,"sight_range":s,"packet_loss":l,"method":m,"mean_pol":avg(x),"pol_std":std(x),"pol_ci95_half_width":ci(x)})
            for other,name in [("db_all","invisible_minus_all"),("match","invisible_minus_match")]:
                d=gaps(rows,c,other); stat,p=wilcoxon(d); pair.append({"condition":c,"sight_range":s,"packet_loss":l,"comparison":name,"mean_paired_pol_difference":avg(d),"difference_ci95_half_width":ci(d),"wins":sum(x>0 for x in d),"losses":sum(x<0 for x in d),"wilcoxon_statistic":stat,"wilcoxon_exact_p":p,"seed_differences":";".join(f"seed{i}:{x:.9f}" for i,x in enumerate(d))})
    write_csv(a.output_dir/"grid_method_summary.csv",summary); write_csv(a.output_dir/"grid_pairwise_gaps.csv",pair)
    write_matrix(a.output_dir/"delta_pol_invisible_minus_all_matrix.csv",ma); write_matrix(a.output_dir/"delta_pol_invisible_minus_match_matrix.csv",mm)
    heatmap(ma,a.output_dir,"01_invisible_minus_all_heatmap","ΔPOL (DB-Invisible − DB-All)")
    heatmap(mm,a.output_dir,"02_invisible_minus_match_heatmap","ΔPOL (DB-Invisible − MATCH)")
    combined(ma,mm,a.output_dir)
    print("Validated: 1815/1815 results"); print("Saved:",a.output_dir)
if __name__=="__main__": main()
