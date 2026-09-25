#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First-pass elastic-equivalent E_cm reconstruction for 14O + alpha events.
Python 3.6+ compatible.

Version: static-ML-vertex-input-compatible-v2-paired-output

Workflow:
  * read detected vertex and event_<id>.dat points
  * exclude points near the vertex
  * transform to alpha-beta
  * force alpha > beam threshold into one beam cluster
  * split remaining points into two clusters
  * choose alpha cluster by proximity to mapped silicon direction
  * fit beam/alpha/recoil clusters in XYZ with PCA/TLS
  * compute beam-alpha lab angle
  * combine angle with previously reconstructed alpha kinetic energy
    to obtain elastic-equivalent E_cm

Usage:

    python3 alpha_cm_reconstruction_static_vertices.py \
  --points-dir reconstructed_points_and_plots_alphas/reconstructed_dat \
  --vertices static_candidate_alphas_apply/selected_vertices \
  --alpha-energy-csv alphas_analysis/alpha_energy/alpha_kinetic_energy.csv \
  --detector-map silicon_channel_positions_forward.csv \
  --elastic-mapping elastic_mapping.csv \
  --hough-lines-dir reconstructed_lines_alphas \
  --output-dir alphas_analysis/alpha_cm_compare_static_ml \
  --exclude-radius-mm 5 \
  --beam-alpha-min 180 \
  --entrance-z-mm -80 \
  --detector-y-offset 0 \
  --detector-z-offset 0 \
  --annotate-events \
  --verbose

For inelastic events with unknown Q, E_cm_elastic_equiv_MeV is only an
elastic-equivalent quantity.
"""
from __future__ import print_function
import argparse, csv, glob, math, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def clean_text(v):
    if v is None: return ''
    return str(v).replace('\r','').replace('^M','').strip()


def finite_float(v):
    try: x=float(clean_text(v))
    except Exception: return None
    return x if math.isfinite(x) else None


def discover_events(points_dir):
    rx=re.compile(r'^event_(\d+)\.dat$')
    out=[]
    for name in os.listdir(points_dir):
        m=rx.match(name)
        if m: out.append(int(m.group(1)))
    return sorted(out)


def read_points(points_dir,event_id):
    path=os.path.join(points_dir,'event_{0}.dat'.format(int(event_id)))
    rows=[]
    with open(path,'r') as f:
        for raw in f:
            s=raw.strip()
            if not s or s.startswith('#') or s.lower().startswith('x_fit'): continue
            p=s.replace(',',' ').split()
            if len(p)<3: continue
            try: rows.append((float(p[0]),float(p[1]),float(p[2])))
            except Exception: pass
    if not rows: return np.zeros((0,3),dtype=float)
    P=np.asarray(rows,dtype=float)
    return P[np.isfinite(P).all(axis=1)]


def read_vertices(path):
    """
    Read vertex inputs in several compatible formats.

    Supported inputs:

      1. Old Analysis.py / BMeleton CSV:
             event_id, vert_1_x, vert_1_y, vert_1_z

      2. Static ML selected-candidate CSV:
             event, cand_x, cand_y, cand_z
         or common variants such as:
             event_id, vertex_x_mm, vertex_y_mm, vertex_z_mm

      3. Directory of per-event vertex text files:
             event_<id>_vertex.txt
         with one line:
             x y z

    Returns:
        dict[event_id] = np.array([x,y,z])
    """
    out={}

    if os.path.isdir(path):
        rx=re.compile(r'^event_(\d+)_vertex\.txt$')
        for name in sorted(os.listdir(path)):
            m=rx.match(name)
            if not m:
                continue
            ev=int(m.group(1))
            fpath=os.path.join(path,name)
            try:
                with open(fpath,'r') as f:
                    raw=f.readline().strip()
                parts=raw.replace(',',' ').split()
                if len(parts)<3:
                    continue
                xyz=[finite_float(parts[0]),finite_float(parts[1]),finite_float(parts[2])]
                if any(x is None for x in xyz):
                    continue
                out[ev]=np.asarray(xyz,dtype=float)
            except Exception:
                continue
        print('[INFO] read vertices: {0} entries from text-file directory {1}'.format(len(out),path))
        return out

    if not os.path.isfile(path):
        raise IOError('vertex input does not exist: {0}'.format(path))

    event_keys=['event_id','event','Event','evt','event_num']
    coordinate_sets=[
        ('static_ml_cand_xyz',       ['cand_x','cand_y','cand_z']),
        ('static_ml_candidate_xyz',  ['candidate_x','candidate_y','candidate_z']),
        ('selected_xyz_mm',          ['selected_x_mm','selected_y_mm','selected_z_mm']),
        ('selected_xyz',             ['selected_x','selected_y','selected_z']),
        ('final_xyz_mm',             ['final_x_mm','final_y_mm','final_z_mm']),
        ('final_xyz',                ['final_x','final_y','final_z']),
        ('vertex_xyz_mm',            ['vertex_x_mm','vertex_y_mm','vertex_z_mm']),
        ('seed_xyz',                 ['seed_x','seed_y','seed_z']),
        ('old_analysis_vert_1_xyz',  ['vert_1_x','vert_1_y','vert_1_z']),
        ('plain_xyz',                ['x','y','z']),
    ]

    with open(path,'r',newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError('vertex CSV has no header: {0}'.format(path))

        fields=set([clean_text(x) for x in r.fieldnames])

        event_key=None
        for k in event_keys:
            if k in fields:
                event_key=k
                break
        if event_key is None:
            raise ValueError(
                'Could not find event column in vertex CSV. Tried: {0}. File: {1}'.format(
                    ', '.join(event_keys), path
                )
            )

        coord_label=None
        coord_keys=None
        for label,keys in coordinate_sets:
            if all(k in fields for k in keys):
                coord_label=label
                coord_keys=keys
                break
        if coord_keys is None:
            raise ValueError(
                'Could not find vertex coordinate columns in CSV {0}. '
                'Tried old Analysis columns vert_1_x/y/z, static ML columns cand_x/y/z, '
                'and selected/vertex/seed variants.'.format(path)
            )

        for row in r:
            try:
                ev=int(float(clean_text(row[event_key])))
            except Exception:
                continue

            xyz=[finite_float(row.get(coord_keys[0])),
                 finite_float(row.get(coord_keys[1])),
                 finite_float(row.get(coord_keys[2]))]
            if any(x is None for x in xyz):
                continue
            out[ev]=np.asarray(xyz,dtype=float)

    print('[INFO] read vertices: {0} entries from CSV {1}'.format(len(out),path))
    print('[INFO] vertex CSV format: event column = {0}, coordinate set = {1} ({2})'.format(
        event_key, coord_label, ','.join(coord_keys)
    ))
    return out



def read_detector_map(path,flip_x=False,flip_y=False):
    out={}
    with open(path,'r',newline='') as f:
        r=csv.DictReader(f)
        for row in r:
            try:
                ch=int(float(clean_text(row['SiChan'])))
                x=float(clean_text(row['x_mm'])); y=float(clean_text(row['y_mm'])); z=float(clean_text(row['z_mm']))
                if flip_x: x=-x
                if flip_y: y=-y
                out[ch]={'SiChan':ch,'DetectorID':int(float(clean_text(row['DetectorID']))),'Quadrant':clean_text(row['Quadrant']),'x':x,'y':y,'z':z}
            except Exception: pass
    return out


def read_alpha_energy_csv(path):
    """
    Read the previously reconstructed alpha-energy table.

    In addition to SiChan and E_alpha, preserve the original Morgan identity:
        Run, Subrun, OriginalEvent

    This lets the elastic gate be applied using the same identifiers as
    elastic_mapping.csv rather than relying on reconstructed event_id.
    """
    out={}
    with open(path,'r',newline='') as f:
        r=csv.DictReader(f)
        for row in r:
            try:
                ev=int(float(clean_text(row['event_id'])))
                ch=int(float(clean_text(row['SiChan'])))
                e=float(clean_text(row['E_kinetic_alpha_vertex_MeV']))
            except Exception:
                continue

            if not math.isfinite(e) or e<=0:
                continue

            run=subrun=original_event=None
            try:
                run=int(float(clean_text(row.get('Run'))))
                subrun=int(float(clean_text(row.get('Subrun'))))
                original_event=int(float(clean_text(row.get('OriginalEvent'))))
            except Exception:
                pass

            out.setdefault(ev,[]).append({
                'SiChan':ch,
                'Ealpha_MeV':e,
                'Run':run,
                'Subrun':subrun,
                'OriginalEvent':original_event
            })
    return out


def parse_bool_text(value):
    s=clean_text(value).lower()
    if s in ('true','1','yes','y','t'):
        return True
    if s in ('false','0','no','n','f'):
        return False
    return None


def read_elastic_mapping(path):
    """
    Read:
        Run,Subrun,Event,...,elastic

    Returns:
        gate[(Run,Subrun,Event)] = set([True/False])

    A set is retained because a small number of duplicated event rows can
    disagree. The conflict policy is handled explicitly in main().
    """
    out={}
    with open(path,'r',newline='') as f:
        r=csv.DictReader(f)
        required=('Run','Subrun','Event','elastic')
        if r.fieldnames is None:
            raise ValueError('Elastic mapping has no header')
        missing=[x for x in required if x not in r.fieldnames]
        if missing:
            raise ValueError(
                'Elastic mapping missing columns: {0}'.format(', '.join(missing))
            )

        for row in r:
            try:
                key=(
                    int(float(clean_text(row['Run']))),
                    int(float(clean_text(row['Subrun']))),
                    int(float(clean_text(row['Event'])))
                )
            except Exception:
                continue

            value=parse_bool_text(row.get('elastic'))
            if value is None:
                continue
            out.setdefault(key,set()).add(value)
    return out


def elastic_gate_decision(values, conflict_policy):
    """
    Return:
        True  -> keep as elastic
        False -> reject as nonelastic
        None  -> ambiguous/conflicting or missing
    """
    if not values:
        return None

    if values == set([True]):
        return True
    if values == set([False]):
        return False

    # Conflicting duplicate rows.
    if conflict_policy == 'any':
        return True if True in values else False
    if conflict_policy == 'all':
        return True if values == set([True]) else False

    # safest default: skip
    return None


def xyz_to_alpha_beta(P,v):
    P=np.asarray(P,dtype=float); v=np.asarray(v,dtype=float).reshape(1,3)
    R=P-v; x=R[:,0]; y=R[:,1]; z=R[:,2]
    a=np.mod(np.degrees(np.arctan2(z,x)),360.0)
    b=np.degrees(np.arctan2(y,np.sqrt(x*x+z*z)))
    return a,b


def one_alpha_beta(p,v):
    a,b=xyz_to_alpha_beta(np.asarray(p,dtype=float).reshape(1,3),v)
    return float(a[0]),float(b[0])


def circular_mean_deg(a):
    r=np.radians(np.asarray(a,dtype=float))
    return math.degrees(math.atan2(np.mean(np.sin(r)),np.mean(np.cos(r))))%360.0


def ab_centroid(a,b): return circular_mean_deg(a),float(np.mean(b))


def circ_delta(a,b):
    d=abs(float(a)-float(b))%360.0
    return min(d,360.0-d)


def ab_distance(a1,b1,a2,b2):
    return math.sqrt(circ_delta(a1,a2)**2+(float(b1)-float(b2))**2)


def embed_ab(a,b):
    r=np.radians(a)
    return np.column_stack([np.cos(r),np.sin(r),np.asarray(b,dtype=float)/90.0])


def kmeans_two(a,b,max_iter=100):
    if len(a)<2: return None
    X=embed_ab(a,b)
    c0=X[0].copy(); d=np.sum((X-c0.reshape(1,-1))**2,axis=1); c1=X[int(np.argmax(d))].copy()
    labels=np.zeros(len(X),dtype=int)
    for it in range(max_iter):
        d0=np.sum((X-c0.reshape(1,-1))**2,axis=1); d1=np.sum((X-c1.reshape(1,-1))**2,axis=1)
        new=(d1<d0).astype(int)
        if it>0 and np.all(new==labels): break
        labels=new
        if np.sum(labels==0)==0 or np.sum(labels==1)==0: return None
        c0=np.mean(X[labels==0],axis=0); c1=np.mean(X[labels==1],axis=0)
    return labels


def fit_line(P):
    P=np.asarray(P,dtype=float)
    if len(P)<2: return None
    c=np.mean(P,axis=0); X=P-c.reshape(1,3)
    try: u,s,vt=np.linalg.svd(X,full_matrices=False)
    except Exception: return None
    d=vt[0]; n=np.linalg.norm(d)
    if n<=0: return None
    d=d/n; t=np.dot(X,d); resid=X-np.outer(t,d); orth=np.sqrt(np.sum(resid*resid,axis=1))
    rms=math.sqrt(float(np.mean(orth*orth)))
    return c,d,rms


def orient(d,c,v,outward):
    d=np.asarray(d,dtype=float).copy(); radial=np.asarray(c)-np.asarray(v)
    dot=float(np.dot(d,radial))
    if outward and dot<0: d=-d
    if (not outward) and dot>0: d=-d
    return d


def angle_deg(u,v):
    u=np.asarray(u); v=np.asarray(v); nu=np.linalg.norm(u); nv=np.linalg.norm(v)
    if nu<=0 or nv<=0: return None
    c=float(np.dot(u,v)/(nu*nv)); c=max(-1.0,min(1.0,c))
    return math.degrees(math.acos(c))


def beam_path_from_entrance(vertex, beam_dir, entrance_z_mm):
    v=np.asarray(vertex,dtype=float)
    d=np.asarray(beam_dir,dtype=float)
    if abs(float(d[2]))<1.0e-8:
        raise ValueError('beam direction has near-zero z component')
    t=(float(entrance_z_mm)-float(v[2]))/float(d[2])
    if t>=0.0:
        raise ValueError('entrance-plane intersection is not upstream')
    entrance=v+t*d
    return abs(float(t)), entrance, float(t)



def parse_hough_lines_file(path):
    """
    Parse headers of the form:
      # Line i: n=j a=(ax,ay,az) b=(bx,by,bz)
      #         start=(sx,sy,sz) end=(ex,ey,ez)
    """
    lines=[]
    vec=r"\(\s*([+-]?[0-9.eE]+)\s*,\s*([+-]?[0-9.eE]+)\s*,\s*([+-]?[0-9.eE]+)\s*\)"
    pat=re.compile(
        r"#\s*Line\s*(\d+).*?n=(\d+).*?a=\s*"+vec+
        r".*?b=\s*"+vec+r".*?start=\s*"+vec+r".*?end=\s*"+vec,
        re.IGNORECASE
    )
    with open(path,'r') as f:
        for raw in f:
            m=pat.search(raw)
            if not m:
                continue
            g=m.groups()
            try:
                line_num=int(g[0]); n=int(g[1])
                a=np.array([float(g[2]),float(g[3]),float(g[4])],dtype=float)
                b=np.array([float(g[5]),float(g[6]),float(g[7])],dtype=float)
                start=np.array([float(g[8]),float(g[9]),float(g[10])],dtype=float)
                end=np.array([float(g[11]),float(g[12]),float(g[13])],dtype=float)
            except Exception:
                continue
            nb=np.linalg.norm(b)
            if nb<=0.0:
                b=end-start
                nb=np.linalg.norm(b)
            if nb<=0.0:
                continue
            lines.append({
                'line_num':line_num,'n':n,'a':a,'b':b/nb,
                'start':start,'end':end
            })
    return lines


def find_hough_file(lines_dir,event_id):
    direct=[
        os.path.join(lines_dir,'event_{0}'.format(event_id),'lines.txt'),
        os.path.join(lines_dir,'event_{0}_lines.txt'.format(event_id))
    ]
    for path in direct:
        if os.path.isfile(path):
            return path
    patterns=[
        os.path.join(lines_dir,'event_{0}'.format(event_id),'*.txt'),
        os.path.join(lines_dir,'*event_{0}*line*.txt'.format(event_id))
    ]
    for pat in patterns:
        found=sorted(glob.glob(pat))
        if found:
            return found[0]
    return None


def line_point_distance(point,line):
    p=np.asarray(point,dtype=float)
    a=np.asarray(line['a'],dtype=float)
    b=np.asarray(line['b'],dtype=float)
    b=b/np.linalg.norm(b)
    return float(np.linalg.norm(np.cross(p-a,b)))


def closest_points_on_lines(line1,line2):
    p1=np.asarray(line1['a'],dtype=float)
    p2=np.asarray(line2['a'],dtype=float)
    d1=np.asarray(line1['b'],dtype=float); d1=d1/np.linalg.norm(d1)
    d2=np.asarray(line2['b'],dtype=float); d2=d2/np.linalg.norm(d2)
    r=p1-p2
    aa=float(np.dot(d1,d1)); bb=float(np.dot(d1,d2)); cc=float(np.dot(d2,d2))
    dd=float(np.dot(d1,r)); ee=float(np.dot(d2,r))
    den=aa*cc-bb*bb
    if abs(den)<1.0e-10:
        t2=float(np.dot(p1-p2,d2))
        q1=p1
        q2=p2+t2*d2
    else:
        t1=(bb*ee-cc*dd)/den
        t2=(aa*ee-bb*dd)/den
        q1=p1+t1*d1
        q2=p2+t2*d2
    return q1,q2,float(np.linalg.norm(q1-q2))


def hough_vertex_from_lines(lines,max_pair_dist,line_tol):
    """
    Basic Hough-only vertex finder.

    Every sufficiently close pair seeds a closest-approach midpoint.  Candidates
    are ranked by number of supporting lines, total Hough point support, mean
    line-to-vertex distance, then pair separation.
    """
    candidates=[]
    for i in range(len(lines)):
        for j in range(i+1,len(lines)):
            q1,q2,sep=closest_points_on_lines(lines[i],lines[j])
            if (not math.isfinite(sep)) or sep>float(max_pair_dist):
                continue
            vertex=0.5*(q1+q2)
            support=[]
            dists=[]
            support_n=0
            for k,line in enumerate(lines):
                miss=line_point_distance(vertex,line)
                if miss<=float(line_tol):
                    support.append(k)
                    dists.append(miss)
                    support_n+=int(line.get('n',0))
            if len(support)>=2:
                candidates.append({
                    'vertex':vertex,
                    'support':support,
                    'support_n':support_n,
                    'mean_dist':float(np.mean(dists)),
                    'pair_sep':sep
                })
    if not candidates:
        return None
    candidates.sort(
        key=lambda c:(-len(c['support']),-c['support_n'],
                      c['mean_dist'],c['pair_sep'])
    )
    return candidates[0]


def orient_hough_outward(line,vertex):
    direction=np.asarray(line['b'],dtype=float)
    direction=direction/np.linalg.norm(direction)
    midpoint=0.5*(np.asarray(line['start'])+np.asarray(line['end']))
    radial=midpoint-np.asarray(vertex,dtype=float)
    if np.linalg.norm(radial)>0.0 and float(np.dot(direction,radial))<0.0:
        direction=-direction
    return direction


def choose_hough_alpha_line(lines,vertex,si_xyz,max_angle_deg):
    to_si=np.asarray(si_xyz,dtype=float)-np.asarray(vertex,dtype=float)
    norm=np.linalg.norm(to_si)
    if norm<=0.0:
        return None
    to_si=to_si/norm

    best=None
    for idx,line in enumerate(lines):
        direction=orient_hough_outward(line,vertex)
        angle=angle_deg(direction,to_si)
        if angle is None:
            continue
        if best is None or angle<best['angle']:
            best={
                'index':idx,'line':line,
                'dir':direction,'angle':angle
            }

    if best is None or best['angle']>float(max_angle_deg):
        return None
    return best


def choose_hough_beam_line(lines,vertex,exclude_index,max_beam_angle_deg):
    """
    Pick the remaining Hough line closest to the known +z beam direction.
    If no line is sufficiently beam-like, use +z exactly.
    """
    zhat=np.array([0.0,0.0,1.0],dtype=float)
    best=None

    for idx,line in enumerate(lines):
        if idx==exclude_index:
            continue
        direction=np.asarray(line['b'],dtype=float)
        direction=direction/np.linalg.norm(direction)
        if float(np.dot(direction,zhat))<0.0:
            direction=-direction

        angle=angle_deg(direction,zhat)
        if angle is None:
            continue

        miss=line_point_distance(vertex,line)
        score=angle+0.20*miss

        if best is None or score<best['score']:
            best={
                'line':line,'dir':direction,'angle':angle,
                'miss':miss,'score':score
            }

    if best is None or best['angle']>float(max_beam_angle_deg):
        return {
            'line':None,'dir':zhat,'angle':0.0,
            'miss':0.0,'fallback':True
        }

    best['fallback']=False
    return best


def hough_reconstruction_for_event(event_id,lines_dir,si_xyz,Ealpha,args):
    path=find_hough_file(lines_dir,event_id)
    if path is None:
        return None,'missing_hough_file'

    lines=parse_hough_lines_file(path)
    if len(lines)<2:
        return None,'too_few_hough_lines'

    vertex_candidate=hough_vertex_from_lines(
        lines,
        args.hough_max_pair_distance_mm,
        args.hough_vertex_line_tolerance_mm
    )
    if vertex_candidate is None:
        return None,'hough_vertex_failed'

    vertex=np.asarray(vertex_candidate['vertex'],dtype=float)

    alpha_sel=choose_hough_alpha_line(
        lines,vertex,si_xyz,args.hough_max_alpha_si_angle_deg
    )
    if alpha_sel is None:
        return None,'hough_alpha_line_failed'

    beam_sel=choose_hough_beam_line(
        lines,vertex,alpha_sel['index'],args.hough_max_beam_angle_deg
    )

    theta=angle_deg(beam_sel['dir'],alpha_sel['dir'])
    if theta is None:
        return None,'hough_invalid_angle'

    ecm=ecm_elastic(
        Ealpha,theta,args.projectile_mass_u,args.target_mass_u
    )
    if ecm is None or not math.isfinite(ecm):
        return None,'hough_invalid_ecm'

    try:
        beam_path,entrance_xyz,entrance_t=beam_path_from_entrance(
            vertex,beam_sel['dir'],args.entrance_z_mm
        )
    except Exception:
        # Required fallback behavior: use +z as beam if Hough beam is unusable.
        beam_sel={
            'line':None,
            'dir':np.array([0.0,0.0,1.0],dtype=float),
            'angle':0.0,
            'miss':0.0,
            'fallback':True
        }
        try:
            beam_path,entrance_xyz,entrance_t=beam_path_from_entrance(
                vertex,beam_sel['dir'],args.entrance_z_mm
            )
        except Exception:
            return None,'hough_invalid_entrance_intersection'

        theta=angle_deg(beam_sel['dir'],alpha_sel['dir'])
        ecm=ecm_elastic(
            Ealpha,theta,args.projectile_mass_u,args.target_mass_u
        )
        if ecm is None or not math.isfinite(ecm):
            return None,'hough_invalid_ecm_after_beam_fallback'

    return {
        'file':os.path.basename(path),
        'vertex':vertex,
        'vertex_support_lines':len(vertex_candidate['support']),
        'vertex_support_n':vertex_candidate['support_n'],
        'vertex_mean_dist':vertex_candidate['mean_dist'],
        'vertex_pair_sep':vertex_candidate['pair_sep'],
        'alpha_line_num':alpha_sel['line']['line_num'],
        'alpha_si_angle':alpha_sel['angle'],
        'alpha_dir':alpha_sel['dir'],
        'beam_line_num':'' if beam_sel['line'] is None
                        else beam_sel['line']['line_num'],
        'beam_fallback':int(bool(beam_sel['fallback'])),
        'beam_angle_plus_z':beam_sel['angle'],
        'beam_miss':beam_sel['miss'],
        'beam_dir':beam_sel['dir'],
        'theta':theta,
        'path_mm':beam_path,
        'entrance_xyz':entrance_xyz,
        'entrance_t':entrance_t,
        'ecm':ecm
    },None


def ecm_elastic(Ealpha,theta_deg,M,m):
    c=math.cos(math.radians(theta_deg)); c2=c*c
    if c2<=1e-12: return None
    return float(Ealpha)*(float(M)+float(m))/(4.0*float(M)*c2)


def write_csv(path,fields,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader()
        for r in rows: w.writerow(r)


def event_id_from_row(row):
    try:
        return int(float(clean_text(row.get('event_id'))))
    except Exception:
        return None


def unique_event_ids_from_rows(rows):
    seen=set()
    out=[]
    for row in rows:
        ev=event_id_from_row(row)
        if ev is None:
            continue
        if ev not in seen:
            seen.add(ev)
            out.append(ev)
    return out


def filter_rows_by_event_ids(rows, keep_event_ids):
    keep=set([int(x) for x in keep_event_ids])
    return [
        row for row in rows
        if event_id_from_row(row) in keep
    ]


def limit_event_ids(event_ids, max_events):
    event_ids=list(event_ids)
    if int(max_events) > 0 and int(max_events) < len(event_ids):
        return event_ids[:int(max_events)]
    return event_ids



def parse_args():
    p=argparse.ArgumentParser(description='Cluster alpha-beta tracks and reconstruct elastic-equivalent E_cm.')
    p.add_argument('--points-dir',required=True)
    p.add_argument('--vertices',required=True,
                   help='Vertex input: old Analysis CSV with event_id,vert_1_x/y/z; static ML selected CSV with event,cand_x/y/z; or selected_vertices directory')
    p.add_argument('--alpha-energy-csv',required=True)
    p.add_argument('--detector-map',required=True)
    p.add_argument(
        '--elastic-mapping',
        required=True,
        help='CSV containing Run,Subrun,Event,...,elastic; only elastic=True events are analyzed'
    )
    p.add_argument(
        '--elastic-conflict-policy',
        choices=('skip','any','all'),
        default='skip',
        help=(
            'How to handle duplicate elastic-gate rows that disagree. '
            'skip=reject ambiguous event (default), any=keep if any row is True, '
            'all=keep only if every row is True.'
        )
    )
    p.add_argument('--output-dir',default='alphas_analysis/alpha_cm')
    p.add_argument('--max-events',type=int,default=0,
                   help='Maximum discovered event_<id>.dat files to process before reconstruction. 0 means all.')
    p.add_argument('--max-output-events',type=int,default=0,
                   help='Maximum successful output events to write after optional ML/Hough pairing. 0 means all.')
    p.set_defaults(pair_method_outputs=True)
    p.add_argument('--pair-method-outputs',action='store_true',dest='pair_method_outputs',
                   help='When --hough-lines-dir is supplied, write only events successful in both ML and Hough outputs. Default: enabled.')
    p.add_argument('--no-pair-method-outputs',action='store_false',dest='pair_method_outputs',
                   help='Disable paired ML/Hough output filtering.')
    p.add_argument('--exclude-radius-mm',type=float,default=5.0)
    p.add_argument('--beam-alpha-min',type=float,default=180.0)
    p.add_argument(
        '--entrance-z-mm',
        type=float,
        default=-80.0,
        help='Entrance-wall z position [mm]; beam path is measured along fitted beam line from this plane to the vertex'
    )
    p.add_argument('--hough-lines-dir',default=None,
                   help='Optional Hough lines directory containing event_<id>/lines.txt')
    p.add_argument('--hough-max-pair-distance-mm',type=float,default=25.0)
    p.add_argument('--hough-vertex-line-tolerance-mm',type=float,default=15.0)
    p.add_argument('--hough-max-alpha-si-angle-deg',type=float,default=45.0)
    p.add_argument('--hough-max-beam-angle-deg',type=float,default=35.0)
    p.add_argument('--min-cluster-points',type=int,default=4)
    p.add_argument('--max-line-rms-mm',type=float,default=None)
    p.add_argument('--max-alpha-si-angle-deg',type=float,default=None)
    p.add_argument('--flip-detector-x',action='store_true')
    p.add_argument('--flip-detector-y',action='store_true')
    p.add_argument('--detector-y-offset',type=float,default=0.0)
    p.add_argument('--detector-z-offset',type=float,default=0.0)
    p.add_argument('--projectile-mass-u',type=float,default=14.0)
    p.add_argument('--target-mass-u',type=float,default=4.0)
    p.add_argument('--annotate-events',action='store_true')
    p.add_argument('--annotation-fontsize',type=float,default=5.0)
    p.add_argument('--verbose',action='store_true')
    return p.parse_args()


def main():
    args=parse_args()
    for k in ['points_dir','vertices','alpha_energy_csv','detector_map','elastic_mapping','output_dir']:
        setattr(args,k,os.path.abspath(getattr(args,k)))
    if args.hough_lines_dir:
        args.hough_lines_dir=os.path.abspath(args.hough_lines_dir)
    if not os.path.isdir(args.output_dir): os.makedirs(args.output_dir)
    vertices=read_vertices(args.vertices)
    energies=read_alpha_energy_csv(args.alpha_energy_csv)
    elastic_gate=read_elastic_mapping(args.elastic_mapping)
    detmap=read_detector_map(args.detector_map,args.flip_detector_x,args.flip_detector_y)

    if args.verbose:
        n_true=sum(1 for v in elastic_gate.values() if v == set([True]))
        n_false=sum(1 for v in elastic_gate.values() if v == set([False]))
        n_conflict=sum(1 for v in elastic_gate.values() if len(v) > 1)
        print('[INFO] elastic gate events: {0}'.format(len(elastic_gate)))
        print('[INFO] unambiguous elastic=True: {0}'.format(n_true))
        print('[INFO] unambiguous elastic=False: {0}'.format(n_false))
        print('[INFO] conflicting duplicate gates: {0}'.format(n_conflict))
        print('[INFO] elastic conflict policy: {0}'.format(args.elastic_conflict_policy))
    rows=[]; skipped=[]; hough_rows=[]; hough_skipped=[]
    event_list=discover_events(args.points_dir)
    n_discovered_events=len(event_list)
    if args.max_events is not None and int(args.max_events)>0:
        event_list=event_list[:int(args.max_events)]
        print('[INFO] max-events: processing first {0} of {1} discovered events'.format(len(event_list),n_discovered_events))
    for ev in event_list:
        v=vertices.get(ev)
        if v is None: skipped.append({'event_id':ev,'reason':'missing_vertex','detail':''}); continue
        elist=energies.get(ev,[])
        if not elist:
            skipped.append({'event_id':ev,'reason':'missing_alpha_energy','detail':''})
            continue

        # Recover the original Morgan identity from the alpha-energy output.
        # All rows for one reconstructed event should refer to the same source
        # Run/Subrun/OriginalEvent. Use the first complete identity.
        source_key=None
        for ei in elist:
            if (
                ei.get('Run') is not None and
                ei.get('Subrun') is not None and
                ei.get('OriginalEvent') is not None
            ):
                source_key=(
                    int(ei['Run']),
                    int(ei['Subrun']),
                    int(ei['OriginalEvent'])
                )
                break

        if source_key is None:
            skipped.append({
                'event_id':ev,
                'reason':'missing_original_event_identity',
                'detail':'alpha-energy CSV needs Run,Subrun,OriginalEvent'
            })
            continue

        gate_values=elastic_gate.get(source_key)
        gate_result=elastic_gate_decision(
            gate_values,
            args.elastic_conflict_policy
        )

        if gate_result is None:
            reason='elastic_gate_conflict' if gate_values else 'missing_elastic_gate'
            skipped.append({
                'event_id':ev,
                'reason':reason,
                'detail':'Run={0},Subrun={1},Event={2},values={3}'.format(
                    source_key[0],
                    source_key[1],
                    source_key[2],
                    sorted(list(gate_values)) if gate_values else []
                )
            })
            continue

        if gate_result is False:
            skipped.append({
                'event_id':ev,
                'reason':'nonelastic_gate',
                'detail':'Run={0},Subrun={1},Event={2}'.format(
                    source_key[0],
                    source_key[1],
                    source_key[2]
                )
            })
            continue

        P0=read_points(args.points_dir,ev)
        if len(P0)<3*args.min_cluster_points: skipped.append({'event_id':ev,'reason':'too_few_total_points','detail':len(P0)}); continue
        R=P0-v.reshape(1,3); dist=np.sqrt(np.sum(R*R,axis=1)); P=P0[dist>=args.exclude_radius_mm]
        if len(P)<3*args.min_cluster_points: skipped.append({'event_id':ev,'reason':'too_few_after_exclusion','detail':len(P)}); continue
        a,b=xyz_to_alpha_beta(P,v); beam_idx=np.where(a>args.beam_alpha_min)[0]; fwd_idx=np.where(a<=args.beam_alpha_min)[0]
        if len(beam_idx)<args.min_cluster_points: skipped.append({'event_id':ev,'reason':'too_few_beam_points','detail':len(beam_idx)}); continue
        if len(fwd_idx)<2*args.min_cluster_points: skipped.append({'event_id':ev,'reason':'too_few_forward_points','detail':len(fwd_idx)}); continue
        lab=kmeans_two(a[fwd_idx],b[fwd_idx])
        if lab is None: skipped.append({'event_id':ev,'reason':'forward_clustering_failed','detail':''}); continue
        f0=fwd_idx[lab==0]; f1=fwd_idx[lab==1]
        if len(f0)<args.min_cluster_points or len(f1)<args.min_cluster_points:
            skipped.append({'event_id':ev,'reason':'small_forward_cluster','detail':'{0},{1}'.format(len(f0),len(f1))}); continue
        c0=ab_centroid(a[f0],b[f0]); c1=ab_centroid(a[f1],b[f1])
        candidates=[]
        for ei in elist:
            det=detmap.get(ei['SiChan'])
            if det is None: continue
            si=np.array([det['x'],det['y']+args.detector_y_offset,det['z']+args.detector_z_offset],dtype=float)
            sa,sb=one_alpha_beta(si,v)
            candidates.append({'energy':ei['Ealpha_MeV'],'ch':ei['SiChan'],'det':det,'si':si,'sa':sa,'sb':sb,'d0':ab_distance(c0[0],c0[1],sa,sb),'d1':ab_distance(c1[0],c1[1],sa,sb)})
        if not candidates: skipped.append({'event_id':ev,'reason':'no_valid_silicon_candidate','detail':''}); continue
        best=None
        for c in candidates:
            for cid,dv in [(0,c['d0']),(1,c['d1'])]:
                if best is None or dv<best['d']:
                    best={'cid':cid,'d':dv,'c':c}
        if args.max_alpha_si_angle_deg is not None and best['d']>args.max_alpha_si_angle_deg:
            skipped.append({'event_id':ev,'reason':'alpha_si_mismatch_too_large','detail':best['d']}); continue
        if best['cid']==0: ai,ri=f0,f1; ac,rc=c0,c1
        else: ai,ri=f1,f0; ac,rc=c1,c0
        bf=fit_line(P[beam_idx]); af=fit_line(P[ai]); rf=fit_line(P[ri])
        if bf is None or af is None: skipped.append({'event_id':ev,'reason':'line_fit_failed','detail':''}); continue
        bc,bd0,brms=bf; ac3,ad0,arms=af
        bd=orient(bd0,bc,v,False); ad=orient(ad0,ac3,v,True)
        rrms=''; rd=['','','']
        if rf is not None:
            rc3,rd0,rr=rf; rd=orient(rd0,rc3,v,True); rrms=rr
        if args.max_line_rms_mm is not None and (brms>args.max_line_rms_mm or arms>args.max_line_rms_mm):
            skipped.append({'event_id':ev,'reason':'line_rms_too_large','detail':'beam={0:.3f},alpha={1:.3f}'.format(brms,arms)}); continue
        theta=angle_deg(bd,ad)
        if theta is None: skipped.append({'event_id':ev,'reason':'invalid_scattering_angle','detail':''}); continue

        try:
            beam_path_mm, entrance_xyz, entrance_t = beam_path_from_entrance(
                v, bd, args.entrance_z_mm
            )
        except Exception as exc:
            skipped.append({'event_id':ev,'reason':'invalid_beam_entrance_intersection','detail':str(exc)})
            continue

        ealpha=best['c']['energy']; ecm=ecm_elastic(ealpha,theta,args.projectile_mass_u,args.target_mass_u)
        if ecm is None or not math.isfinite(ecm): skipped.append({'event_id':ev,'reason':'invalid_cm_energy','detail':theta}); continue
        row={'event_id':ev,
             'Run':source_key[0],
             'Subrun':source_key[1],
             'OriginalEvent':source_key[2],
             'elastic_gate':True,
             'vertex_x_mm':v[0],'vertex_y_mm':v[1],'vertex_z_mm':v[2],
             'NumPointsOriginal':len(P0),'NumPointsAfterVertexExclusion':len(P),'NumBeamPoints':len(beam_idx),'NumAlphaPoints':len(ai),'NumRecoilPoints':len(ri),
             'beam_alpha_centroid_deg':circular_mean_deg(a[beam_idx]),'beam_beta_centroid_deg':float(np.mean(b[beam_idx])),
             'alpha_alpha_centroid_deg':ac[0],'alpha_beta_centroid_deg':ac[1],'recoil_alpha_centroid_deg':rc[0],'recoil_beta_centroid_deg':rc[1],
             'SiChan':best['c']['ch'],'DetectorID':best['c']['det']['DetectorID'],'Quadrant':best['c']['det']['Quadrant'],
             'Si_x_mm':best['c']['si'][0],'Si_y_mm':best['c']['si'][1],'Si_z_mm':best['c']['si'][2],
             'Si_alpha_deg':best['c']['sa'],'Si_beta_deg':best['c']['sb'],'AlphaSiMismatchDeg':best['d'],
             'beam_dir_x':bd[0],'beam_dir_y':bd[1],'beam_dir_z':bd[2],'alpha_dir_x':ad[0],'alpha_dir_y':ad[1],'alpha_dir_z':ad[2],
             'recoil_dir_x':rd[0],'recoil_dir_y':rd[1],'recoil_dir_z':rd[2],
             'BeamLineRMS_mm':brms,'AlphaLineRMS_mm':arms,'RecoilLineRMS_mm':rrms,
             'theta_alpha_lab_deg':theta,
             'beam_path_from_entrance_mm':beam_path_mm,
             'entrance_plane_z_mm':args.entrance_z_mm,
             'beam_entrance_x_mm':entrance_xyz[0],
             'beam_entrance_y_mm':entrance_xyz[1],
             'beam_entrance_z_mm':entrance_xyz[2],
             'beam_entrance_line_t_mm':entrance_t,
             'E_alpha_vertex_MeV':ealpha,'E_cm_elastic_equiv_MeV':ecm,
             'projectile_mass_units':args.projectile_mass_u,'target_mass_units':args.target_mass_u,
             'exclude_radius_mm':args.exclude_radius_mm,'beam_alpha_min_deg':args.beam_alpha_min,
             'detector_y_offset_mm':args.detector_y_offset,'detector_z_offset_mm':args.detector_z_offset}
        rows.append(row)

        if args.hough_lines_dir:
            si_xyz=np.array(
                [row['Si_x_mm'],row['Si_y_mm'],row['Si_z_mm']],
                dtype=float
            )
            hough_result,hough_reason=hough_reconstruction_for_event(
                ev,args.hough_lines_dir,si_xyz,ealpha,args
            )

            if hough_result is None:
                hough_skipped.append({
                    'event_id':ev,
                    'Run':source_key[0],
                    'Subrun':source_key[1],
                    'OriginalEvent':source_key[2],
                    'reason':hough_reason
                })
            else:
                hv=hough_result['vertex']
                entrance=hough_result['entrance_xyz']
                hough_rows.append({
                    'event_id':ev,
                    'Run':source_key[0],
                    'Subrun':source_key[1],
                    'OriginalEvent':source_key[2],
                    'elastic_gate':True,
                    'hough_vertex_x_mm':hv[0],
                    'hough_vertex_y_mm':hv[1],
                    'hough_vertex_z_mm':hv[2],
                    'hough_vertex_support_lines':hough_result['vertex_support_lines'],
                    'hough_vertex_support_n':hough_result['vertex_support_n'],
                    'hough_vertex_mean_line_distance_mm':hough_result['vertex_mean_dist'],
                    'hough_vertex_seed_pair_distance_mm':hough_result['vertex_pair_sep'],
                    'SiChan':row['SiChan'],
                    'DetectorID':row['DetectorID'],
                    'Quadrant':row['Quadrant'],
                    'Si_x_mm':row['Si_x_mm'],
                    'Si_y_mm':row['Si_y_mm'],
                    'Si_z_mm':row['Si_z_mm'],
                    'hough_alpha_line_num':hough_result['alpha_line_num'],
                    'hough_alpha_si_angle_deg':hough_result['alpha_si_angle'],
                    'hough_alpha_dir_x':hough_result['alpha_dir'][0],
                    'hough_alpha_dir_y':hough_result['alpha_dir'][1],
                    'hough_alpha_dir_z':hough_result['alpha_dir'][2],
                    'hough_beam_line_num':hough_result['beam_line_num'],
                    'hough_beam_fallback_plus_z':hough_result['beam_fallback'],
                    'hough_beam_angle_to_plus_z_deg':hough_result['beam_angle_plus_z'],
                    'hough_beam_vertex_miss_mm':hough_result['beam_miss'],
                    'hough_beam_dir_x':hough_result['beam_dir'][0],
                    'hough_beam_dir_y':hough_result['beam_dir'][1],
                    'hough_beam_dir_z':hough_result['beam_dir'][2],
                    'hough_theta_alpha_lab_deg':hough_result['theta'],
                    'E_alpha_vertex_MeV':ealpha,
                    'hough_E_cm_elastic_equiv_MeV':hough_result['ecm'],
                    'hough_beam_path_from_entrance_mm':hough_result['path_mm'],
                    'entrance_plane_z_mm':args.entrance_z_mm,
                    'hough_beam_entrance_x_mm':entrance[0],
                    'hough_beam_entrance_y_mm':entrance[1],
                    'hough_beam_entrance_z_mm':entrance[2],
                    'hough_file':hough_result['file']
                })

        if args.verbose:
            print('[OK] event={0} ch={1} theta={2:.3f} Ealpha={3:.4f} Ecm={4:.4f} mismatch={5:.3f}'.format(ev,row['SiChan'],theta,ealpha,ecm,row['AlphaSiMismatchDeg']))
    fields=['event_id','Run','Subrun','OriginalEvent','elastic_gate','vertex_x_mm','vertex_y_mm','vertex_z_mm','NumPointsOriginal','NumPointsAfterVertexExclusion','NumBeamPoints','NumAlphaPoints','NumRecoilPoints',
            'beam_alpha_centroid_deg','beam_beta_centroid_deg','alpha_alpha_centroid_deg','alpha_beta_centroid_deg','recoil_alpha_centroid_deg','recoil_beta_centroid_deg',
            'SiChan','DetectorID','Quadrant','Si_x_mm','Si_y_mm','Si_z_mm','Si_alpha_deg','Si_beta_deg','AlphaSiMismatchDeg',
            'beam_dir_x','beam_dir_y','beam_dir_z','alpha_dir_x','alpha_dir_y','alpha_dir_z','recoil_dir_x','recoil_dir_y','recoil_dir_z',
            'BeamLineRMS_mm','AlphaLineRMS_mm','RecoilLineRMS_mm','theta_alpha_lab_deg',
            'beam_path_from_entrance_mm','entrance_plane_z_mm',
            'beam_entrance_x_mm','beam_entrance_y_mm','beam_entrance_z_mm','beam_entrance_line_t_mm',
            'E_alpha_vertex_MeV','E_cm_elastic_equiv_MeV',
            'projectile_mass_units','target_mass_units','exclude_radius_mm','beam_alpha_min_deg','detector_y_offset_mm','detector_z_offset_mm']
    outcsv=os.path.join(args.output_dir,'alpha_cm_reconstruction.csv'); skipcsv=os.path.join(args.output_dir,'alpha_cm_skipped.csv')
    # Optional final pairing/limiting of method outputs.
    #
    # If Hough output is requested, the default behavior is to keep only events
    # that were reconstructed successfully by BOTH methods. This makes the
    # ML and Hough CSVs directly comparable before they ever enter the SRIM
    # comparison script.
    if args.hough_lines_dir and args.pair_method_outputs:
        ml_ids=unique_event_ids_from_rows(rows)
        hough_ids=unique_event_ids_from_rows(hough_rows)
        hough_set=set(hough_ids)
        common_ids=[ev for ev in ml_ids if ev in hough_set]
        common_ids=limit_event_ids(common_ids,args.max_output_events)

        before_ml=len(rows)
        before_hough=len(hough_rows)
        rows=filter_rows_by_event_ids(rows,common_ids)
        hough_rows=filter_rows_by_event_ids(hough_rows,common_ids)

        print('[INFO] paired ML/Hough output filtering enabled')
        print('[INFO] ML successful before pairing: {0}'.format(before_ml))
        print('[INFO] Hough successful before pairing: {0}'.format(before_hough))
        print('[INFO] common successful events written: {0}'.format(len(common_ids)))
        if args.max_output_events is not None and int(args.max_output_events)>0:
            print('[INFO] max-output-events requested: {0}'.format(int(args.max_output_events)))
    elif args.max_output_events is not None and int(args.max_output_events)>0:
        ml_ids=limit_event_ids(unique_event_ids_from_rows(rows),args.max_output_events)
        rows=filter_rows_by_event_ids(rows,ml_ids)
        if args.hough_lines_dir:
            hough_ids=limit_event_ids(unique_event_ids_from_rows(hough_rows),args.max_output_events)
            hough_rows=filter_rows_by_event_ids(hough_rows,hough_ids)
        print('[INFO] max-output-events requested: {0}'.format(int(args.max_output_events)))
        print('[INFO] ML successful events written after limit: {0}'.format(len(unique_event_ids_from_rows(rows))))
        if args.hough_lines_dir:
            print('[INFO] Hough successful events written after independent limit: {0}'.format(len(unique_event_ids_from_rows(hough_rows))))

    write_csv(outcsv,fields,rows); write_csv(skipcsv,['event_id','reason','detail'],skipped)
    hough_outcsv=os.path.join(
        args.output_dir,'alpha_cm_hough_reconstruction.csv'
    )
    hough_skipcsv=os.path.join(
        args.output_dir,'alpha_cm_hough_skipped.csv'
    )

    if args.hough_lines_dir:
        hough_fields=[
            'event_id','Run','Subrun','OriginalEvent','elastic_gate',
            'hough_vertex_x_mm','hough_vertex_y_mm','hough_vertex_z_mm',
            'hough_vertex_support_lines','hough_vertex_support_n',
            'hough_vertex_mean_line_distance_mm',
            'hough_vertex_seed_pair_distance_mm',
            'SiChan','DetectorID','Quadrant',
            'Si_x_mm','Si_y_mm','Si_z_mm',
            'hough_alpha_line_num','hough_alpha_si_angle_deg',
            'hough_alpha_dir_x','hough_alpha_dir_y','hough_alpha_dir_z',
            'hough_beam_line_num','hough_beam_fallback_plus_z',
            'hough_beam_angle_to_plus_z_deg','hough_beam_vertex_miss_mm',
            'hough_beam_dir_x','hough_beam_dir_y','hough_beam_dir_z',
            'hough_theta_alpha_lab_deg',
            'E_alpha_vertex_MeV','hough_E_cm_elastic_equiv_MeV',
            'hough_beam_path_from_entrance_mm','entrance_plane_z_mm',
            'hough_beam_entrance_x_mm','hough_beam_entrance_y_mm',
            'hough_beam_entrance_z_mm','hough_file'
        ]
        write_csv(hough_outcsv,hough_fields,hough_rows)
        write_csv(
            hough_skipcsv,
            ['event_id','Run','Subrun','OriginalEvent','reason'],
            hough_skipped
        )
    if rows:
        path=[r['beam_path_from_entrance_mm'] for r in rows]
        e=[r['E_cm_elastic_equiv_MeV'] for r in rows]
        th=[r['theta_alpha_lab_deg'] for r in rows]

        fig=plt.figure(figsize=(9,6)); ax=fig.add_subplot(111); ax.scatter(path,e,s=22,alpha=.8)
        #if args.annotate_events:
        #    for r in rows:
        #        ax.annotate(str(r['event_id']),(r['beam_path_from_entrance_mm'],r['E_cm_elastic_equiv_MeV']),
        #                    ha='center',va='center',fontsize=args.annotation_fontsize)
        ax.set_xlabel('Path length from entrance plane to reconstructed vertex [mm]')
        ax.set_ylabel(r'Center-of-mass energy $E_{cm}$ [MeV]')
        ax.set_ylim(0, 10)
        ax.set_xlim(90, 230)
        ax.set_title(r'$E_{cm}$ vs Reconstructed Vertex Position'+'\n'+'Static ML Candidate Selection')
        ax.grid(True,alpha=.25); fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir,'alpha_cm_energy_vs_beam_path.png'),dpi=170); plt.close(fig)

        if args.hough_lines_dir and hough_rows:
            hough_path=[
                r['hough_beam_path_from_entrance_mm']
                for r in hough_rows
            ]
            hough_ecm=[
                r['hough_E_cm_elastic_equiv_MeV']
                for r in hough_rows
            ]

            fig=plt.figure(figsize=(9,6))
            ax=fig.add_subplot(111)
            ax.scatter(hough_path,hough_ecm,s=22,alpha=.8)

            #if args.annotate_events:
            #    for r in hough_rows:
            #        ax.annotate(
            #            str(r['event_id']),
            #            (
            #                r['hough_beam_path_from_entrance_mm'],
            #                r['hough_E_cm_elastic_equiv_MeV']
            #            ),
            #            ha='center',
            #            va='center',
            #            fontsize=args.annotation_fontsize
            #        )

            ax.set_xlabel(
                'Path length from entrance plane to reconstructed vertex [mm]'
            )
            ax.set_ylabel(
                'Center-of-mass energy $E_{cm}$ [MeV]'
            )
            ax.set_title(
                r'$E_{cm}$ vs Reconstructed Vertex Position'+'\n'+'Hough-Line CLosest Approach'
            )
            ax.set_ylim(0, 10)
            ax.set_xlim(90, 230)
            ax.grid(True,alpha=.25)
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    args.output_dir,
                    'alpha_cm_hough_vs_beam_path.png'
                ),
                dpi=170
            )
            plt.close(fig)

        fig=plt.figure(figsize=(9,6)); ax=fig.add_subplot(111); ax.scatter(path,th,s=22,alpha=.8)
        if args.annotate_events:
            for r in rows:
                ax.annotate(str(r['event_id']),(r['beam_path_from_entrance_mm'],r['theta_alpha_lab_deg']),
                            ha='center',va='center',fontsize=args.annotation_fontsize)
        ax.set_xlabel('Beam path length from entrance plane [mm]')
        ax.set_ylabel('Alpha lab scattering angle [deg]')
        ax.set_title('Alpha lab scattering angle vs incoming-beam path length')
        ax.grid(True,alpha=.25); fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir,'alpha_lab_angle_vs_beam_path.png'),dpi=170); plt.close(fig)

    counts={}
    for r in skipped: counts[r['reason']]=counts.get(r['reason'],0)+1
    print('\nElastic-gated center-of-mass reconstruction complete')
    print('  successful events: {0}'.format(len(rows)))
    print('  output CSV: {0}'.format(outcsv)); print('  skipped CSV: {0}'.format(skipcsv))
    if rows:
        print('  E_cm plot: {0}'.format(os.path.join(args.output_dir,'alpha_cm_energy_vs_beam_path.png')))
        print('  angle plot: {0}'.format(os.path.join(args.output_dir,'alpha_lab_angle_vs_beam_path.png')))
    if args.hough_lines_dir:
        print('  Hough successful events: {0}'.format(len(hough_rows)))
        print('  Hough CSV: {0}'.format(hough_outcsv))
        print('  Hough skipped CSV: {0}'.format(hough_skipcsv))
        if hough_rows:
            print(
                '  Hough E_cm plot: {0}'.format(
                    os.path.join(
                        args.output_dir,
                        'alpha_cm_hough_vs_beam_path.png'
                    )
                )
            )
    if counts:
        print('  skipped/rejected:')
        for k in sorted(counts):
            print('    {0}: {1}'.format(k,counts[k]))

    if args.hough_lines_dir and hough_skipped:
        hough_counts={}
        for r in hough_skipped:
            hough_counts[r['reason']]=hough_counts.get(r['reason'],0)+1
        print('  Hough skipped/rejected:')
        for k in sorted(hough_counts):
            print('    {0}: {1}'.format(k,hough_counts[k]))

    return 0

if __name__=='__main__':
    sys.exit(main())