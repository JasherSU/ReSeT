#!/usr/bin/env python3
"""Evaluate the saved AntiUAV410 or CST-AntiUAV test predictions."""
import argparse,csv,importlib.util,io,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
CST_THRESHOLDS={'ORTrack':0.4148160000000001,'ROMTrack':0.4489270000000001,'OSTrack':0.48340900000000003}

def official_evaluator():
    spec=importlib.util.spec_from_file_location('antiuav410_reference',ROOT/'src/antiuav410_evaluator.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    instance=object.__new__(module.ExperimentAntiUAV410)
    instance.nbins_iou=21;instance.nbins_ce=51
    return instance

def read_boxes(path):
    boxes=np.loadtxt(io.StringIO(path.read_text().replace(',',' ')),ndmin=2)
    if boxes.shape[1]!=4 or not np.isfinite(boxes).all():raise ValueError('Invalid boxes: '+str(path))
    return boxes

def cst_scores(path,count):
    values=[float(json.loads(line)['output_score']) for line in path.read_text().splitlines() if line.strip()]
    if not all(math.isfinite(x) for x in values):raise ValueError('Non-finite score: '+str(path))
    if len(values)==count-1:values.insert(0,float('inf'))
    if len(values)!=count:raise ValueError('Score count mismatch: '+str(path))
    return np.asarray(values)

def cst_metrics(boxes,label,scores,threshold):
    gt=np.asarray(label.get('gt',label.get('gt_rect')),dtype=np.float64)
    present=np.asarray(label['exist'],dtype=np.int64)>0
    tl=np.maximum(boxes[:,:2],gt[:,:2]);br=np.minimum(boxes[:,:2]+boxes[:,2:],gt[:,:2]+gt[:,2:])
    size=np.maximum(0,br-tl);intersection=size[:,0]*size[:,1]
    union=np.maximum(boxes[:,2],0)*np.maximum(boxes[:,3],0)+np.maximum(gt[:,2],0)*np.maximum(gt[:,3],0)-intersection
    iou=np.divide(intersection,union,out=np.zeros_like(union),where=union>0)
    errors=np.linalg.norm(boxes[:,:2]+.5*boxes[:,2:]-gt[:,:2]-.5*gt[:,2:],axis=1)
    predicted_present=scores>=threshold
    state=np.where(present,np.where(predicted_present,iou,0.),(~predicted_present).astype(float))
    visible_iou=np.where(predicted_present[present],iou[present],0.)
    return {'mSA':float(state.mean()),
            'S_AUC':float(np.mean([(visible_iou>v).mean() for v in np.linspace(0,1,21)])),
            'P20':float(np.mean(predicted_present[present] & (errors[present]<20.)))}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['AntiUAV410','CST-AntiUAV'],required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    labels={p.parent.name:json.loads(p.read_text(encoding='utf-8')) for p in (args.data/'test').glob('*/IR_label.json')}
    if not labels:raise ValueError('No test annotations found')
    official=official_evaluator() if args.dataset=='AntiUAV410' else None
    root=ROOT/'predictions'/args.dataset
    folders=sorted(p for tracker in root.iterdir() if tracker.is_dir() for p in tracker.iterdir() if p.is_dir())
    if not folders:raise ValueError('No prediction folders found')
    output=[]
    for folder in folders:
        files={p.stem:p for p in folder.glob('*.txt')}
        if set(files)!=set(labels):raise ValueError('Sequence membership mismatch: '+str(folder))
        rows=[]
        for name in sorted(labels):
            path=files[name];boxes=read_boxes(path);label=labels[name]
            if len(boxes)!=len(label['exist']):raise ValueError('Frame count mismatch: '+name)
            if official is not None:
                if 'gt_rect' not in label:label['gt_rect']=label['gt']
                sa=float(official.eval(boxes,label))
                gt=np.asarray([b if len(b)==4 else [0,0,0,0] for b in label['gt_rect']],dtype=float)
                curves=boxes.copy();curves[0]=gt[0]
                ious,errors=official._calc_metrics(curves,gt)
                success,precision=official._calc_curves(ious,errors)
                rows.append({'SA':sa,'S_AUC':float(success.mean()),'P20':float(precision[20])})
            else:
                scores=cst_scores(path.with_name(name+'_scores.jsonl'),len(boxes))
                rows.append(cst_metrics(boxes,label,scores,CST_THRESHOLDS[folder.parent.name]))
        metrics={'dataset':args.dataset,'tracker':folder.parent.name,'method':folder.name,'sequences':len(rows),
                 **{k:float(np.mean([r[k] for r in rows])) for k in rows[0]}}
        output.append(metrics);print(json.dumps(metrics),flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)

if __name__=='__main__':main()
