#!/usr/bin/env python3
"""Run ReSeT with the ORTrack interface."""
import argparse,gc,importlib,json,os,random,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','data','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sequence')
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
    os.environ['ANTIUAV410_ROOT']=str(args.data.resolve())
    os.environ['RESET_OUTPUT_ROOT']=str(args.output.resolve().parent)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    sys.path.insert(0,str(args.repo.resolve()))
    import cv2
    import numpy as np
    import torch
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cv2.setNumThreads(0)
    params=importlib.import_module('lib.test.parameter.ortrack_reset').parameters('ortrack_reset')
    params.checkpoint=str(args.checkpoint.resolve());params.debug=0
    module=importlib.import_module('lib.test.tracker.ortrack_reset')
    sequences={p.parent.name:p.parent for p in (args.data/'test').glob('*/IR_label.json')}
    if args.sequence and args.sequence not in sequences:raise ValueError('Unknown sequence: '+args.sequence)
    names=[args.sequence] if args.sequence else sorted(sequences)
    if not names:raise ValueError('No test sequences found')
    args.output.mkdir(parents=True,exist_ok=True)
    if any((args.output/(n+'.txt')).exists() for n in names):raise FileExistsError('Use a new output directory')
    for name in names:
        seq=sequences[name]
        label=json.loads((seq/'IR_label.json').read_text(encoding='utf-8'))
        images=sorted(p for p in seq.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
        if not images or len(images)!=len(label['exist']):raise ValueError('Frame count mismatch: '+name)
        gt=label.get('gt_rect',label.get('gt'))
        def frame(path):
            image=cv2.imread(str(path))
            if image is None:raise ValueError('Unreadable image: '+str(path))
            return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        tracker=module.ORTrack(params,'antiuav410')
        tracker.initialize(frame(images[0]),{'init_bbox':list(gt[0])})
        boxes=[list(gt[0])]
        with torch.no_grad():
            for path in images[1:]:
                # 跟踪只使用首帧初始化框，不传入后续真值。
                boxes.append(list(tracker.track(frame(path))['target_bbox']))
        np.savetxt(args.output/(name+'.txt'),np.trunc(np.asarray(boxes)).astype(np.int64),fmt='%d',delimiter='\t')
        print(name, len(boxes),flush=True)
        del tracker
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
