"""500-call controlled parameter experiment. Prepare, resume, thermal gate, durable raw data."""
from __future__ import annotations
import argparse, collections, ctypes, fcntl, hashlib, itertools, json, math, os, platform, random, statistics, time
from datetime import datetime, timezone
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/parameter_500'
MODEL = 'qwen3:30b'
BASE = dict(temperature=0.5, top_p=0.9, top_k=40, min_p=0.0, repeat_penalty=1.0, num_ctx=4096, num_predict=256, seed=42)


def thermal():
    ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
    lib = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
    lib.objc_getClass.argtypes = [ctypes.c_char_p]; lib.objc_getClass.restype = ctypes.c_void_p
    lib.sel_registerName.argtypes = [ctypes.c_char_p]; lib.sel_registerName.restype = ctypes.c_void_p
    send = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', lib))
    info = send(lib.objc_getClass(b'NSProcessInfo'), lib.sel_registerName(b'processInfo'))
    integer = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', lib))
    value = integer(info, lib.sel_registerName(b'thermalState'))
    if value not in range(4):
        raise RuntimeError(f'Invalid thermal state: {value}')
    return value


def tasks():
    definitions = [
        ('extract', '只输出 JSON 对象：客户月收入8000元、月负债5000元、近六月逾期2次。字段 income、debt、overdue，值为整数。', {'income':8000,'debt':5000,'overdue':2}),
        ('missing', '只输出 JSON，提取 income、debt、children。月收入6000元、子女0人，未提供的信息必须为null，不猜测。', {'income':6000,'debt':None,'children':0}),
        ('rule_high', '仅按本题规则判定：负债/月收入>=0.5为high，否则low。收入8000、月负债5000。只输出JSON字段risk。', {'risk':'high'}),
        ('rule_low', '仅按本题规则判定：负债/月收入>=0.5为high，否则low。收入10000、月负债2000。只输出JSON字段risk。', {'risk':'low'}),
        ('conversion', '只输出JSON字段income_yuan、debt_yuan：月收入1.2万元、月负债3500元。统一换算为元，数值类型。', {'income_yuan':12000,'debt_yuan':3500}),
        ('conflict', '只输出JSON字段income、conflict。两处收入分别写8000和9000元；数值矛盾时income必须null且conflict为true，不自行择一。', {'income':None,'conflict':True}),
        ('definition', '面向初学者，用100到150字解释信用风险，并举一个简单例子。避免承诺预测一定准确。', None),
        ('explanation', '本题规则规定负债/月收入>=0.5是高风险。客户收入8000、月负债5000。用150到200字解释计算、结论及两项核实事项。', None),
        ('rewrite', '用150到200字写一段礼貌的客户沟通话术：需要补充收入证明和现有负债信息才能继续评估，不承诺审批结果。', None),
        ('creative', '用150到200字写一个关于如实填写贷款资料的小故事，不出现真实人物，不承诺贷款获批。', None),
    ]
    return [dict(id=i,prompt=p,expected=e,kind='json' if e is not None else 'text') for i,p,e in definitions]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs=[]; rng=random.Random(20261117); ts=tasks()
    def add(group,value,task,rep,options,phase='measure'):
        jobs.append(dict(id=f'{len(jobs):04d}',group=group,value=value,task=task,repeat=rep,options=options,phase=phase))
    add('warmup','initial',ts[0],0,dict(BASE), 'warmup')
    # Balance condition order within each task/repeat. Match seeds across conditions.
    for rep in range(3):
        groups=[('temperature',[0.1,0.5,0.9]),('top_p',[0.5,0.9,1.0]),('top_k',[10,40,100]),('num_predict',[64,256,512])]
        rng.shuffle(groups)
        for group,values in groups:
            block=list(ts); rng.shuffle(block)
            for task in block:
                order=list(values);rng.shuffle(order)
                for value in order:
                    add(group,value,task,rep,dict(BASE,**{group:value,'seed':42+rep}))
    # 5 short / 3 medium / 2 long contexts, same prompt across all capacities.
    ctx_tasks=[]
    for i in range(10):
        lines=0 if i<5 else 70 if i<8 else 240
        filler='\n'.join(f'背景记录{j:04d}：该记录与目标字段无关，状态正常，禁止将背景当作答案。' for j in range(lines))
        prompt=f'文首校验码为ALPHA-{i}。\n{filler}\n文末校验码为OMEGA-{i}。只输出JSON，字段head和tail，分别填写文首和文末校验码；若已看不到则填null，不猜测。'
        ctx_tasks.append(dict(id=f'ctx{i}',prompt=prompt,expected={'head':f'ALPHA-{i}','tail':f'OMEGA-{i}'},kind='json',length_class='short' if i<5 else 'medium' if i<8 else 'long'))
    for rep in range(3):
        # Rotate block order to avoid confounding capacity with elapsed time.
        capacities=[4096,8192,16384];capacities=capacities[rep:]+capacities[:rep]
        for cap in capacities:
            opts=dict(BASE,num_ctx=cap,seed=42+rep)
            add('warmup',cap,ts[0],rep,opts,'warmup')
            block=list(ctx_tasks);rng.shuffle(block)
            for task in block:add('num_ctx',cap,task,rep,opts)
    for task in ts[6:]+[ts[1]]:
        for rep in range(4):
            conditions=[('fixed',42),('varied',42+rep)]
            if rep%2:conditions.reverse()
            for condition,seed in conditions:add('seed',condition,task,rep,dict(BASE,seed=seed))
    assert len(jobs)==500
    assert collections.Counter(j['group'] for j in jobs)==dict(warmup=10,temperature=90,top_p=90,top_k=90,num_predict=90,num_ctx=90,seed=40)
    data=''.join(json.dumps(j,ensure_ascii=False,sort_keys=True)+'\n' for j in jobs)
    path=OUT/'plan.jsonl'
    if path.exists() and path.read_text()!=data:raise RuntimeError('Existing plan differs; use a new run directory')
    path.write_text(data)
    return jobs


def strict_score(text, expected):
    if expected is None:return {'strict_json':None,'exact':None}
    try:
        def reject_constant(x): raise ValueError(x)
        def pairs(items):
            d={}
            for k,v in items:
                if k in d: raise ValueError('duplicate key')
                d[k]=v
            return d
        obj=json.loads(text,parse_constant=reject_constant,object_pairs_hook=pairs)
        exact=type(obj) is dict and obj.keys()==expected.keys() and all(type(obj[k]) is type(v) and obj[k]==v for k,v in expected.items())
        return {'strict_json':type(obj) is dict,'exact':exact}
    except (ValueError,TypeError):return {'strict_json':False,'exact':False}


def stamp():return datetime.now(timezone.utc).isoformat()

def persist(path, data):
    with path.open('a') as f:
        f.write(json.dumps(data,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def records():
    path=OUT/'responses.jsonl'
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def run(max_calls):
    jobs=prepare(); lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    client=httpx.Client(base_url='http://127.0.0.1:11434',trust_env=False,timeout=180)
    def api(route,payload=None):
        r=client.get(route) if payload is None else client.post(route,json=payload)
        r.raise_for_status();return r.json()
    tags=api('/api/tags'); model=next(m for m in tags['models'] if m['name']==MODEL)
    version=api('/api/version');show=api('/api/show',{'model':MODEL})
    identity={'model':MODEL,'digest':model['digest'],'version':version,'plan_sha256':hashlib.sha256((OUT/'plan.jsonl').read_bytes()).hexdigest()}
    meta=OUT/'metadata.json'
    if meta.exists():
        prior=json.loads(meta.read_text())
        if prior['identity']!=identity:raise RuntimeError('Model, backend or plan changed; refusing to mix runs')
    else:meta.write_text(json.dumps({'identity':identity,'show':show,'platform':platform.platform(),'created':stamp()},ensure_ascii=False,indent=2))
    done={r['id'] for r in records()};count=0;block_calls=0;block_start=time.monotonic();errors=0
    if done:
        print('Resume cooling 60s',flush=True)
        for _ in range(2):time.sleep(30)
        block_start=time.monotonic()
    for job in jobs:
        if job['id'] in done:continue
        if count>=max_calls:break
        if (OUT/'STOP').exists():break
        state=thermal()
        # Serious or critical: abort; fair: wait until nominal, at most 10 minutes.
        if state>=2:
            persist(OUT/'events.jsonl',dict(time=stamp(),event='thermal_stop',state=state));break
        waits=0
        while state==1 and waits<20:
            print('Thermal fair: cooling 30s',flush=True);time.sleep(30);waits+=1;state=thermal()
        if state!=0:break
        if block_calls>=20 or time.monotonic()-block_start>=120:
            print('Cooling 60s',flush=True)
            for _ in range(2):time.sleep(30)
            block_start=time.monotonic();block_calls=0
            if thermal()!=0:continue
        started=time.monotonic();rec={'id':job['id'],'time':stamp(),'thermal_before':state}
        try:
            payload={'model':MODEL,'messages':[{'role':'user','content':job['task']['prompt']}],'options':job['options'],'stream':False,'keep_alive':'5m','think':False}
            response=api('/api/chat',payload)
            rec.update(response=response,error=None,wall_s=time.monotonic()-started)
            rec['score']=strict_score(response['message']['content'],job['task']['expected'])
            rec['loaded_models']=api('/api/ps')
            errors=0
        except Exception as exc:
            rec.update(error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started);errors+=1
        rec['thermal_after']=thermal()
        persist(OUT/'responses.jsonl',rec)
        count+=1;block_calls+=1
        print(f"{len(done)+count}/500 {job['group']}={job['value']} {job['task']['id']} {rec['wall_s']:.1f}s thermal={rec['thermal_after']} error={rec['error']}",flush=True)
        if errors>=3 or rec['thermal_after']>=2:break
    print('Session finished',len(records()),'records',flush=True)


def summarize():
    jobs={j['id']:j for j in prepare()};rs=records();rows=[]
    for group in ['temperature','top_p','top_k','num_predict','num_ctx','seed']:
        vals=list(dict.fromkeys(j['value'] for j in jobs.values() if j['group']==group))
        for val in vals:
            subset=[(jobs[r['id']],r) for r in rs if jobs[r['id']]['group']==group and jobs[r['id']]['value']==val]
            ok=[(j,r) for j,r in subset if r.get('error') is None]
            keyed=collections.defaultdict(list)
            for j,r in ok:keyed[j['task']['id']].append(r['response']['message']['content'])
            # Exact agreement within the same task only; no cross-task text comparison.
            pairs=[a==b for outputs in keyed.values() for a,b in itertools.combinations(outputs,2)]
            structured=[(j,r) for j,r in subset if j['task']['expected'] is not None]
            speeds=[r['response']['eval_count']/(r['response']['eval_duration']/1e9) for j,r in ok if r['response'].get('eval_duration',0)>0]
            latency=[r['wall_s'] for j,r in ok]
            row=dict(group=group,value=val,n=len(subset),errors=len(subset)-len(ok),structured_n=len(structured),exact=sum(r.get('score',{}).get('exact') is True for j,r in structured),strict_json=sum(r.get('score',{}).get('strict_json') is True for j,r in structured),length_stops=sum(r['response'].get('done_reason')=='length' for j,r in ok),median_wall_s=statistics.median(latency) if latency else None,median_decode_tps=statistics.median(speeds) if speeds else None,identical_pairs=sum(pairs),pairs=len(pairs))
            rows.append(row)
    summary={'completed_calls':len(rs),'planned_calls':500,'rows':rows}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','summarize','thermal']);p.add_argument('--max-calls',type=int,default=100);args=p.parse_args()
    if args.action=='prepare':print('Prepared',len(prepare()),'calls')
    elif args.action=='thermal':print(thermal())
    elif args.action=='run':run(args.max_calls)
    else:summarize()
