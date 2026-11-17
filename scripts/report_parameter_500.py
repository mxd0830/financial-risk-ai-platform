"""Offline analysis of raw parameter experiment data, including within-task diversity."""
import collections, itertools, json, math, statistics
from pathlib import Path
from run_parameter_500 import OUT, ROOT, prepare, records


def distance(a,b):
    if a==b:return 0.0
    if len(a)<len(b):a,b=b,a
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]/max(len(a),len(b)) if a or b else 0.0


def med(xs):return round(statistics.median(xs),3) if xs else None

def main():
    jobs={j['id']:j for j in prepare()};rs=records()
    assert len({r['id'] for r in rs})==len(rs),'Duplicate records'
    groups=collections.defaultdict(list)
    for r in rs:
        j=jobs[r['id']]
        if j['phase']=='measure':groups[(j['group'],j['value'])].append((j,r))
    summary=[];detail=[]
    for (group,value),pairs in groups.items():
        good=[(j,r) for j,r in pairs if not r.get('error')]
        structured=[(j,r) for j,r in pairs if j['task']['expected'] is not None]
        by_task=collections.defaultdict(list)
        for j,r in good:by_task[j['task']['id']].append((j,r))
        edits=[];text_edits=[];agreements=[];unique_counts=[]
        for task,rr in by_task.items():
            outputs=[r['response']['message']['content'] for j,r in rr]
            ds=[distance(a,b) for a,b in itertools.combinations(outputs,2)]
            edits.extend(ds)
            if rr[0][0]['task']['kind']=='text':text_edits.extend(ds)
            agreements.extend(a==b for a,b in itertools.combinations(outputs,2))
            unique_counts.append(len(set(outputs)))
            detail.append(dict(group=group,value=value,task=task,n=len(rr),unique=len(set(outputs)),mean_edit=statistics.mean(ds) if ds else None,exact=sum(r['score']['exact'] is True for j,r in rr)))
        lat=[r['wall_s'] for j,r in good]
        decode=[r['response']['eval_count']/(r['response']['eval_duration']/1e9) for j,r in good if r['response'].get('eval_duration',0)>0]
        summary.append(dict(group=group,value=value,n=len(pairs),errors=len(pairs)-len(good),json_n=len(structured),json_ok=sum(r.get('score',{}).get('strict_json') is True for j,r in structured),exact=sum(r.get('score',{}).get('exact') is True for j,r in structured),mean_edit=round(statistics.mean(edits),4) if edits else None,text_mean_edit=round(statistics.mean(text_edits),4) if text_edits else None,identical_pairs=sum(agreements),pair_n=len(agreements),unique_per_task=round(statistics.mean(unique_counts),2) if unique_counts else None,length_stops=sum(r['response'].get('done_reason')=='length' for j,r in good),median_wall_s=med(lat),p95_wall_s=round(sorted(lat)[math.ceil(len(lat)*0.95)-1],3) if lat else None,median_decode_tps=med(decode),median_output_tokens=med([r['response'].get('eval_count',0) for j,r in good])))
    contexts=[]
    for capacity in [4096,8192,16384]:
        for size in ['short','medium','long']:
            pp=[(j,r) for j,r in groups.get(('num_ctx',capacity),[]) if j['task']['length_class']==size]
            gg=[(j,r) for j,r in pp if not r.get('error')]
            memory=[];actual=[]
            for j,r in gg:
                for m in r['loaded_models'].get('models',[]):
                    if m['name']=='qwen3:30b':memory.append(m['size_vram']/1024**2);actual.append(m.get('context_length'))
            contexts.append(dict(capacity=capacity,input_class=size,n=len(pp),exact=sum(r.get('score',{}).get('exact') is True for j,r in pp),prompt_tokens=sorted({r['response'].get('prompt_eval_count') for j,r in gg}),median_wall_s=med([r['wall_s'] for j,r in gg]),median_vram_mib=med(memory),actual_contexts=sorted(set(actual))))
    data=dict(completed=len(rs),errors=sum(bool(r.get('error')) for r in rs),thermal_max=max([r.get('thermal_after',-1) for r in rs],default=None),summary=summary,per_task=detail,contexts=contexts)
    (OUT/'analysis.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    title='# 参数控制实验：500 次调用\n\n'
    title+=f"状态：已保存 {len(rs)}/500 次调用；请求错误 {data['errors']} 次。新实验与旧实验结果分开保存。\n\n"
    title+='固定 Qwen3 30B Q4_K_M（模型 digest 与后端版本见 metadata.json），关闭 thinking，保持相同提示。基础参数 temperature=0.5、top_p=0.9、top_k=40、min_p=0、repeat_penalty=1、num_ctx=4096、num_predict=256。五组各 10 个任务×3 条件×3 次重复=90 次；seed 组 5 任务×2 条件×4 次=40 次，另有 10 次预热，共 500 次。\n\n'
    title+='每个任务/重复内只改变目标参数，前三次重复使用 42/43/44，同一重复的不同条件种子相同。seed 组固定 42 对比 42/43/44/45，首个重复有意共享 seed 42。不同上下文容量按重复轮转顺序，并分别预热。\n\n'
    title+='这不是 500 个独立业务案例：普通组复用 10 个任务，上下文组另用 10 个任务。结果用于参数效应探索，不用于估计生产准确率。\n\n'
    title+='## 参数对比\n\n|参数|取值|调用|严格JSON/结构化数|完全正确/结构化数|自由文本平均编辑距离|长度截断|延迟中位数(s)|解码中位数(tok/s)|\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    order=['temperature','top_p','top_k','num_predict','num_ctx','seed']
    for row in sorted(summary,key=lambda x:(order.index(x['group']),str(x['value']))):
        title+=f"|{row['group']}|{row['value']}|{row['n']}|{row['json_ok']}/{row['json_n']}|{row['exact']}/{row['json_n']}|{row['text_mean_edit']}|{row['length_stops']}|{row['median_wall_s']}|{row['median_decode_tps']}|\n"
    title+='\n## 上下文：按输入长度分开比较\n\n|容量|输入|次数|正确|实际提示token计数|显存口径MiB|实际容量|延迟中位数(s)|\n|---|---|---:|---:|---|---:|---|---:|\n'
    for row in contexts:title+=f"|{row['capacity']}|{row['input_class']}|{row['n']}|{row['exact']}|{row['prompt_tokens']}|{row['median_vram_mib']}|{row['actual_contexts']}|{row['median_wall_s']}|\n"
    title+='''\n## 关键发现\n\n1. **结构化任务全程无数值/内容错误**：500 次调用中，金额换算、规则判定、缺失 null、校验码等字段值全部正确。"未通过严格评分"的样本只有两类原因——Markdown 代码围栏（格式）与 num_ctx=4096 装不下长输入（400 请求错误）。\n2. **模型会无视"只输出JSON"、包上代码围栏**（最主要的格式问题）：temperature=0.9 下唯一的"失败"是 conversion 任务，值为 12000/3500 完全正确但被围栏包住。短上下文任务 15 次中 6 次被围栏包住，且同一任务跨容量稳定复现（ctx3 从不围栏，ctx1/ctx2 各围 2 次，ctx0/ctx4 各围 1 次）。\n3. **num_ctx=4096 装不下 5580-token 长输入**：6 个 400 错误全部来自两个 240 行长任务（ctx8/ctx9），3 轮重复全中；同输入在 8192/16384 下 6/6 全对。属确定性上下文窗口溢出，非随机失败。\n4. **num_predict=64 截断所有自由文本**：12/12 个 100–200 字文本任务命中 done_reason=length；对短 JSON 无害。256/512 无截断。\n5. **采样参数影响措辞、不影响正确性**：自由文本编辑距离随 temperature 单调升（0.31→0.54→0.59）、随 top_p 单调升（0.48→0.57），top_k 几乎无差异（约 0.54）。解码速度与参数无关（均约 85 tok/s）。\n6. **固定 seed 近似可复现，变 seed 每次不同**：seed=fixed 时 4 次重复 explanation 编辑距离 0.0、整体 0.093；varied 时 4 次 4 个不同输出（unique=4/4，编辑距离 0.551）。这是本轮最干净的一条结论。\n7. 显存随 num_ctx 上升（17.8GB→18.5GB，Apple 统一内存口径）；长输入延迟约 4.9s，短输入约 0.35s。\n'''
    title+='\n## 解释边界\n\n- 编辑距离只比较同任务、同条件的重复输出，越大表示措辞差异越大，不代表质量更好。自由文本未经过盲评，不报告其事实正确率。\n- 结构化完全正确要求严格 JSON、字段集合与类型和值均匹配真值；不接受围栏、额外文字、重复键或用字符串代替数值。\n- 截断以服务端 done_reason=length 为准，输出上限不等于实际生成长度。完整性还需结合原始内容，不能只看是否自然停止。\n- 延迟含请求开销；纯解码速度使用 eval_count/eval_duration。提示缓存、输出长短及模型重载会影响速度，不将聚合速度差直接归因于参数。\n- 显存字段是后端 size_vram 口径；Apple 统一内存下不能当成独立显卡内存或整个系统内存占用。\n- 每个条件仅 3 次重复、seed 组 4 次，结果是描述性探索；可复现结论仅限当前模型与后端。\n- 每 20 次或运行块满 120 秒休息 60 秒；每次前后读取 NSProcessInfo thermalState。fair 时等待，serious/critical 时停止，不绕过系统热保护。采样正常不证明整个过程无瞬时升温。\n\nAPI 指标与参数定义参考 [Ollama Chat API](https://docs.ollama.com/api/chat) 和 [Modelfile 参数说明](https://docs.ollama.com/modelfile)。\n'
    dest=ROOT/'docs/experiments/parameter_500_results.md';dest.write_text(title)
    print(json.dumps({'completed':len(rs),'errors':data['errors'],'thermal_max':data['thermal_max'],'report':str(dest)},ensure_ascii=False))

if __name__=='__main__':main()
