import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, json, hashlib, io, time, gc
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import snapshot_download
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from importlib.metadata import version

PROJECT=Path('/content/drive/MyDrive/embodied-smolvla')
DATA=PROJECT/'finetune/task0_demonstrations_v1'
AUDIT=PROJECT/'finetune/data_audit/20261011T195352_660570Z'
OUT=PROJECT/'finetune/smoke_train_v1'
OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def write(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False));os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--stop',type=int,choices=[10,20],required=True);args=parser.parse_args()
assert version('lerobot')=='0.6.1', '必须沿用0.6.1，以免接口改变'
assert torch.cuda.is_available(), '本单元需要GPU；准备步骤不需要'
report=read(DATA/'preparation_report.json');split=read(DATA/'split.json');plan=read(AUDIT/'evaluation_plan.json')
assert report['episodes']==45 and report['frames']==4487
assert report['plan_hash']==split['plan_hash']
assert not set(split['train_episode_ids'])&set(split['validation_episode_ids'])
assert plan['model_revision']=='6721902bc4d61e50a3bfdb11dfb4cb626f05d102'
assert plan['vlm_revision']=='7b375e1b73b11138ff12fe22c8f2822d8fe03467'
protocol={'data_plan_hash':report['plan_hash'],'model_revision':plan['model_revision'],
 'vlm_revision':plan['vlm_revision'],'lerobot':'0.6.1','batch_size':1,'steps':20,'lr':1e-5,
 'chunk_size':50,'seed':42,'scope':'interface smoke test, not formal fine-tuning or task-success experiment'}
if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
else:write(OUT/'protocol.json',protocol)
if (OUT/'smoke_report.json').exists() and read(OUT/'smoke_report.json').get('completed_steps')==20:
 print('20步测试已完成，不重复训练。',OUT/'smoke_report.json');raise SystemExit(0)
# 缓存到Drive，断开后不重复下载；只下载VLM配置和分词器，不下载它的权重。
cache=Path('/content/smolvla_hf_cache')
model_dir=snapshot_download(plan['model_repo'],revision=plan['model_revision'],cache_dir=cache,
 allow_patterns=['config.json','model.safetensors','policy_*.json','policy_*.safetensors'])
vlm_dir=snapshot_download(plan['vlm_repo'],revision=plan['vlm_revision'],cache_dir=cache,
 allow_patterns=['*.json','*.txt','*.model','*.jinja','*.tiktoken'])
print('已取得锁定模型与分词器；开始加载。',flush=True)
cfg=SmolVLAConfig.from_pretrained(model_dir)
cfg.device='cuda';cfg.load_vlm_weights=False;cfg.vlm_model_name=vlm_dir
cfg.push_to_hub=False;cfg.compile_model=False;cfg.use_amp=False
cfg.freeze_vision_encoder=True;cfg.train_expert_only=True;cfg.train_state_proj=True
policy=SmolVLAPolicy.from_pretrained(model_dir,config=cfg,strict=True)
pre,post=make_pre_post_processors(cfg,pretrained_path=model_dir,
 preprocessor_overrides={'device_processor':{'device':'cuda'},'tokenizer_processor':{'tokenizer_name':vlm_dir}})
# 直接沿用检查点归一化；不将新子集统计值覆盖到预处理器。
params={n:p for n,p in policy.named_parameters() if p.requires_grad}
assert params and all(not p.requires_grad for p in policy.model.vlm_with_expert.vlm.parameters())
opt=torch.optim.AdamW(list(params.values()),lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
print('GPU:',torch.cuda.get_device_name(0),'可训练参数:',sum(p.numel() for p in params.values()),flush=True)
TASK='pick up the black bowl between the plate and the ramekin and place it on the plate'
receipts={r['episode_index']:r for r in report['reports']}
# 仅取训练集。为控制内存，每次保留一条完整轨迹。
last_episode=None;rows=None

def batch_at(episode,frame):
 global last_episode,rows
 if last_episode!=episode:
  path=DATA/'episodes'/f'episode_{episode:06d}.parquet'
  assert sha(path)==receipts[episode]['sha256'],f'轨迹{episode}文件校验失败'
  rows=pq.read_table(path).to_pylist();last_episode=episode
 n=len(rows);ids=np.minimum(np.arange(frame,frame+50),n-1)
 out={'observation.state':torch.tensor(rows[frame]['observation.state'],dtype=torch.float32).unsqueeze(0),
      'action':torch.tensor([rows[int(i)]['action'] for i in ids],dtype=torch.float32).unsqueeze(0),
      'action_is_pad':torch.tensor(np.arange(frame,frame+50)>=n).unsqueeze(0),'task':[TASK]}
 for key in ['observation.images.image','observation.images.image2']:
  im=Image.open(io.BytesIO(rows[frame][key]['bytes'])).convert('RGB')
  out[key]=torch.from_numpy(np.array(im)).permute(2,0,1).float().div(255).unsqueeze(0)
 b=pre(out)
 assert b['action'].shape==(1,50,7) and b['action_is_pad'].shape==(1,50)
 assert b['observation.state'].shape==(1,8)
 assert torch.isfinite(b['action']).all() and torch.isfinite(b['observation.state']).all()
 return b

probe_episode=split['train_episode_ids'][0]
def probe():
 policy.eval()
 torch.manual_seed(777);torch.cuda.manual_seed_all(777)
 with torch.no_grad():value=float(policy.forward(batch_at(probe_episode,0))[0])
 assert np.isfinite(value)
 return value

checkpoint=OUT/'checkpoint_step10.pt';commit=OUT/'checkpoint_step10.json'
start=0;losses=[];restore_ok=False
if checkpoint.exists() and commit.exists():
 info=read(commit);assert sha(checkpoint)==info['sha256'],'检查点校验失败，停止'
 # 只读取本脚本自己保存的本地检查点；不加载外来pickle文件。
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert saved['protocol']==protocol and set(saved['trainable'])==set(params)
 with torch.no_grad():
  for name,p in params.items():p.copy_(saved['trainable'][name].to(p.device))
 opt.load_state_dict(saved['optimizer'])
 start=int(saved['step']);losses=saved['losses'];reference=saved['probe_loss']
 restored=probe();assert np.isclose(restored,reference,rtol=1e-5,atol=1e-6),f'恢复前后固定输入loss不一致：{reference} vs {restored}'
 restore_ok=True;print(f'恢复检查通过，从第{start+1}步继续。',flush=True)
 del saved;gc.collect()
else:
 assert args.stop==10,'请先完成第一个10步单元格'
 print('初始固定输入loss:',probe(),flush=True)
 # 专门核验轨迹末尾补齐不跨入下一条示范。
 tail=batch_at(probe_episode,receipts[probe_episode]['frames']-3)
 assert tail['action_is_pad'].sum().item()==47
 del tail
if start>=args.stop:
 print('该阶段已经完成，无需重跑。');raise SystemExit(0)
watch=policy.model.action_out_proj.weight
before=watch.detach().cpu().clone();started=time.monotonic()
for step in range(start+1,args.stop+1):
 episode=split['train_episode_ids'][(step-1)%len(split['train_episode_ids'])]
 length=receipts[episode]['frames'];frame=((step-1)*7)%length
 torch.manual_seed(42+step);torch.cuda.manual_seed_all(42+step)
 policy.train();opt.zero_grad(set_to_none=True)
 loss,details=policy.forward(batch_at(episode,frame))
 assert torch.isfinite(loss).all(), 'loss出现NaN/Inf，停止'
 loss.backward()
 norm=torch.nn.utils.clip_grad_norm_(list(params.values()),10.0,error_if_nonfinite=True)
 opt.step();losses.append({'step':step,'episode':episode,'frame':frame,'loss':float(loss.detach()),'grad_norm':float(norm)})
 print(f'训练 {step}/20 | loss={float(loss.detach()):.6f} | 用时{time.monotonic()-started:.1f}秒',flush=True)
 del loss;opt.zero_grad(set_to_none=True)
changed=float((watch.detach().cpu()-before).abs().max());assert changed>0,'动作输出层未发生更新'
if args.stop==10:
 value=probe()
 payload={'protocol':protocol,'step':10,'trainable':{n:p.detach().cpu() for n,p in params.items()},
          'optimizer':opt.state_dict(),'losses':losses,'probe_loss':value}
 temp=checkpoint.with_suffix('.pt.tmp');torch.save(payload,temp);os.replace(temp,checkpoint)
 write(commit,{'sha256':sha(checkpoint),'step':10,'probe_loss':value})
 write(OUT/'phase1_report.json',{'completed_steps':10,'parameter_max_change':changed,'checkpoint_saved':True,'losses':losses})
 print('前10步完成并已保存。下一步运行第二个训练单元格，使用新进程恢复。',flush=True)
else:
 assert restore_ok
 write(OUT/'smoke_report.json',{'status':'passed','completed_steps':20,'resumed_from_step':10,
    'restore_fixed_input_loss_match':restore_ok,'parameter_max_change_phase2':changed,
    'trainable_parameters':sum(p.numel() for p in params.values()),'losses':losses,
    'note':'仅工程检查；第20步权重未另存，保留第10步恢复检查点；不用于最终效果对比'})
 print('20步训练/保存/恢复测试通过。第20步权重仅用于测试，不作为正式微调模型。',flush=True)
