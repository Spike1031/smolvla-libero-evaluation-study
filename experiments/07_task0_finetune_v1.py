import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, json, hashlib, io, time, gc, shutil, fcntl
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
OUT=PROJECT/'finetune/task0_finetune_v1'
OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def write(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False));os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--stop',type=int,choices=[50,300],required=True);args=parser.parse_args()
assert version('lerobot')=='0.6.1', '必须沿用0.6.1，以免接口改变'
assert torch.cuda.is_available(), '本单元需要GPU；准备步骤不需要'
report=read(DATA/'preparation_report.json');split=read(DATA/'split.json');plan=read(AUDIT/'evaluation_plan.json')
assert report['episodes']==45 and report['frames']==4487
assert report['plan_hash']==split['plan_hash']
assert not set(split['train_episode_ids'])&set(split['validation_episode_ids'])
assert plan['model_revision']=='6721902bc4d61e50a3bfdb11dfb4cb626f05d102'
assert plan['vlm_revision']=='7b375e1b73b11138ff12fe22c8f2822d8fe03467'
# 同一个Colab运行时禁止启动两个训练进程。
lock=open('/content/task0_finetune_v1.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert read(PROJECT/'finetune/smoke_train_v1/smoke_report.json')['status']=='passed'
protocol={'data_plan_hash':report['plan_hash'],'split':split,
 'model_revision':plan['model_revision'],'vlm_revision':plan['vlm_revision'],
 'lerobot':'0.6.1','updates':300,'micro_batch':1,'accumulation':4,'lr':1e-5,
 'chunk_size':50,'seed':42,'validation':'9 episodes x 3 anchors, fixed noise seeds',
 'sampling':'uniform train episode per update; 4 uniform frames with replacement',
 'selection':'minimum validation loss among steps 50,100,150,200,250,300',
 'scope':'single-task exploratory fine-tuning; no claim of unseen pretraining data'}
if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol,'实验协议变化，停止'
else:write(OUT/'protocol.json',protocol)
if (OUT/'training_report.json').exists():
 print('正式训练已完成，不重复运行。',OUT/'training_report.json');raise SystemExit(0)
# HF符号链接缓存只放本地。Drive保存数据、检查点与导出的真实文件。
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
# 数据读取函数供训练和固定验证使用；每次只保留一条轨迹。
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


from safetensors.torch import save_file, load_file
# 验证锚点固定；不利用仿真测试成功率选择训练检查点。
val_anchors=[(e,t) for e in split['validation_episode_ids']
             for t in [0,(receipts[e]['frames']-1)//2,receipts[e]['frames']-1]]
def validate():
 policy.eval();values=[]
 with torch.no_grad():
  for j,(e,t) in enumerate(val_anchors):
   torch.manual_seed(9000+j);torch.cuda.manual_seed_all(9000+j)
   v=float(policy.forward(batch_at(e,t))[0]);assert np.isfinite(v)
   values.append(v)
 return {'mean_loss':float(np.mean(values)),'per_anchor_loss':values,'anchors':val_anchors}
def trainable_cpu():return {n:p.detach().cpu().contiguous().clone() for n,p in params.items()}
def load_trainable(state):
 assert set(state)==set(params)
 with torch.no_grad():
  for n,p in params.items():p.copy_(state[n].to(p.device))
# commit写完才把检查点视为有效；中断写入的文件不会作为恢复来源。
def committed():
 return sorted(OUT.glob('checkpoint_*.json'),reverse=True)
start=0;history=[];validations=[];best=None;base=None
for c in committed():
 info=read(c);p=OUT/info['file']
 if not p.exists() or sha(p)!=info['sha256']:
  print('跳过不完整检查点：',c.name,flush=True);continue
 saved=torch.load(p,map_location='cpu',weights_only=False)
 assert saved['protocol']==protocol
 load_trainable(saved['trainable']);opt.load_state_dict(saved['optimizer'])
 start=saved['step'];history=saved['history'];validations=saved['validations'];best=saved['best'];base=saved['base']
 restored=validate()['mean_loss']
 assert np.isclose(restored,validations[-1]['mean_loss'],rtol=1e-5,atol=1e-6),'恢复验证不一致，停止'
 print(f'恢复第{start}步；从第{start+1}步继续。',flush=True)
 del saved;gc.collect();break
if start==0:
 base=validate();write(OUT/'baseline_validation.json',base)
 print('原始模型验证损失：',base['mean_loss'],flush=True)
if start>=args.stop and start<300:
 print('本阶段已完成。下一阶段运行到300步。');raise SystemExit(0)
started=time.monotonic()
for step in range(start+1,args.stop+1):
 rng=np.random.default_rng(42+step)
 episode=int(rng.choice(split['train_episode_ids']))
 frames=rng.integers(0,receipts[episode]['frames'],size=4).tolist()
 policy.train();opt.zero_grad(set_to_none=True);losses=[]
 for micro,frame in enumerate(frames):
  seed=42000+step*4+micro;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  loss,_=policy.forward(batch_at(episode,frame));assert torch.isfinite(loss).all()
  (loss/4).backward();losses.append(float(loss.detach()));del loss
 norm=torch.nn.utils.clip_grad_norm_(list(params.values()),10.,error_if_nonfinite=True)
 opt.step();opt.zero_grad(set_to_none=True)
 history.append({'step':step,'episode':episode,'frames':frames,'loss':float(np.mean(losses)),'grad_norm_before_clip':float(norm)})
 if step%5==0:print(f'更新 {step}/300 | loss={np.mean(losses):.5f} | 本次用时{time.monotonic()-started:.0f}秒',flush=True)
 if step%50==0:
  v=validate();v['step']=step;validations.append(v)
  print(f'验证 step={step} loss={v["mean_loss"]:.6f}；正在保存，请勿手动断开。',flush=True)
  if best is None or v['mean_loss']<best['loss']:
   bp=OUT/f'candidate_{step:06d}.safetensors';tmp=bp.with_suffix('.tmp')
   save_file(trainable_cpu(),str(tmp));os.replace(tmp,bp)
   best={'step':step,'loss':v['mean_loss'],'file':bp.name,'sha256':sha(bp)}
  cp=OUT/f'checkpoint_{step:06d}.pt';tmp=cp.with_suffix('.tmp')
  torch.save({'protocol':protocol,'step':step,'trainable':trainable_cpu(),'optimizer':opt.state_dict(),
              'history':history,'validations':validations,'best':best,'base':base},tmp)
  os.replace(tmp,cp);digest=sha(cp)
  write(cp.with_suffix('.json'),{'file':cp.name,'step':step,'sha256':digest,'best':best})
  write(OUT/'progress.json',{'completed_updates':step,'target_updates':300,'baseline_val_loss':base['mean_loss'],
    'best':best,'validations':validations,'last_checkpoint':cp.name})
  # 保留最近两个已提交检查点以及各自引用的候选；先保存成功再清理旧文件。
  commits=committed();keep=commits[:2]
  keep_best={read(c)['best']['file'] for c in keep}
  for c in commits[2:]:
   (OUT/read(c)['file']).unlink(missing_ok=True);c.unlink()
  for p in OUT.glob('candidate_*.safetensors'):
   if p.name not in keep_best:p.unlink()
  print(f'已安全保存第{step}步。',flush=True)
if args.stop<300:
 print('首50步完成。请发送进度输出，确认耗时与验证结果后，再运行后续单元格。');raise SystemExit(0)
# 导出验证集选择的候选模型；原始预训练权重和历史仿真结果完全保留。
bp=OUT/best['file'];assert sha(bp)==best['sha256']
load_trainable(load_file(str(bp)));policy.eval()
export=OUT/'selected_model'
export.mkdir(exist_ok=True)
# 先本地完整写入，再逐文件复制到Drive；清单最后写，清单缺失表示导出未完成。
local=Path('/content/task0_finetune_export');local.mkdir(exist_ok=True)
cfg.vlm_model_name=plan['vlm_repo'];cfg.load_vlm_weights=False
policy.save_pretrained(local)
# 直接复制锁定原模型的处理器，确保归一化与原评估一致。
for p in Path(model_dir).glob('policy_*'):
 if p.is_file():shutil.copyfile(p,local/p.name)
for p in local.iterdir():
 if p.is_file():
  temp=export/(p.name+'.tmp');shutil.copyfile(p,temp);os.replace(temp,export/p.name)
files={p.name:sha(p) for p in export.iterdir() if p.is_file() and not p.name.endswith('.tmp')}
write(OUT/'training_report.json',{'status':'completed','completed_updates':300,'selected':best,
 'baseline_validation_loss':base['mean_loss'],'validations':validations,
 'model_dir':str(export),'export_sha256':files,'protocol':protocol,
 'note':'验证损失用于选候选；尚未完成微调后的仿真评估，不能声称成功率提高。评估时必须沿用锁定VLM revision。'})
print('训练与模型导出完成。可以断开GPU；请发送汇总，下一步准备固定条件仿真评估。',flush=True)
