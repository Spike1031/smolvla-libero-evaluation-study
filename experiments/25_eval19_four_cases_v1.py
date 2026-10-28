"""Only four policy-only rollouts of fixed19; reuse verified15 original controls."""
import os
os.environ.update(MPLBACKEND='Agg',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
import argparse,fcntl,hashlib,importlib.metadata as md,importlib.util,inspect,json,platform,shutil,subprocess,sys,uuid
from pathlib import Path
import numpy as np
CORE_SHA='1a4891181e0fa5e3984aee778b93e4b1fb07eede555cc269fd60013685ec4677'
BASELINE15_SHA='4bb29c96155c5becfdef2bb4924fcd1fd832c9f9a75704141066eae85c53314a'
TRAIN19_SHA='388aa112c35777f78b2694f7197de127f618aa9bacdd0e3a963560e03c312a83'
MODEL='hold_lr1e5_fixed50'
CASES=[[1,'clean'],[5,'clean'],[16,'occluded'],[8,'clean']]

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def load_core(project):
 path=project/'finetune/18_finish_policy_eval_v1.py'
 assert sha(path)==CORE_SHA,'15号执行代码变化，停止，不追加对照回合'
 s=importlib.util.spec_from_file_location('locked15_rollout_core',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 assert m.MAX_STEPS==440 and m.WINDOW_STEPS==40 and m.CASES[:4]==CASES
 return m

def validate_baseline(p,core):
 plan=p['plan'];assert plan['runner_sha256']==CORE_SHA
 assert plan['max_steps']==440 and plan['control_hz']==20 and plan['n_action_steps']==1
 assert plan['suite']=='libero_spatial' and plan['task_id']==0 and plan['seed_base']==42
 assert plan['mask']=={'camera':'observation.images.image','box_yxyx':[96,160,96,160],'fill':.5,'timing':'every_policy_observation'}
 selected=[]
 for sid,condition in CASES:
  rows=[x for x in p['cases'] if x['result']['model']=='original' and x['result']['state_id']==sid and x['result']['condition']==condition]
  assert len(rows)==1;entry=rows[0];r=entry['result']
  assert r['status']=='completed' and r['control_steps']==440 and r['policy_only'] and r['n_action_steps']==1
  assert r['seed']==42+sid and r['plan_hash']==core.digest(plan)
  trace=entry['step_trace'];assert len(trace)==440 and [x['step'] for x in trace]==list(range(1,441))
  assert all(x['action_source']=='policy' for x in trace)
  assert len(r['videos'])==1 and r['videos'][0]['frames']==441 and r['videos'][0]['fps']==20
  selected.append(entry)
 return selected

def prepare(project,root,core,check_weights=True):
 baseline_path=project/'results/finish_policy_eval_v1/smolvla_finish_policy_eval_v1.json'
 training_path=project/'finetune/task0_hold_lr1e5_fixed50_v1/smolvla_hold_lr1e5_fixed50_v1.json'
 assert sha(baseline_path)==BASELINE15_SHA,'15号完成报告变化或缺失，停止，不自动重跑原模型'
 assert sha(training_path)==TRAIN19_SHA,'19号完成报告变化或缺失，停止'
 baseline=read(baseline_path);training=read(training_path);selected=validate_baseline(baseline,core)
 t=training['training'];v=training['reload_check']
 assert t['status']=='completed' and t['completed_updates']==t['selected_step']==50
 assert training['run_protocol']==t['run_protocol'] and t['run_protocol']['lr']==1e-5
 assert v['passed'] and v['selected_step']==50 and v['run_protocol_hash']==digest(t['run_protocol'])
 assert v['export_sha256']==t['export_check']['export_sha256']
 assert v['predictions']==t['final_action_diagnostic']
 assert t['export_check']['frozen_unchanged'] and t['export_check']['processors_identical_to_original']
 base=baseline['plan'];hashes=t['export_check']['export_sha256']
 for name in t['export_check']['processor_files']:assert hashes[name]==base['model_files']['original'][name]
 # Verify retained original artifacts, including videos. Never rerun original controls.
 for entry in selected:
  r=entry['result'];old=project/'results/finish_policy_eval_v1/original'
  assert read(old/f"state_{r['state_id']:03d}"/r['condition']/'result.json')==r
  for name,value in r['artifact_sha256'].items():
   path=(old/name).resolve();assert path.is_relative_to(old.resolve()) and sha(path)==value,'15号原模型文件缺失或变化：'+name
 model_source=project/'finetune/task0_hold_lr1e5_fixed50_v1'/MODEL/'selected_model'
 if check_weights:
  for name,value in hashes.items():assert sha(model_source/name)==value,'19号固定50步模型文件变化：'+name
 keys=['suite','task_id','seed_base','mask','model_repo','model_revision','vlm_repo','vlm_revision',
       'assets_repo','assets_revision','control_hz','max_steps','benchmark_horizon','n_action_steps',
       'evaluation_actions','finish_rule','manual_review']
 plan={k:base[k] for k in keys}
 plan.update(protocol='19_four_case_eval_v1',cases=CASES,models=[MODEL],model_files={MODEL:hashes},
  model19_result_sha256=TRAIN19_SHA,baseline15_result_sha256=BASELINE15_SHA,
  baseline15_plan_hash=core.digest(base),baseline_model_files=base['model_files']['original'],
  core_sha256=CORE_SHA,runner_sha256=sha(__file__),max_new_completed_cases=4,max_policy_actions=1760,
  scope='three known training cases plusone known development validation; no unseen test or generalization claim',
  baseline_reuse='only matching four original15 cases; no extra baseline rollouts',
  case_order='001 clean,005 clean,016 occluded,008 clean',
  manual_review_fields=['grasp_lift','placement','actual_release','safe_withdrawal','jitter','recontact','regrasp','final_bowl_state'])
 root.mkdir(parents=True,exist_ok=True);path=root/'plan.json'
 if path.exists():assert read(path)==plan,'22号计划变化，停止覆盖'
 else:core.write_json(path,plan)
 return plan,baseline,selected,model_source

def summarize(root,plan,core):
 rows=[]
 assert plan['cases']==CASES and plan['models']==[MODEL]
 for sid,c in CASES:
  r=core.completed(root/MODEL,sid,c,digest(plan))
  if r is not None:rows.append(r)
 result={'completed':len(rows),'requested':4,'completed_policy_actions':sum(x['control_steps'] for x in rows),
  'benchmark_goal_within_280':sum(x['scores']['benchmark_goal_within_280'] for x in rows),
  'goal_at_end':sum(x['scores']['goal_at_end'] for x in rows),
  'complete_finish_automatic':sum(x['scores']['complete_finish_automatic'] for x in rows),
  'manual_review':'pending','limits':'four known development cases; no statistical success-rate or generalization claim'}
 core.write_json(root/'summary.json',result);return result

def compare_compatibility(compat,expected):
 for key in ['initial_states_sha256','shape','dtype']:assert compat[key]==expected[key],'起始状态不一致：'+key
 for key in ['python','packages','gpu','cuda','eval_source_sha256','env_source_sha256']:
  assert compat['audit'][key]==expected['audit'][key],'环境与15号不一致：'+key

def run_four(root,plan,baseline,audit,core,**kw):
 from lerobot.envs.libero import LiberoEnv
 envs=[(g,t,e) for g,v in kw['envs'].items() for t,e in v.items()]
 assert len(envs)==1 and envs[0][:2]==('libero_spatial',0)
 env=envs[0][2];assert env.num_envs==1 and hasattr(env,'envs')
 base=env.envs[0].unwrapped;assert isinstance(base,LiberoEnv) and base.control_freq==20
 states=np.asarray(base._init_states)
 compat={'initial_states_sha256':hashlib.sha256(states.tobytes()).hexdigest(),'shape':list(states.shape),
         'dtype':str(states.dtype),'audit':audit}
 compare_compatibility(compat,baseline['compatibility']);assert states.shape[0]>16
 cp=root/'compatibility.json'
 if cp.exists():assert read(cp)==compat,'重连后环境变化，停止覆盖'
 else:core.write_json(cp,compat)
 for index,(sid,c) in enumerate(CASES,1):
  if core.completed(root/MODEL,sid,c,digest(plan)) is not None:
   print(f'跳过已完成回合 {index}/4：{sid:03d}/{c}',flush=True);continue
  print(f'开始回合 {index}/4：{sid:03d}/{c}，种子{42+sid}，固定440步',flush=True)
  core.run_case(root,plan,MODEL,sid,c,base,env,kw)
  summary=summarize(root,plan,core)
  print('本轮已完成',summary['completed'],'/4；不运行其他模型或案例。',flush=True)
 return {'overall':summarize(root,plan,core)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--status',action='store_true');a=p.parse_args()
 project=Path(a.project);root=project/'results/19_four_case_eval_v1';root.mkdir(parents=True,exist_ok=True)
 lock=open('/tmp/19_four_case_'+hashlib.sha256(str(root.resolve()).encode()).hexdigest()+'.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 core=load_core(project)
 plan,baseline,selected,model_source=prepare(project,root,core,check_weights=not a.status)
 if a.status:print(json.dumps(summarize(root,plan,core),ensure_ascii=False,indent=2));return
 if summarize(root,plan,core)['completed']==4:
  print('4回合已完成并校验，跳过；不增加评估。',flush=True);return
 import torch
 assert md.version('lerobot')=='0.6.1' and torch.cuda.is_available() and torch.cuda.get_device_name(0)=='Tesla T4'
 from huggingface_hub import snapshot_download
 hashes=plan['model_files'][MODEL];model=Path('/content/smolvla_eval19_fixed50');model.mkdir(exist_ok=True)
 for name,value in hashes.items():
  target=model/name;target.parent.mkdir(parents=True,exist_ok=True)
  if not target.is_file() or sha(target)!=value:shutil.copyfile(model_source/name,target)
  assert sha(target)==value
 vlm=snapshot_download(plan['vlm_repo'],revision=plan['vlm_revision'],cache_dir='/content/smolvla_hf_cache',
  allow_patterns=['*.json','*.txt','*.model','*.jinja','*.tiktoken'])
 proc_path=model/'policy_preprocessor.json';proc=read(proc_path)
 def fix_tokenizer(x):
  count=0
  if isinstance(x,dict):
   for k,value in x.items():
    if k=='tokenizer_name':x[k]=str(vlm);count+=1
    else:count+=fix_tokenizer(value)
  elif isinstance(x,list):
   for value in x:count+=fix_tokenizer(value)
  return count
 assert fix_tokenizer(proc)==1;core.write_json(proc_path,proc)
 import libero.libero as libero_core
 snapshot_download(plan['assets_repo'],repo_type='dataset',revision=plan['assets_revision'],
  local_dir=str(Path(libero_core.__file__).parent/'assets'),max_workers=4)
 from lerobot.scripts import lerobot_eval as ev
 from lerobot.envs.libero import LiberoEnv
 audit={'python':platform.python_version(),'packages':{d.metadata['Name'].lower():d.version for d in md.distributions() if d.metadata['Name']},
  'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'eval_source_sha256':sha(inspect.getfile(ev)),
  'env_source_sha256':sha(inspect.getfile(LiberoEnv)),'runner_sha256':sha(__file__),'core_sha256':CORE_SHA}
 # Fail before environment rollouts if execution packages differ from retained controls.
 expected=baseline['compatibility']['audit']
 for k in ['python','packages','gpu','cuda','eval_source_sha256','env_source_sha256']:assert audit[k]==expected[k],'环境不一致：'+k
 session=root/'sessions'/uuid.uuid4().hex[:12];session.mkdir(parents=True)
 core.write_json(session/'environment.json',audit)
 ev.eval_policy_all=lambda **kw:run_four(root,plan,baseline,audit,core,**kw)
 sys.argv=[sys.argv[0],f'--policy.path={model}',f'--policy.vlm_model_name={vlm}',
  '--policy.device=cuda','--policy.use_amp=false','--policy.load_vlm_weights=false','--policy.n_action_steps=1',
  '--env.type=libero','--env.task=libero_spatial','--env.task_ids=[0]','--env.observation_height=256','--env.observation_width=256',
  '--env.control_mode=relative','--env.init_states=true','--env.max_parallel_tasks=1','--eval.batch_size=1',
  '--eval.n_episodes=1','--eval.use_async_envs=false','--seed=42',f'--output_dir={session}']
 core.write_json(session/'command.json',sys.argv);ev.main()

if __name__=='__main__':main()
