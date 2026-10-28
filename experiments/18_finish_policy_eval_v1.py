"""LeRobot 0.6.1：同一任务的配对评估；每次完成后独立提交结果。"""
import os
os.environ.update(MPLBACKEND="Agg", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import platform
import subprocess
import sys
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


import time
import numpy as np

class MaskInput:
    def __init__(self, inner, condition, output):
        self.inner, self.condition, self.output = inner, condition, output
        self.calls = 0
    def reset(self):
        if hasattr(self.inner, "reset"):
            self.inner.reset()
    def __call__(self, obs):
        import torch
        from PIL import Image
        obs = self.inner(obs)
        key = "observation.images.image"
        x = obs[key]
        assert tuple(x.shape) == (1,3,256,256), f"Unexpected input shape: {x.shape}"
        assert x.is_floating_point() and bool(torch.isfinite(x).all())
        assert float(x.min()) >= 0 and float(x.max()) <= 1, "Mask must precede normalization"
        y = x.clone()
        if self.condition == "occluded":
            y[:,:,96:160,96:160] = 0.5
        out = dict(obs)
        out[key] = y
        if self.calls == 0:
            self.output.mkdir(parents=True, exist_ok=True)
            def save(t, name):
                arr = (t[0].detach().cpu().permute(1,2,0).numpy()*255).round().astype("uint8")
                Image.fromarray(arr).save(self.output / name)
            save(x, "main_original.png")
            save(y, "main_policy_input.png")
            save(obs["observation.images.image2"], "wrist_unchanged.png")
            write_json(self.output / "input_check.json", {"condition": self.condition,
                "shape": list(x.shape), "changed_pixels": int((x != y).any(dim=1).sum().item()),
                "mask_box_yxyx": [96,160,96,160], "fill": 0.5})
        self.calls += 1
        return out

class SimulatorEvidence:
    def __init__(self, base):
        self.raw=base._env.env;self.sim=self.raw.sim;robot=self.raw.robots[0]
        gripper=robot.gripper
        if isinstance(gripper,dict):
            assert len(gripper)==1;gripper=next(iter(gripper.values()))
        prefix=gripper.naming_prefix;assert isinstance(prefix,str) and prefix
        self.gripper_ids={i for i in range(self.sim.model.ngeom)
                          if (self.sim.model.geom_id2name(i) or '').startswith(prefix)}
        self.bowl_body=int(self.raw.obj_body_id['akita_black_bowl_1'])
        descendants=set()
        for body in range(self.sim.model.nbody):
            b=body
            while b:
                if b==self.bowl_body:descendants.add(body);break
                b=int(self.sim.model.body_parentid[b])
        self.bowl_ids={i for i in range(self.sim.model.ngeom) if int(self.sim.model.geom_bodyid[i]) in descendants}
        assert self.gripper_ids and self.bowl_ids and self.gripper_ids.isdisjoint(self.bowl_ids)
        site=robot.eef_site_id
        if isinstance(site,dict):assert len(site)==1;site=next(iter(site.values()))
        self.eef_site=int(site)
        qindices=robot._ref_gripper_joint_pos_indexes
        if isinstance(qindices,dict):assert len(qindices)==1;qindices=next(iter(qindices.values()))
        self.gripper_qindices=qindices
    def measure(self):
        sim=self.sim
        contacts=[]
        for i in range(sim.data.ncon):
            c=sim.data.contact[i];a,b=int(c.geom1),int(c.geom2)
            if (a in self.gripper_ids and b in self.bowl_ids) or (b in self.gripper_ids and a in self.bowl_ids):
                contacts.append([a,b])
        # Bounding spheres enclose the complete geometries, including mesh geometry.
        # They give a conservative clearance, not a claim of exact surface distance.
        lower=min(float(sim.data.geom_xpos[i,2]-sim.model.geom_rbound[i]) for i in self.gripper_ids)
        upper=max(float(sim.data.geom_xpos[i,2]+sim.model.geom_rbound[i]) for i in self.bowl_ids)
        return {'goal':bool(self.raw._check_success()),'gripper_bowl_contact':bool(contacts),
                'contact_geom_ids':contacts,'clearance_lower_bound_m':lower-upper,
                'eef_position':np.asarray(sim.data.site_xpos[self.eef_site]).tolist(),
                'eef_matrix':np.asarray(sim.data.site_xmat[self.eef_site]).reshape(-1).tolist(),
                'bowl_position':np.asarray(sim.data.body_xpos[self.bowl_body]).tolist(),
                'bowl_quaternion':np.asarray(sim.data.body_xquat[self.bowl_body]).tolist(),
                'gripper_joint_positions':np.asarray(sim.data.qpos[self.gripper_qindices]).tolist()}
    def schema(self):
        return {'gripper_geom_ids':sorted(self.gripper_ids),'bowl_geom_ids':sorted(self.bowl_ids),
                'gripper_geom_names':[self.sim.model.geom_id2name(i) for i in sorted(self.gripper_ids)],
                'bowl_geom_names':[self.sim.model.geom_id2name(i) for i in sorted(self.bowl_ids)],
                'clearance':'min(gripper_geom_z-rbound)-max(bowl_geom_z+rbound), conservative bounds'}

EXPECTED_TRAINING_METADATA = {'run_protocol.json': '7bb4d7705553843c135b93c7a4a441f7f48b49830cd6aad165d31423e8ac8f06', 'original_baseline.json': '4fefc75b9ac40b7e3f016e30c0efdd64cc3bfada7ec2b9b4ae04311804ec64ce', 'original_only/training_report.json': '7a1a158360a70c5031cb5413e66cd8e4fbde9c790257aadfefec202a407d5103', 'original_only/reload_check.json': 'a7796ed371e5e240ff7f08ca0011bf09948ea6179768f8ca27ba3636216e46eb', 'original_only/baseline_validation.json': 'cb36e6b525be847700b8dea8b0cb38853f8699af0b32d4e23b84076bfa0a047c', 'with_finish/training_report.json': '657287c2128e32a19ee0547be6a146488eac38e66b9218aba8eb390f8a1cc7ff', 'with_finish/reload_check.json': '871d11ab23839222734520e43abdd65d3489ecb9661be29e47559065fc9df672', 'with_finish/baseline_validation.json': 'cb36e6b525be847700b8dea8b0cb38853f8699af0b32d4e23b84076bfa0a047c'}
EXPECTED_EXPORTS = {'original_only': {'policy_postprocessor_step_1_unnormalizer_processor.safetensors': '7008ba73c73887e5a7e631d478de9563ec5a0c51b7f0327f2530d7326c4b039e', 'policy_preprocessor_step_5_normalizer_processor.safetensors': '7008ba73c73887e5a7e631d478de9563ec5a0c51b7f0327f2530d7326c4b039e', 'policy_preprocessor.json': '29267ff4423fcce93bbd76ad0f386905f9de5d267ca13379a5795a514a81b7a9', 'model.safetensors': 'e6095e61f040da08a1d3c76fcd40d21b017983a2f92249798df4e60adf96d91f', 'policy_postprocessor.json': 'ad2d22405cb63be4932c4a4d9a69bc9953df02a01235c5f420b679dbe6bfe7c5', 'config.json': '9aec46548bb90d7564b9690cf42f235c79b8cf63e8f4cb4b29104212f1ff21ca'}, 'with_finish': {'policy_postprocessor_step_1_unnormalizer_processor.safetensors': '7008ba73c73887e5a7e631d478de9563ec5a0c51b7f0327f2530d7326c4b039e', 'policy_preprocessor_step_5_normalizer_processor.safetensors': '7008ba73c73887e5a7e631d478de9563ec5a0c51b7f0327f2530d7326c4b039e', 'policy_preprocessor.json': '29267ff4423fcce93bbd76ad0f386905f9de5d267ca13379a5795a514a81b7a9', 'model.safetensors': 'f4dd0d94d34839079846134f7887960d6fa7d99d1f623cb60329e2b749b3b6fe', 'policy_postprocessor.json': 'ad2d22405cb63be4932c4a4d9a69bc9953df02a01235c5f420b679dbe6bfe7c5', 'config.json': '9aec46548bb90d7564b9690cf42f235c79b8cf63e8f4cb4b29104212f1ff21ca'}}

MODELS=['original','original_only','with_finish']
CASES=[[1,'clean'],[5,'clean'],[16,'occluded'],[8,'clean']]+[[s,c] for s in [20,21,22] for c in ['clean','occluded']]
MAX_STEPS=440
WINDOW_STEPS=40

def read(path):return json.loads(Path(path).read_text())

def split_name(sid):
    return 'reserved_project_test' if sid>=20 else ('development_validation' if sid==8 else 'training_cases')

def completed(root,sid,condition,plan_hash):
    path=root/f'state_{sid:03d}'/condition/'result.json'
    if not path.exists():return None
    r=read(path)
    assert r['status']=='completed' and r['plan_hash']==plan_hash
    assert r['state_id']==sid and r['condition']==condition and r['control_steps']==MAX_STEPS
    for name,h in r['artifact_sha256'].items():
        p=(root/name).resolve();assert p.is_relative_to(root.resolve())
        assert sha_file(p)==h, '已完成回合文件变化：'+str(p)
    return r

def hold_window(rows):
    if len(rows)!=WINDOW_STEPS:return {'passed':False,'reason':'不足40步'}
    def diameter(points):
        a=np.asarray(points,float);return float(np.linalg.norm(a[:,None,:]-a[None,:,:],axis=-1).max())
    matrices=np.asarray([r['eef_matrix'] for r in rows]).reshape(-1,3,3)
    # trace(Ra.T Rb) is the elementwise inner product of the two matrices.
    cosine=np.clip((np.einsum('aij,bij->ab',matrices,matrices)-1)/2,-1,1)
    eef_angle=float(np.degrees(np.arccos(cosine)).max())
    q=np.asarray([r['bowl_quaternion'] for r in rows],float)
    q=q/np.linalg.norm(q,axis=1,keepdims=True)
    bowl_angle=float(np.degrees(2*np.arccos(np.clip(np.abs(q@q.T),0,1))).max())
    eef_d=diameter([r['eef_position'] for r in rows]);bowl_d=diameter([r['bowl_position'] for r in rows])
    checks={'goal_every_step':all(r['goal'] for r in rows),
            'no_gripper_bowl_contact':all(not r['gripper_bowl_contact'] for r in rows),
            'whole_gripper_clearance_5cm':all(r['clearance_lower_bound_m']>=.05 for r in rows),
            'eef_position_stable':eef_d<=.002,'bowl_position_stable':bowl_d<=.002,
            'eef_orientation_stable':eef_angle<=2.,'bowl_orientation_stable':bowl_angle<=2.}
    fingers=np.asarray([r['gripper_joint_positions'] for r in rows],float)
    return {'passed':all(checks.values()),'checks':checks,
            'eef_position_diameter_m':eef_d,'bowl_position_diameter_m':bowl_d,
            'eef_orientation_range_deg':eef_angle,'bowl_orientation_range_deg':bowl_angle,
            'minimum_clearance_lower_bound_m':min(r['clearance_lower_bound_m'] for r in rows),
            'gripper_joint_max_range_m_diagnostic':float(np.ptp(fingers,axis=0).max())}

def score_trace(trace):
    assert len(trace)==MAX_STEPS and [r['step'] for r in trace]==list(range(1,MAX_STEPS+1))
    first_goal=next((r['step'] for r in trace if r['goal']),None)
    windows=[hold_window(trace[i-WINDOW_STEPS:i]) for i in range(WINDOW_STEPS,len(trace)+1)]
    first=next((i for i,w in enumerate(windows) if w['passed']),None)
    first_end=first+WINDOW_STEPS if first is not None else None
    later_failures=[] if first is None else [i+WINDOW_STEPS for i in range(first+1,len(windows)) if not windows[i]['passed']]
    recontacts=[] if first_end is None else [r['step'] for r in trace[first_end:] if r['gripper_bowl_contact']]
    return {'benchmark_goal_within_280':first_goal is not None and first_goal<=280,
            'first_goal_step':first_goal,'goal_at_end':trace[-1]['goal'],
            'goal_steps_total':sum(r['goal'] for r in trace),
            'first_complete_40step_window_end':first_end,
            'last_40_steps':windows[-1],
            'later_failed_window_end_steps':later_failures,
            'recontact_steps_after_first_complete_window':recontacts,
            'complete_finish_automatic':first is not None and not later_failures,
            'automatic_label_note':'First full 40-step finish window must pass; every subsequent rolling 40-step window through step440 must also pass. Video review remains required.'}

def summarize(root,plan):
    rows=[]
    for model in MODELS:
        for sid,c in CASES:
            r=completed(root/model,sid,c,digest(plan))
            if r is not None:rows.append(r)
    counts=[]
    for model in MODELS:
        for split in ['training_cases','development_validation','reserved_project_test']:
            rs=[r for r in rows if r['model']==model and r['split']==split]
            counts.append({'model':model,'split':split,'completed':len(rs),
                'benchmark_goal_within_280':sum(r['scores']['benchmark_goal_within_280'] for r in rs),
                'complete_finish_automatic':sum(r['scores']['complete_finish_automatic'] for r in rs)})
    report={'completed':len(rows),'requested':30,'groups':counts,'manual_review':'pending',
            'note':'Reserved test: 3 paired states, two conditions; not six independent states. No checkpoint selection.'}
    write_json(root/'summary.json',report);return report

def validate_training(project):
    folder=project/'finetune/task0_finish_learning_pilot_v1'
    for name,h in EXPECTED_TRAINING_METADATA.items():
        assert sha_file(folder/name)==h,'14号训练记录变化：'+name
    common=read(folder/'run_protocol.json')
    for arm in ['original_only','with_finish']:
        t=read(folder/arm/'training_report.json');v=read(folder/arm/'reload_check.json')
        assert t['status']=='completed' and t['completed_updates']==t['selected_step']==50
        assert t['run_protocol']==dict(common,arm=arm)
        assert v['passed'] and v['max_absolute_loss_difference']==0
        assert v['run_protocol_hash']==digest(t['run_protocol'])
        assert t['export_check']['export_sha256']==v['export_sha256']==EXPECTED_EXPORTS[arm]
        assert t['export_check']['frozen_unchanged'] and t['export_check']['processors_identical_to_original']
        for name,h in EXPECTED_EXPORTS[arm].items():
            assert sha_file(folder/arm/'selected_model'/name)==h,'固定50步导出文件变化：'+arm+'/'+name
    assert common['dataset_protocol']['reserved_project_holdout_states']==[20,21,22]
    return common

def run_case(root,plan,model,sid,condition,base,env,kw):
    import torch
    from lerobot.scripts import lerobot_eval as ev
    folder=root/model/f'state_{sid:03d}'/condition
    attempt=folder/('attempt_'+uuid.uuid4().hex[:12]);attempt.mkdir(parents=True)
    write_json(attempt/'started.json',{'model':model,'state_id':sid,'condition':condition,
                                     'plan_hash':digest(plan),'policy_only':True})
    seed=42+sid;base._max_episode_steps=MAX_STEPS;base.init_state_id=sid
    base.metadata=dict(base.metadata,render_fps=20)
    env.unwrapped.metadata=dict(env.unwrapped.metadata,render_fps=20)
    policy=kw['policy'];policy.config.n_action_steps=1
    ev.set_seed(seed)
    for key in ['env_preprocessor','env_postprocessor','preprocessor','postprocessor']:
        if hasattr(kw[key],'reset'):kw[key].reset()
    wrapper=MaskInput(kw['env_preprocessor'],condition,attempt/'inputs')
    evidence=None;trace=[];final_frame=None
    original_step,original_writer=base.step,ev.write_video
    def step(action):
        nonlocal evidence,final_frame
        if evidence is None:
            evidence=SimulatorEvidence(base);write_json(attempt/'geometry_schema.json',evidence.schema())
        actual=np.asarray(action,dtype=np.float32).copy()
        assert actual.shape==(7,) and np.isfinite(actual).all()
        obs,reward,terminated,truncated,info=original_step(actual)
        measured=evidence.measure();assert measured['goal']==bool(info['is_success'])
        n=len(trace)+1
        trace.append(dict(measured,step=n,actual_action=actual.tolist(),action_source='policy',
                          original_terminated=bool(terminated),original_truncated=bool(truncated)))
        done=n>=MAX_STEPS
        if done:final_frame=np.asarray(base.render()).copy()
        if n%40==0:print(f'进度：{model}/{sid:03d}/{condition} {n}/440，当前盘上条件={measured["goal"]}',flush=True)
        # Continue the same episode after the benchmark goal; no scripted action or reset.
        return obs,reward,done,False,info
    def writer(path,frames,fps):
        assert fps==20 and len(frames)==MAX_STEPS and final_frame is not None
        return original_writer(path,np.concatenate([frames,final_frame[None]],axis=0),fps)
    base.step,ev.write_video=step,writer
    began=time.monotonic()
    try:
        info=ev.eval_policy(env=env,policy=policy,env_preprocessor=wrapper,
            **{k:kw[k] for k in ['env_postprocessor','preprocessor','postprocessor']},
            n_episodes=1,max_episodes_rendered=1,videos_dir=attempt/'videos',
            return_episode_data=False,start_seed=seed)
        torch.cuda.synchronize()
    finally:base.step,ev.write_video=original_step,original_writer
    assert len(trace)==wrapper.calls==MAX_STEPS and base.init_state_id==sid+1
    assert len(info['per_episode'])==1 and info['per_episode'][0]['seed']==seed
    write_json(attempt/'step_trace.json',trace);write_json(attempt/'eval_info.json',info)
    scores=score_trace(trace)
    if model=='original' and sid<20:
        historic=read(root.parent/'occlusion_center_task0_states0_19_v1'/f'state_{sid:03d}'/condition/'result.json')
        match=(scores['benchmark_goal_within_280']==historic['success'] and
               (scores['first_goal_step'] if scores['benchmark_goal_within_280'] else 280)==historic['control_steps'])
        assert match,'原模型前280步未复现历史结果；保留attempt用于诊断，停止后续案例。'
    videos=[]
    for item in info.get('video_paths',[]):
        path=Path(item)
        checked=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],capture_output=True,text=True)
        assert path.is_file() and checked.returncode==0 and not checked.stderr.strip(),'视频解码失败'
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(path)],text=True))['streams'][0]
        assert int(probe['nb_read_frames'])==MAX_STEPS+1 and probe['r_frame_rate']=='20/1'
        videos.append({'path':str(path.relative_to(root/model)),'decode_ok':True,'frames':MAX_STEPS+1,'fps':20})
    assert len(videos)==1
    hashes={str(p.relative_to(root/model)):sha_file(p) for p in [attempt/'step_trace.json',attempt/'geometry_schema.json',attempt/'eval_info.json']}
    for v in videos:hashes[v['path']]=sha_file(root/model/v['path'])
    result={'status':'completed','plan_hash':digest(plan),'model':model,'state_id':sid,
        'condition':condition,'split':split_name(sid),'seed':seed,'control_steps':MAX_STEPS,
        'policy_only':True,'n_action_steps':1,'scores':scores,'videos':videos,
        'attempt':str(attempt.relative_to(root/model)),'artifact_sha256':hashes,
        'manual_review':'pending','eval_seconds':time.monotonic()-began,
        'completed_utc':datetime.now(timezone.utc).isoformat()}
    write_json(folder/'result.json',result)
    print(f'已保存：{model}/{sid:03d}/{condition}；280步目标={scores["benchmark_goal_within_280"]}；完整收尾自动指标={scores["complete_finish_automatic"]}',flush=True)
    return result

def run_pairs(root,plan,model,max_new,audit,**kw):
    from lerobot.envs.libero import LiberoEnv
    envs=[(g,t,e) for g,v in kw['envs'].items() for t,e in v.items()]
    assert len(envs)==1 and envs[0][:2]==('libero_spatial',0)
    env=envs[0][2];assert env.num_envs==1 and hasattr(env,'envs')
    base=env.envs[0].unwrapped;assert isinstance(base,LiberoEnv) and base.control_freq==20
    states=np.asarray(base._init_states)
    compat={'initial_states_sha256':hashlib.sha256(states.tobytes()).hexdigest(),
            'shape':list(states.shape),'dtype':str(states.dtype),
            'audit':{k:v for k,v in audit.items() if k!='preview_only'}}
    historic=read(root.parent/'occlusion_center_task0_states0_19_v1/compatibility.json')
    for k in ['initial_states_sha256','shape','dtype']:assert compat[k]==historic[k],k
    for k in ['python','packages','gpu','cuda','eval_source_sha256','env_source_sha256']:
        assert compat['audit'][k]==historic['audit'][k],'运行环境与旧评估不一致：'+k
    # Match all three models and every resumed session to one environment.
    cp=root/'compatibility.json'
    if cp.exists():assert read(cp)==compat,'评估环境改变，停止覆盖'
    else:write_json(cp,compat)
    assert states.shape[0]>22
    count=0
    for sid,c in CASES:
        if completed(root/model,sid,c,digest(plan)) is not None:continue
        if count>=max_new:break
        run_case(root,plan,model,sid,c,base,env,kw);count+=1
        print('累计完成：',summarize(root,plan)['completed'],'/30',flush=True)
    return {'overall':summarize(root,plan)}
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--max-new", type=int, default=10)
    p.add_argument("--status", action="store_true")
    p.add_argument("--preview", action="store_true")
    a = p.parse_args()
    root=Path(a.root)
    if a.status:
        print(json.dumps(summarize(root,read(root/'plan.json')),indent=2,ensure_ascii=False));return
    import fcntl
    root.mkdir(parents=True,exist_ok=True)
    lock=open('/tmp/finish_eval_'+hashlib.sha256(str(root.resolve()).encode()).hexdigest()+'.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert md.version('lerobot')=='0.6.1' and a.max_new==10
    import torch
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='Tesla T4'
    from huggingface_hub import snapshot_download
    project=root.parent.parent
    training=validate_training(project)
    prior=read(project/'results/occlusion_center_task0_states0_19_v1/plan.json')
    source_plan=read(project/'results/finish_demo_batch_v1/original/plan.json')
    assert prior['model_revision']==training['dataset_protocol']['model_revision']
    assert prior['vlm_revision']==training['dataset_protocol']['vlm_revision']
    assert prior['suite']=='libero_spatial' and prior['task_id']==0 and prior['seed_base']==42
    assert prior['mask']=={'camera':'observation.images.image','box_yxyx':[96,160,96,160],'fill':0.5,'timing':'every_policy_observation'}
    original_hashes=source_plan['export_sha256']
    model_hashes={'original':original_hashes,**EXPECTED_EXPORTS}
    expected={'protocol':'finish_policy_eval_v1','cases':CASES,'models':MODELS,'model_files':model_hashes,
        'training_metadata_sha256':EXPECTED_TRAINING_METADATA,'training_protocol':training['dataset_protocol'],
        'suite':'libero_spatial','task_id':0,'seed_base':42,'mask':prior['mask'],
        'model_repo':prior['model_repo'],'model_revision':prior['model_revision'],
        'vlm_repo':prior['vlm_repo'],'vlm_revision':prior['vlm_revision'],
        'assets_repo':prior['assets_repo'],'assets_revision':prior['assets_revision'],
        'control_hz':20,'max_steps':440,'benchmark_horizon':280,'n_action_steps':1,
        'evaluation_actions':'policy only for all440 steps; no teacher, no post-goal stop, no action override',
        'finish_rule':{'window_steps':40,'goal_all':True,'no_contact':True,'clearance_m':.05,
                       'position_diameter_m':.002,'orientation_range_deg':2.,
                       'after_first_passing_window':'every later rolling40 window must pass through440',
                       'finger_motion':'recorded diagnostic plus manual video review; no invented threshold'},
        'manual_review':'release/no adhesion, whole-arm/finger jitter, withdrawal and recontact; automatic metrics alone are provisional',
        'test_limits':'3 paired reserved states20,21,22, not6 independent states; prior pretraining overlap unknown',
        'runner_sha256':sha_file(__file__)}
    path=root/'plan.json'
    if path.exists():assert read(path)==expected,'评估方案变化，保留结果并停止'
    else:write_json(path,expected)
    plan=expected
    if all(completed(root/a.model,s,c,digest(plan)) is not None for s,c in CASES):
        print(a.model+'全部10回合已完成，校验后跳过。',flush=True);return
    if a.model=='original':
        model_source=Path(snapshot_download(prior['model_repo'],revision=prior['model_revision'],
            cache_dir='/content/smolvla_hf_cache',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','*.tiktoken']))
    else:model_source=project/'finetune/task0_finish_learning_pilot_v1'/a.model/'selected_model'
    hashes=model_hashes[a.model]
    for name,h in hashes.items():assert sha_file(model_source/name)==h,'模型校验失败：'+name
    cache=Path("/content/smolvla_hf_cache")
    vlm=snapshot_download(plan["vlm_repo"],revision=plan["vlm_revision"],cache_dir=cache,
        allow_patterns=["*.json","*.txt","*.model","*.jinja","*.tiktoken"])
    # 评估使用完整导出模型；复制到本地，原始导出文件不修改。
    model=Path("/content/smolvla_finish_eval_"+a.model)
    model.mkdir(exist_ok=True)
    for name in hashes:
        target=model/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() or sha_file(target)!=hashes[name]:
            shutil.copyfile(model_source/name,target)
    # 分词器同样固定到锁定版本，避免默默使用远程最新版本。
    proc_path=model/"policy_preprocessor.json"
    proc=json.loads(proc_path.read_text())
    def fix_tokenizer(x):
        count=0
        if isinstance(x,dict):
            for key,value in x.items():
                if key=="tokenizer_name":x[key]=str(vlm);count+=1
                else:count+=fix_tokenizer(value)
        elif isinstance(x,list):
            for value in x:count+=fix_tokenizer(value)
        return count
    assert fix_tokenizer(proc)==1, "分词器配置结构变化，停止"
    write_json(proc_path,proc)
    import libero.libero as libero_core
    snapshot_download(plan["assets_repo"], repo_type="dataset", revision=plan["assets_revision"],
                      local_dir=str(Path(libero_core.__file__).parent / "assets"), max_workers=4)
    from lerobot.scripts import lerobot_eval as ev
    from lerobot.envs.libero import LiberoEnv
    packages = {d.metadata["Name"].lower(): d.version for d in md.distributions() if d.metadata["Name"]}
    audit = {"python": platform.python_version(), "packages": packages,
             "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
             "eval_source_sha256": sha_file(inspect.getfile(ev)),
             "env_source_sha256": sha_file(inspect.getfile(LiberoEnv)),
             "runner_sha256": sha_file(__file__), "preview_only": a.preview}
    session = root / "sessions" / uuid.uuid4().hex[:12]
    session.mkdir(parents=True)
    write_json(session / "environment.json", audit)
    freeze = subprocess.check_output(["uv", "pip", "freeze", "--python", sys.executable], text=True)
    (session / "requirements.txt").write_text(freeze)
    if not (root / "requirements.txt").exists():
        (root / "requirements.txt").write_text(freeze)
    # 保留官方模型加载和处理流程，仅替换批量调度，单次任务仍用官方 eval_policy。
    ev.eval_policy_all = lambda **kw: run_pairs(root, plan, a.model, a.max_new, audit, **kw)
    sys.argv = [sys.argv[0], f"--policy.path={model}", f"--policy.vlm_model_name={vlm}",
        "--policy.device=cuda", "--policy.use_amp=false", "--policy.load_vlm_weights=false",
        "--policy.n_action_steps=1", "--env.type=libero", "--env.task=libero_spatial",
        "--env.task_ids=[0]", "--env.observation_height=256", "--env.observation_width=256",
        "--env.control_mode=relative", "--env.init_states=true", "--env.max_parallel_tasks=1",
        "--eval.batch_size=1", "--eval.n_episodes=1", "--eval.use_async_envs=false",
        "--seed=42", f"--output_dir={session}"]
    write_json(session / "command.json", sys.argv)
    ev.main()


if __name__ == "__main__":
    main()
