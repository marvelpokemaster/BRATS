"""Resumable advisor diagnostics for the actual Part 1 graph; descriptive only."""
from pathlib import Path
import json
import random
import time
import numpy as np
import torch
from brats_experiments import BudgetPause, case_id
from brats_workflow import export_reports_zip


def distribution(values, bins):
    values=np.asarray(values,dtype=float)
    return dict(n=int(values.size),mean=float(values.mean()) if values.size else None,
        std=float(values.std()) if values.size else None,hist=np.histogram(values,bins)[0].tolist(),bins=list(map(float,bins)))


@torch.no_grad()
def node_diagnostics(c,item):
    graph=item[0].clone().to(c['device']);model=c['model'];model.eval()
    model(graph,teacher_prob=0.) if hasattr(model,'base_model') else model(graph)
    base=c['base_model']; hops={};gates={};truth={}
    for nt in c['NODE_TYPES']:
        y=graph[nt].y.cpu().numpy();truth[nt]=y
        values=base.last_hop_info[nt]['effective_hop'].cpu().numpy()
        masks=dict(BG=y==0,WT=y>0,TC=(y==1)|(y==3),ET=y==3)
        hops[nt]={region:distribution(values[mask],np.linspace(0,c['K_MAX'],41)) for region,mask in masks.items()}
    for et in graph.edge_types:
        steps=[step[et].detach().cpu().numpy().reshape(-1) for step in base.last_edge_gates if et in step and step[et].numel()]
        if not steps:continue
        alpha=np.mean(steps,axis=0);src,dst=graph[et].edge_index.cpu().numpy()
        boundary=(truth[et[0]][src]>0)!=(truth[et[2]][dst]>0)
        gates['-'.join(et)]={name:distribution(alpha[mask],np.linspace(0,1,41))
                            for name,mask in [('boundary',boundary),('interior',~boundary)]}
    return dict(case_id=case_id(item),hops=hops,gates=gates)


def reconstruction_diagnostic(c,item,seed):
    if c['rec_module'] is None:return None
    result=c['rec_recover_case'](item[0].clone(),seed=seed)
    rows={}
    for nt in c['NODE_TYPES']:
        if nt not in result['pred']:continue
        target=result['target'][nt];pred=result['pred'][nt];idx=result['idx'][nt]
        appearance=item[0][nt].x[:,:c['APPEARANCE_DIM']].cpu().numpy()
        unmasked=np.ones(len(appearance),bool);unmasked[idx]=False
        mean=appearance[unmasked].mean(0) if unmasked.any() else np.zeros(c['APPEARANCE_DIM'])
        src,dst=item[0][nt,'spatial',nt].edge_index.cpu().numpy();keep=unmasked[src]
        sums=np.zeros_like(appearance);counts=np.zeros(len(appearance))
        np.add.at(sums,dst[keep],appearance[src[keep]]);np.add.at(counts,dst[keep],1)
        neighbors=np.repeat(mean[None,:],len(idx),axis=0);has=counts[idx]>0
        neighbors[has]=sums[idx[has]]/counts[idx[has],None]
        rows[nt]=dict(masked_nodes=len(idx),mae_model=float(np.abs(pred-target).mean()),
            mae_unmasked_mean=float(np.abs(mean-target).mean()),mae_unmasked_neighbors=float(np.abs(neighbors-target).mean()))
    link=None
    if 'link' in result:
        from scipy.stats import rankdata
        data=result['link'];n=len(data['s_pos']);m=len(data['s_neg'])
        def auc(a,b):
            ranks=rankdata(np.concatenate((a,b)),method='average')
            return float((ranks[:n].sum()-n*(n+1)/2)/(n*m)) if n and m else None
        link=dict(positive_edges=n,negative_edges=m,model_auc=auc(data['s_pos'],data['s_neg']),geometry_auc=auc(data['g_pos'],data['g_neg']))
    return dict(case_id=case_id(item),seed=seed,appearance=rows,correspondence=link)


def run_advisor_reports(c,store):
    previous=store.read('report_status.json')
    if previous and previous.get('complete') and (store.api is None or store.read('final_export_receipt.json') is not None):
        return previous
    def check():
        if time.time()>=c['SESSION_DEADLINE']:raise BudgetPause('Advisor reports paused; rerun Part 5')
    status=dict(complete=False,paused=False)
    test=c['test_items'];selected=random.Random(c['SEED']+703).sample(test,min(10,len(test)))
    store.save('reports/scope.json',dict(stage1_sha256=c['STAGE1_MODEL_SHA256'],
        graph_scope='Graph branch only, fixed topology; no CNN/SLIC attribution',
        hop_scope='Node-majority regions, unweighted node histograms; additional propagation after HGT',
        xai_scope='Exact four-player feature-replacement Shapley of WT/TC/ET probabilities, fractional region-volume weighting',
        xai_patient_ids=list(map(case_id,selected)),selection='Fixed random seed; descriptive cohort, no directional significance claim'),push=False)
    try:
        for item in test:
            name='reports/nodes/'+case_id(item)+'.json'
            if store.read(name) is None:
                check();store.save(name,node_diagnostics(c,item),push=True)
        for i,item in enumerate(selected):
            name='reports/xai/'+case_id(item)+'.json'
            if store.read(name) is None:
                check();results=c['exact_modality_shapley'](item,c['xai_background'])
                values={}
                for j,region in enumerate(('WT','TC','ET')):
                    mass=sum(r['region_voxel_weight'][j] for r in results.values())
                    values[region]=(sum(r['region_attribution_sum'][j] for r in results.values())/mass).tolist() if mass>0 else None
                error=max(r['region_efficiency_error'] for r in results.values())
                if not np.isfinite(error) or error>1e-4:raise RuntimeError('Region Shapley efficiency check failed')
                store.save(name,dict(case_id=case_id(item),regions=values,efficiency_error=error),push=True)
            rec_name='reports/reconstruction/'+case_id(item)+'.json'
            if c['rec_module'] is not None and store.read(rec_name) is None:
                check();store.save(rec_name,reconstruction_diagnostic(c,item,c['REC_VAL_SEED']+i),push=True)
        import matplotlib.pyplot as plt
        xai=[store.read('reports/xai/'+case_id(item)+'.json') for item in selected]
        summary={}
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,region in zip(axes,('WT','TC','ET')):
            values=np.asarray([row['regions'][region] for row in xai if row['regions'][region] is not None])
            mean=values.mean(0) if len(values) else np.zeros(4);std=values.std(0) if len(values) else np.zeros(4)
            summary[region]=dict(n_present=len(values),mean=mean.tolist(),std=std.tolist())
            ax.bar(c['MODALITIES'],mean,yerr=std,capsize=3);ax.axhline(0,color='black',lw=.8)
            ax.set(title=region+' graph probability',ylabel='Shapley contribution (patient mean +/- SD)')
        fig.tight_layout();name='reports/region_shapley.png';fig.savefig(store.path(name),dpi=220);plt.close(fig);store.push(name)
        store.save('reports/region_shapley_summary.json',summary)
        nodes=[store.read('reports/nodes/'+case_id(item)+'.json') for item in test]
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        hop_summary={}
        for ax,region in zip(axes,('WT','TC','ET')):
            for nt in c['NODE_TYPES']:
                rows=[row['hops'][nt][region] for row in nodes]
                counts=np.sum([row['hist'] for row in rows],axis=0);bins=np.asarray(rows[0]['bins'])
                ax.stairs(counts/max(1,counts.sum()),bins,label=nt)
                patient_means=[row['mean'] for row in rows if row['mean'] is not None]
                hop_summary[nt+'/'+region]=dict(patient_mean=float(np.mean(patient_means)) if patient_means else None,
                    patient_sd=float(np.std(patient_means)) if patient_means else None,n_present=len(patient_means))
            ax.set(title=region,xlabel='Additional effective hops after HGT',ylabel='Fraction of nodes');ax.legend()
        fig.tight_layout();name='reports/hop_distributions.png';fig.savefig(store.path(name),dpi=220);plt.close(fig);store.push(name)
        store.save('reports/hop_summary.json',hop_summary)
        local_dir=Path(c['PERSISTENT_BASE'])/'figures';local_dir.mkdir(exist_ok=True)
        if store.pull('reports/local_graph.png') is None:
            check();c['explain_local_hetero_graph'](selected[0]);c['global_graph_explanation_summary'](selected[0])
            for leaf in ('local_graph.png','gate_hop_summary.png'):
                store.add_file(local_dir/leaf,'reports/'+leaf)
        research=c['RESEARCH_STORE_FACTORY'](c['RESEARCH_STUDY_KEY'])
        for name in research.catalog['files'] if hasattr(research,'catalog') else []:
            if name.startswith(('reports/','evaluation/')) or name=='protocol.json':
                check();store.add_file(research.pull(name),'research/'+name)
        main_store=c.get('MAIN_STORE')
        if main_store is not None:
            for name in main_store.catalog['files']:
                if name.startswith('figures/') or name=='gpu_runtime.json':
                    check();store.add_file(main_store.pull(name),'main/'+name)
        store.save('reports/main_training_history.json',dict(graph=c['stage1_checkpoint'].get('history',[]),
            refinement=c['MAIN_STAGE2_STATE']['vox_history'],graph_best_epoch=c['stage1_checkpoint']['best_epoch'],
            refinement_best_epoch=c['MAIN_STAGE2_STATE']['best_vox_epoch']))
        pilot = main_store.read('slic_pilot_receipt.json') if main_store is not None else None
        store.save('reports/slic_pilot_scope.json',pilot or {'complete':False,'message':'SLIC pilot not supplied'})
        if pilot:
            pilot_store=c['RESEARCH_STORE_FACTORY'](pilot['plan'])
            for name in pilot_store.catalog['files']:
                if name.startswith(('reports/','evaluation/')) or name=='protocol.json':
                    check();store.add_file(pilot_store.pull(name),'slic_pilot/'+name)
        status['advisor_evidence_complete']=bool(pilot and pilot.get('complete'))
        status['publication_ready_certified']=False
        store.save('reports/claims_scope.json',dict(single_seed=True,training_is_compute_limited=True,
            multi_seed_reproducibility_established=False,slic_is_validation_pilot=True,
            journal_readiness_requires_assessment_of_actual_curves_and_results=True))
        archive=export_reports_zip(store.root,store.path('final_reports.zip'))
        store.push('final_reports.zip')
        status.update(complete=True,local_export=str(archive))
        store.save('report_status.json',status)
        store.flush()
        if store.api is not None:
            from brats_transfer import upload_verified,sha256_path
            saved=store.read('final_export_receipt.json')
            if saved is None or saved.get('sha256')!=sha256_path(archive):
                receipt=upload_verified(store.api,lambda **kw:store.download(kw['filename'],kw['revision']),
                    archive,'continuation_v8/'+c['STAGE1_MODEL_SHA256'][:20]+'/final_reports.zip',
                    store.repo_id,store.repo_type,'Final BraTS report ZIP')
                store.save('final_export_receipt.json',receipt)
            print('Final report ZIP is in your model repository under continuation_v8/'+c['STAGE1_MODEL_SHA256'][:20]+'/final_reports.zip')
    except BudgetPause as exc:
        status.update(paused=True,message=str(exc));store.save('report_status.json',status)
    finally:
        store.flush()
    return status
