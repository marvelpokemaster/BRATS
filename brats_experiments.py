"""Predeclared, resumable BraTS component experiments. No work at import time.

Training/selection use train/validation only. Test evaluation is a separate,
explicit phase after every planned run is complete. Sources/papers are not
reproductions: see RESEARCH_PROTOCOL.md for precise comparison scope.
"""
from pathlib import Path
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import time
import zipfile
import numpy as np
import torch
from brats_transfer import sha256_path, write_json_atomic, extract_verified_zip
from brats_protocol import make_training_patch, sliding_window_predict, mean_region_dice, segmentation_metrics, summarize_metric_rows
from brats_gpu import configure_gpu, bounded_prefetch, patient_groups, train_patient_group

RESEARCH_VERSION = "research-v8-practical"


class BudgetPause(RuntimeError):
    pass


class Deadline:
    def __init__(self, hours, start=None):
        self.end = (time.time() if start is None else start) + float(hours) * 3600

    def check(self):
        if time.time() >= self.end:
            raise BudgetPause("Session work budget reached; export an incomplete-study status rather than silently extending the five-session plan")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def model_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    torch.set_rng_state(state["torch"]); np.random.set_state(state["numpy"]); random.setstate(state["python"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def case_id(item):
    return Path(item[1]).name.removesuffix(".meta.pt")


def frozen_subset(groups, maximum=0, seed=42):
    selected = {}
    for i, (name, items) in enumerate(groups.items()):
        items = sorted(items, key=case_id)
        if maximum:
            items = sorted(random.Random(seed + i).sample(items, min(maximum, len(items))), key=case_id)
        ids = [case_id(item) for item in items]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Empty or duplicate research split: " + name)
        selected[name] = items
    sets = [set(map(case_id, items)) for items in selected.values()]
    if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]):
        raise ValueError("Research patient splits overlap")
    return selected


def experiment_plan(n_segments, hops, k_max=2, masked_reconstruction=False):
    if masked_reconstruction:
        raise ValueError('Practical v8 fixes reconstruction off; use a new declared study for extensions.')
    full=dict(backbone='hgt',structural=False,descriptors=False,hop_mode='adaptive',fixed_hops=hops,
              cross_edges=True,soft_labels=True,voxel=True,boundary=False,focal=False,hop_penalty=True,
              n_segments=n_segments,masked_reconstruction=False,voxel_kind='refinement')
    changes=[('full',{},'Practical HGT + learned 0..2 hop mixture + voxel refinement'),
        ('no_voxel',{'voxel':False},'Same selected HGT graph; no extra training'),
        ('cnn_only',{'backbone':'none'},'Same refinement CNN using four MRI channels only'),
        ('graphsage_backbone',{'backbone':'sage','voxel':False},'Ordinary GraphSAGE graph-only control; compare against no_voxel'),
        ('fixed_shared_hops',{'hop_mode':'fixed','voxel':False},'Fixed shared two-hop graph-only control; compare against no_voxel'),
        ('segresnet',{'backbone':'none','voxel_kind':'segresnet'},'Independent MONAI SegResNet voxel baseline')]
    changes += [(f'slic_{n}',{'n_segments':n,'voxel':False},'Validation-only SLIC pilot')
                for n in (5000,10000,15000,20000) if n!=n_segments]
    return [dict(name=name,description=description,**dict(full,**delta)) for name,delta,description in changes]


class ArtifactStore:
    """Local atomic files plus verified Hub receipts; no token stored in files."""
    def __init__(self, root, prefix, upload=None, receipt=None, download=None):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix.strip("/")
        self.upload, self.receipt, self.download = upload, receipt, download

    def path(self, name):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def push(self, name):
        if self.upload is None:
            return
        remote = self.prefix + "/" + name
        self.upload(str(self.path(name)), remote)
        receipt = self.receipt(remote)
        pointer = self.path(name + ".receipt.json")
        write_json_atomic(pointer, receipt)
        self.upload(str(pointer), remote + ".receipt.json")

    def pull(self, name):
        local = self.path(name)
        if local.exists() or self.download is None:
            return local if local.exists() else None
        remote = self.prefix + "/" + name
        pointer = self.download(remote + ".receipt.json", "main")
        if pointer is None:
            return None
        receipt = json.loads(Path(pointer).read_text(encoding="utf-8"))
        artifact = self.download(remote, receipt["revision"])
        if artifact is None or sha256_path(artifact) != receipt["sha256"]:
            raise RuntimeError("Research artifact checksum mismatch: " + name)
        import shutil
        shutil.copyfile(artifact, str(local) + ".tmp")
        os.replace(str(local) + ".tmp", local)
        write_json_atomic(self.path(name + ".receipt.json"), receipt)
        return local

    def save(self, name, value, push=False):
        path = self.path(name)
        if name.endswith(".pt"):
            torch.save(value, str(path) + ".tmp"); os.replace(str(path) + ".tmp", path)
        else:
            write_json_atomic(path, value)
        if push:
            self.push(name)

    def read(self, name):
        path = self.pull(name)
        if path is None:
            return None
        return (torch.load(path, map_location="cpu", weights_only=False) if name.endswith(".pt")
                else json.loads(path.read_text(encoding="utf-8")))


from brats_practical import train_phase


def paired_summary(records, reference="full", draws=5000, seed=1942):
    """Patient-paired bootstrap after averaging repeated seeds per patient.

    Seed SD is reported separately. Patients, not seed-patient pairs, are
    resampled. CIs are conditional on this cohort/protocol and chosen seeds.
    """
    keys = ("Dice_WT", "Dice_TC", "Dice_ET", "Dice_mean")
    grouped = {}
    for record in records:
        name, sd, cid = record["experiment"], int(record["seed"]), record["case_id"]
        scores = dict(record["metrics"])
        scores["Dice_mean"] = float(np.mean([scores[k] for k in keys[:3]]))
        bucket = grouped.setdefault(name, {}).setdefault(sd, {})
        if cid in bucket:
            raise ValueError("Duplicate patient/seed result")
        if not all(np.isfinite(scores[k]) for k in keys):
            raise ValueError("Non-finite result")
        bucket[cid] = scores
    if reference not in grouped:
        raise ValueError("Reference experiment missing")
    seeds = sorted(grouped[reference]); ids = sorted(grouped[reference][seeds[0]])
    for runs in grouped.values():
        if sorted(runs)!=seeds or any(sorted(run)!=ids for run in runs.values()):
            raise ValueError("Comparisons require identical patients and seeds")
    rng = np.random.default_rng(seed)
    result = {"n_patients":len(ids), "seeds":seeds, "ci_unit":"patient after seed averaging",
              "ci_scope":"conditional on this cohort, training protocol and fixed seed set", "comparisons":{}}
    for name, runs in grouped.items():
        if name==reference:
            continue
        comparison = {}
        for key in keys:
            ref = np.array([[grouped[reference][sd][cid][key] for cid in ids] for sd in seeds])
            other = np.array([[runs[sd][cid][key] for cid in ids] for sd in seeds])
            delta = (ref-other).mean(axis=0)
            boots, permuted = [], 0
            observed = abs(float(delta.mean()))
            for _ in range(draws):
                boots.append(float(rng.choice(delta,len(delta),replace=True).mean()))
                permuted += abs(float((delta*rng.choice((-1,1),len(delta))).mean())) >= observed - 1e-15
            comparison[key] = dict(reference_minus_control=float(delta.mean()),
                ci95=np.quantile(boots,[.025,.975]).tolist(),
                paired_sign_flip_p=(permuted+1)/(draws+1),
                reference_seed_mean_sd=float(ref.mean(1).std(ddof=1)) if len(seeds)>1 else None,
                control_seed_mean_sd=float(other.mean(1).std(ddof=1)) if len(seeds)>1 else None)
        result["comparisons"][name] = comparison
    family = [metric for row in result["comparisons"].values() for metric in row.values()]
    previous = 0.
    for rank, metric in enumerate(sorted(family,key=lambda v:v["paired_sign_flip_p"])):
        previous = max(previous,min(1.,(len(family)-rank)*metric["paired_sign_flip_p"]))
        metric["holm_adjusted_p"] = previous
    return result


def controlled_hop_mix(self, x_dict, edge_index_dict, edge_attr_dict):
    """Fixed/uniform controls reuse the proposed model's SAME propagation layer."""
    states = [x_dict]
    steps = self.control_depth if self.control_mode=="fixed" else self.k_max
    current = x_dict
    for _ in range(steps):
        logits = {nt:self.pre_head[nt](current[nt]) for nt in current}
        from torch.utils.checkpoint import checkpoint as _checkpoint
        current = (_checkpoint(self.adaptive_prop,current,edge_index_dict,edge_attr_dict,logits,use_reentrant=False)
                   if self.training and torch.is_grad_enabled() else self.adaptive_prop(current,edge_index_dict,edge_attr_dict,logits))
        states.append(current)
    self.hop_regularization = torch.zeros((),device=next(iter(current.values())).device)
    if self.control_mode=="fixed":
        return states[-1]
    return {nt:torch.stack([state[nt] for state in states]).mean(0) for nt in current}


def apply_hop_control(model, mode, depth):
    if mode=="adaptive":
        return model
    if mode not in ("fixed","uniform") or not 0<=depth<=model.k_max:
        raise ValueError("Invalid shared-hop control")
    from types import MethodType
    model.control_mode, model.control_depth = mode, depth
    model._adaptive_hop_mix = MethodType(controlled_hop_mix,model)
    for parameter in model.hop_selector.parameters():
        parameter.requires_grad_(False)
    if mode=="fixed" and depth==0:
        for module in (model.adaptive_prop,model.pre_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return model


class HomogeneousSAGE(torch.nn.Module):
    """All relation edges merged; matched features/targets, GraphSAGE-pool layers."""
    def __init__(self, node_types, in_dim, hidden_dim, num_classes, layers, dropout):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.node_types = node_types
        self.layers = torch.nn.ModuleList([SAGEConv(in_dim if i==0 else hidden_dim,hidden_dim,
            aggr="max",project=True) for i in range(layers)])
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_dim,num_classes)

    def forward(self,data):
        sizes = [data[nt].x.shape[0] for nt in self.node_types]
        offsets = dict(zip(self.node_types,np.cumsum([0]+sizes[:-1]).tolist()))
        x = torch.cat([data[nt].x for nt in self.node_types])
        edges = [data[et].edge_index + torch.tensor([[offsets[et[0]]],[offsets[et[2]]]],device=x.device)
                 for et in data.edge_types]
        edge_index = torch.cat(edges,1)
        for layer in self.layers:
            x = self.drop(torch.relu(layer(x,edge_index)))
        embeddings = dict(zip(self.node_types,x.split(sizes)))
        return {nt:self.head(h) for nt,h in embeddings.items()}, embeddings


def save_imported_selection(store,name,signature,model,epoch,best_epoch,score,history,runtime,lr,decay,epochs,origin,termination=None):
    # Complete selections are inference-only; these placeholder optimizer states
    # are never used to continue training imported models.
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=lr,weight_decay=decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=lr*.02)
    scaler=torch.amp.GradScaler("cuda",enabled=next(model.parameters()).device.type=="cuda" and runtime["precision"]=="fp16")
    weights=cpu_state(model)
    store.save(name,dict(signature=signature,epoch=epoch,model=weights,best=weights,optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),best_epoch=best_epoch,
        best_score=score,stale=0,history=history,complete=True,runtime=runtime,precision_validated=True,origin=origin,**(termination or {})),push=True)


class IndependentUNet(torch.nn.Module):
    """Lightweight independent 3D U-Net for four-channel BraTS volumes."""
    def __init__(self, in_channels=4, num_classes=4, base_channels=8):
        super().__init__()
        self.channels_last_3d = False
        self.amp_enabled = False
        self.amp_dtype = None
        def block(cin, cout):
            return torch.nn.Sequential(
                torch.nn.Conv3d(cin, cout, 3, padding=1, bias=False),
                torch.nn.InstanceNorm3d(cout, affine=True),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Conv3d(cout, cout, 3, padding=1, bias=False),
                torch.nn.InstanceNorm3d(cout, affine=True),
                torch.nn.LeakyReLU(inplace=True),
            )
        self.enc1 = block(in_channels, base_channels)
        self.enc2 = block(base_channels, base_channels * 2)
        self.bottleneck = block(base_channels * 2, base_channels * 4)
        self.pool = torch.nn.MaxPool3d(2)
        self.up2 = torch.nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = block(base_channels * 4, base_channels * 2)
        self.up1 = torch.nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = block(base_channels * 2, base_channels)
        self.out = torch.nn.Conv3d(base_channels, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.up2(b)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = torch.nn.functional.interpolate(d2, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))
        return self.out(d1), None


class ResearchRunner:
    def __init__(self, context, settings):
        self.c, self.settings = context, settings
        c = self.c
        self.deadline = Deadline(settings["budget_hours"],settings.get("session_start"))
        self.deadline.phase_seconds = 2700 if settings.get("pilot") else 8100
        self.groups = frozen_subset(dict(train=c["train_items"],val=c["val_items"],test=c["test_items"]),
                                    settings["max_cases_per_split"],c["SEED"])
        all_rows = experiment_plan(c["N_SEGMENTS"],c["HOPS"],c["K_MAX"],c["USE_MASKED_RECONSTRUCTION"])
        requested = settings["plan_names"] or [row["name"] for row in all_rows]
        if len(set(requested))!=len(requested) or set(requested)-{row["name"] for row in all_rows}:
            raise ValueError("Unknown/duplicate experiment name in frozen plan")
        if "full" not in requested:
            raise ValueError("Include full in the frozen comparison plan")
        self.plan = [row for row in all_rows if row["name"] in requested]
        if settings.get('pilot'):
            self.groups = {split: sorted(random.Random(c['SEED']+i).sample(items,min(cap,len(items))),key=case_id)
                for i,(split,items,cap) in enumerate((('train',self.groups['train'],64),('val',self.groups['val'],16)))}
            self.groups['test']=[]
            self.plan=[dict(row,voxel=False) for row in self.plan]
        self.seeds = tuple(int(seed) for seed in settings["seeds"])
        if not self.seeds or len(set(self.seeds))!=len(self.seeds):
            raise ValueError("Supply distinct research seeds")
        if c["USE_MASKED_RECONSTRUCTION"] and not c["REC_SEPARATE_CLEAN_PASS"]:
            raise ValueError("Continuation research requires Part 1's declared separate clean reconstruction pass")
        if c["USE_STRUCTURAL_REFINEMENT"] or not c["USE_ADAPTIVE_HOPS"]:
            raise ValueError("Practical v8 requires structural refinement off and adaptive hops on")
        source_files = [Path(__file__).with_name("brats_practical.py"),Path(__file__),Path(__file__).with_name("brats_gpu.py"),Path(__file__).with_name("brats_protocol.py"),Path(__file__).with_name("brats_graph.py")]
        self.manifest = dict(version=RESEARCH_VERSION, compute_protocol=dict(phase_seconds=2700 if settings.get('pilot') else 8100, single_seed=True, pilot=bool(settings.get('pilot'))), rows=self.plan,seeds=list(self.seeds),
            split={name:list(map(case_id,items)) for name,items in self.groups.items()},
            dataset=c["DATASET_FINGERPRINT"],run_config=c["RUN_CONFIG"],
            pipeline_definitions_sha256=c.get("RESEARCH_PIPELINE_SHA256","synthetic-test-fixture"),
            postprocess=dict(enabled=c["USE_CC_POSTPROCESS"],et_min_voxels=c["ET_MIN_VOXELS"]),
            structural_teacher=dict(start=c["STRUCTURAL_TEACHER_PROB_START"],end=c["STRUCTURAL_TEACHER_PROB_END"]),
            ssl=dict(enabled=c["USE_MASKED_RECONSTRUCTION"], settings={k:c.get(k) for k in
                ("LAMBDA_REC","REC_WARMUP_EPOCHS","REC_MASK_RATE","REC_EDGE_MASK_RATE","REC_FLAGS","REC_NEG_PER_POS","REC_FEAT_LOSS","REC_SCE_GAMMA","REC_SEPARATE_CLEAN_PASS")}),
            class_weights=c["class_weights"].cpu().tolist(),source_hashes={p.name:sha256_path(p) for p in source_files},
            epochs=c["EPOCHS"],voxel_epochs=c["VOXEL_EPOCHS"],patience=c["EARLY_STOP_PATIENCE"],
            debug=bool(settings["max_cases_per_split"]),selection="validation full-volume postprocessed mean WT/TC/ET Dice",
            metric_policy="custom finite empty-region HD95; see RESEARCH_PROTOCOL.md")
        self.key = identity(self.manifest)[:20]
        remote = "research_v8/"+self.key
        upload = receipt = download = None
        if c["HF_ENABLED"]:
            upload = lambda local,name:c["hf_upload_file_verified"](local,name,c["HF_MODEL_REPO_ID"],c["HF_MODEL_REPO_TYPE"],"research checkpoint/artifact")
            receipt = lambda name:c["hf_upload_receipt"](c["HF_MODEL_REPO_ID"],name)
            download = lambda name,revision:c["hf_try_download"](name,c["HF_MODEL_REPO_ID"],c["HF_MODEL_REPO_TYPE"],revision=revision)
        self.store = (c["RESEARCH_STORE_FACTORY"](self.key) if c.get("RESEARCH_STORE_FACTORY") else
                      ArtifactStore(Path(c["PERSISTENT_BASE"])/"research_v8"/self.key,remote,upload,receipt,download))
        self.store.save("protocol.json",self.manifest,push=True)

    def transformed(self,groups,spec):
        result = {}
        for split,items in groups.items():
            output = []
            for data,path in items:
                if spec["cross_edges"] and spec["soft_labels"]:
                    output.append((data,path));continue
                data = data.clone()
                if not spec["cross_edges"]:
                    for et in data.edge_types:
                        if et[1]=="corresponds":
                            data[et].edge_index = data[et].edge_index[:,:0]
                            data[et].edge_attr = data[et].edge_attr[:0]
                if not spec["soft_labels"]:
                    for nt in self.c["NODE_TYPES"]:
                        data[nt].y_frac = torch.nn.functional.one_hot(data[nt].y,self.c["NUM_CLASSES"]).float()
                output.append((data,path))
            result[split]=output
        return result

    def graphs(self,spec):
        c = self.c
        if spec["n_segments"]==c["N_SEGMENTS"] and not self.settings.get("pilot"):
            return self.transformed(self.groups,spec)
        folder = self.store.path(f"graphs/slic_{spec['n_segments']}/marker").parent
        archive_name = f"graphs/slic_{spec['n_segments']}.zip"
        marker = folder/"complete.json"
        if not marker.exists():
            archive = self.store.pull(archive_name)
            if archive is not None:
                extract_verified_zip(archive,folder,"cache",sha256_path(archive))
        wanted = {cid for values in self.manifest["split"].values() for cid in values}
        files = {c["get_case_id"](f):f for _,f in c["valid_cases"]}
        if wanted-set(files):
            raise RuntimeError("SLIC rebuild cannot find frozen cohort IDs")
        if marker.exists():
            completed=json.loads(marker.read_text())
            if completed!={"ids":sorted(wanted),"plan":self.key}:
                raise RuntimeError("SLIC cache cohort/identity mismatch")
        changed = False
        try:
            for cid in ([] if marker.exists() else sorted(wanted)):
                self.deadline.check()
                gp,mp = folder/(cid+".graph.pt"),folder/(cid+".meta.pt")
                if gp.exists() and mp.exists():
                    continue
                if not c.get("RAW_SOURCE_AVAILABLE",True):
                    raise RuntimeError("The saved SLIC archive is incomplete. Resume Part 4, which has the audited raw MRI, before evaluation.")
                if shutil.disk_usage(folder).free < 2*1024**3:
                    raise RuntimeError("Less than 2 GiB free disk before SLIC build; preserve the cache and use more storage")
                start=time.perf_counter()
                c["build_and_save"](files[cid],spec["n_segments"],c["COMPACTNESS"],c["SLIC_ITERS"],str(gp)+".tmp",str(mp)+".tmp")
                os.replace(str(gp)+".tmp",gp);os.replace(str(mp)+".tmp",mp)
                write_json_atomic(folder/(cid+".build.json"),dict(seconds=time.perf_counter()-start))
                changed=True
            if not marker.exists():
                write_json_atomic(marker,dict(ids=sorted(wanted),plan=self.key))
                changed=True
        finally:
            if changed:
                archive = self.store.path(archive_name)
                needed=sum(p.stat().st_size for p in folder.iterdir() if p.is_file() and not p.name.endswith(".tmp"))
                if shutil.disk_usage(folder).free < needed+2*1024**3:
                    raise RuntimeError("Insufficient free disk for the SLIC ZIP plus 2 GiB reserve; cached cases are retained")
                with zipfile.ZipFile(archive,"w",zipfile.ZIP_STORED,allowZip64=True) as z:
                    for p in folder.iterdir():
                        if p.is_file() and not p.name.endswith(".tmp"):
                            z.write(p,"cache/"+p.name)
                self.store.push(archive_name)
        rebuilt = {}
        for split,ids in self.manifest["split"].items():
            rebuilt[split]=[(torch.load(folder/(cid+".graph.pt"),map_location="cpu",weights_only=False)[0],
                             str(folder/(cid+".meta.pt"))) for cid in ids]
        return self.transformed(rebuilt,spec)

    def graph_factory(self,spec,seed):
        c = self.c;seed_all(seed)
        if spec["backbone"]=="sage":
            base = HomogeneousSAGE(c["NODE_TYPES"],c["NODE_FEAT_DIM"],c["HIDDEN_DIM"],c["NUM_CLASSES"],c["HGT_LAYERS"],c["DROPOUT"])
        else:
            base = c["QoSHRGN"](in_dim=c["NODE_FEAT_DIM"],hidden_dim=c["HIDDEN_DIM"],heads=c["HEADS"],num_classes=c["NUM_CLASSES"],
                hops=c["HOPS"],hgt_layers=c["HGT_LAYERS"],dropout=c["DROPOUT"],adaptive_hops=True,k_max=c["K_MAX"],hop_temperature=c["HOP_TEMPERATURE"])
            apply_hop_control(base,spec["hop_mode"],spec["fixed_hops"])
        if spec["structural"]:
            flags = dict(c["STRUCTURAL_FLAGS"]) if spec["descriptors"] else {k:False for k in c["STRUCTURAL_FLAGS"]}
            model = c["StructuralQoSHRGN"](base,hidden_dim=c["HIDDEN_DIM"],num_classes=c["NUM_CLASSES"],flags=flags,
                                        refine_hidden=c["STRUCTURAL_HIDDEN"],dropout=c["STRUCTURAL_DROPOUT"])
        else:
            model=base
        return model.to(c["device"])

    def forward_graph(self,model,data,teacher=0.):
        output = model(data,teacher_prob=teacher) if hasattr(model,"base_model") else model(data)
        return (output[1],output[0]) if len(output)==3 else (output[0],None)

    @torch.no_grad()
    def probabilities(self,model,item):
        model.eval()
        logits,_=self.forward_graph(model,item[0].clone().to(self.c["device"]))
        return {nt:values.float().softmax(-1).cpu().numpy() for nt,values in logits.items()}

    def graph_loss(self,logits,data,spec):
        from brats_graph import patient_hierarchy_loss
        c=self.c
        return torch.stack([patient_hierarchy_loss(logits[nt],data[nt],data.num_graphs,
            c['class_weights'].to(c['device']),c['DICE_WEIGHT'],c['soft_cross_entropy'],
            c['region_dice_loss'],c['focal_loss_soft'],c['ET_FOCAL_WEIGHT'] if spec['focal'] else 0.,
            c['FOCAL_GAMMA']) for nt in c['NODE_TYPES']]).mean()

    def fit_graph(self,spec,seed,groups):
        c=self.c
        # Voxel/boundary-only changes intentionally reuse identical graph weights.
        graph_spec={k:v for k,v in spec.items() if k not in ("name","description","voxel","boundary","voxel_kind")}
        gid=identity(graph_spec)[:16]
        name=f"models/{gid}/seed_{seed}/graph.pt"
        model=self.graph_factory(spec,seed)
        rec=None
        if spec["masked_reconstruction"]:
            rec=c["HeteroMaskedReconstructor"](hidden_dim=c["HIDDEN_DIM"],appearance_dim=c["APPEARANCE_DIM"],
                node_types=c["NODE_TYPES"],decoder_hidden=c["REC_DECODER_HIDDEN"],dropout=c["DROPOUT"]).to(c["device"])
        training_model=torch.nn.ModuleDict({"graph":model, **({"reconstructor":rec} if rec is not None else {})})
        signature=identity(dict(plan=self.key,graph=graph_spec,seed=seed))
        full=next(row for row in self.plan if row["name"]=="full")
        full_graph={k:v for k,v in full.items() if k not in ("name","description","voxel","boundary","voxel_kind")}
        original=c.get("stage1_checkpoint")
        if seed==c["SEED"] and graph_spec==full_graph and original is not None and self.store.read(name) is None:
            model.load_state_dict(original["model_state_dict"],strict=True)
            if rec is not None: rec.load_state_dict(original["rec_state_dict"],strict=True)
            save_imported_selection(self.store,name,signature,training_model,original.get("epochs_run",0),
                original["best_epoch"],original["best_val_dice"],original.get("history",[]),
                dict(precision="fp16" if c["device"].type=="cuda" else "fp32",microbatch=1,inference_batch=1),
                c["LR"],c["WEIGHT_DECAY"],c["EPOCHS"],"unchanged Part 1 selected graph weights", termination={k:original.get('graph_runtime',{}).get(k) for k in ('stopping_reason','phase_seconds_used','phase_seconds_limit')})
        from brats_graph import GraphBatchEngine, patient_hop_penalty
        def objective_forward(current,batch,teacher):
            output=current(batch,teacher_prob=teacher) if hasattr(current,'base_model') else current(batch)
            return (output[1],output[0],output[-1]) if len(output)==3 else (output[0],None,output[-1])
        def supervised(logits,batch,aux):
            loss=self.graph_loss(logits,batch,spec)
            if aux is not None:
                loss=loss+c['STRUCTURAL_AUX_WEIGHT']*self.graph_loss(aux,batch,spec)
            if spec['backbone']=='hgt' and spec['hop_mode']=='adaptive' and spec['hop_penalty']:
                loss=loss+c['HOP_REG_WEIGHT']*patient_hop_penalty(getattr(model,'base_model',model),batch)
            return loss
        engine=GraphBatchEngine(model,rec,objective_forward,supervised,c['mask_hetero_graph'],c['reconstruction_loss'],
                                separate_clean=c['REC_SEPARATE_CLEAN_PASS'])
        def graph_calibrate(saved, deadline):
            return engine.calibrate([d for d,_ in groups['train']],effective_batch=c['BATCH_SIZE'],
                max_microbatch=c.get('GRAPH_MAX_MICROBATCH',16),memory_fraction=c.get('GRAPH_MEMORY_FRACTION',.75),
                lam=c['LAMBDA_REC'] if rec is not None else 0.,precision=saved,deadline=deadline)
        def train_epoch(epoch,opt,scaler,runtime,deadline):
            graphs=[d for d,_ in groups['train']]
            random.Random(seed+epoch).shuffle(graphs)
            losses=[]
            teacher=c['structural_teacher_prob'](epoch,c['EPOCHS']) if spec['structural'] else 0.
            lam=c['rec_lambda'](epoch) if rec is not None else 0.
            for group in patient_groups(graphs,c['BATCH_SIZE']):
                deadline.check()
                row=engine.train_group(group,opt,scaler,runtime,teacher=teacher,lam=lam)
                losses.extend([row['total']]*len(group))
            return dict(train_loss=float(np.mean(losses)),graph_microbatch=runtime['microbatch'],
                        graph_oom_retries=runtime.get('oom_retries',0))
        def validate(deadline):
            scores=[]
            for item in groups["val"]:
                deadline.check();meta=c["load_meta"](item[1])
                pred=c["postprocess_prediction_auto"](c["project_node_probs"](self.probabilities(model,item),meta))
                scores.append(mean_region_dice(pred,meta["seg"]))
            return np.mean(scores)
        precision="fp16" if c["device"].type=="cuda" else "fp32"
        state=train_phase(training_model,self.store,name,signature,epochs=c["EPOCHS"],patience=c["EARLY_STOP_PATIENCE"],lr=c["LR"],
            decay=c["WEIGHT_DECAY"],initial_precision=precision,calibrate=graph_calibrate,
            train_epoch=train_epoch,validate=validate,deadline=self.deadline,push_every=c["CKPT_PUSH_EVERY_EPOCHS"])
        state["reconstruction_parameters"]=sum(p.numel() for p in rec.parameters()) if rec is not None else 0
        return model,state,name

    def prior_loader(self,model):
        if model is None:
            return lambda item:None
        digest=model_digest(model)
        folder=self.store.path("priors/"+digest+"/marker").parent
        def prior(item,cached_only=False):
            path=folder/(case_id(item)+".npz")
            if not path.exists() and not cached_only and hasattr(self.store,"catalog"):
                self.store.pull("priors/"+digest+"/"+case_id(item)+".npz")
            if not path.exists():
                if cached_only:
                    raise RuntimeError("Missing frozen prior before CPU preparation")
                probs=self.probabilities(model,item)
                with open(str(path)+".tmp","wb") as f:
                    np.savez_compressed(f,**probs)
                os.replace(str(path)+".tmp",path)
            with np.load(path,allow_pickle=False) as z:
                return {k:z[k].copy() for k in z.files}
        return prior

    def fit_voxel(self,spec,seed,groups,graph_model):
        c=self.c;seed_all(seed)
        if graph_model is not None:
            graph_model.eval();graph_model.zero_grad(set_to_none=True)
            for parameter in graph_model.parameters():
                parameter.requires_grad_(False)
        from brats_practical import SegResNetBaseline
        head=(SegResNetBaseline(len(c["MODALITIES"]),c["NUM_CLASSES"]) if spec["voxel_kind"]=="segresnet" else IndependentUNet(len(c["MODALITIES"]),c["NUM_CLASSES"],8) if spec["voxel_kind"]=="unet" else
              c["VoxelRefinementHead"](len(c["MODALITIES"])+(c["NUM_CLASSES"] if graph_model is not None else 0),c["NUM_CLASSES"],c["VOXEL_BASE_CHANNELS"])).to(c["device"])
        prior=self.prior_loader(graph_model)
        name=f"runs/{spec['name']}/seed_{seed}/voxel.pt"
        signature=identity(dict(plan=self.key,spec=spec,seed=seed,graph=model_digest(graph_model) if graph_model is not None else None))
        original=c.get("MAIN_STAGE2_STATE")
        if spec["name"]=="full" and seed==c["SEED"] and original is not None and self.store.read(name) is None:
            head.load_state_dict(original["best_vox_state"],strict=True)
            save_imported_selection(self.store,name,signature,head,original["epoch"],original["best_vox_epoch"],
                original["best_vox_dice"],original["vox_history"],original["gpu_runtime"],
                c["VOXEL_LR"],c["WEIGHT_DECAY"],c["VOXEL_EPOCHS"],"completed Part 2 selected voxel weights", termination={k:original.get(k) for k in ('stopping_reason','phase_seconds_used','phase_seconds_limit')})
        def prepare(value):
            item,local_seed=value
            probs=prior(item,cached_only=True) if graph_model is not None else None
            x,y=make_training_patch(c["load_meta"](item[1]),c["VOXEL_MAX_SIZE"],probs,rng=np.random.RandomState(int(local_seed)))
            tx,ty=torch.from_numpy(x).float(),torch.from_numpy(y).long()
            return (tx.pin_memory(),ty.pin_memory()) if c["device"].type=="cuda" else (tx,ty)
        def criterion(output,target):
            logits,boundary=output
            return c["voxel_loss"](logits,target,c["class_weights"].to(c["device"]),dice_weight=c["VOXEL_DICE_WEIGHT"],ce_weight=c["VOXEL_CE_WEIGHT"],
                focal_weight=c["VOXEL_FOCAL_WEIGHT"] if spec["focal"] else 0.,focal_gamma=c["VOXEL_FOCAL_GAMMA"],boundary_logits=boundary,
                boundary_weight=c["VOXEL_BOUNDARY_WEIGHT"] if spec["boundary"] else 0.)
        def calibrate(saved, deadline):
            for item in groups["train"]+groups["val"]:
                deadline.check();prior(item)
            x,y=prepare((groups["train"][0],seed))
            head.brats_gpu_runtime=configure_gpu(head,x.numpy(),y.numpy(),criterion,roi=c["VOXEL_MAX_SIZE"],precision=c["GPU_PRECISION"],resume_precision=saved,
                max_microbatch=c["VOXEL_MAX_MICROBATCH"],max_inference_batch=c["INFERENCE_MAX_BATCH"],memory_fraction=c["GPU_MEMORY_FRACTION"],autotune=c["GPU_AUTOTUNE"])
            return head.brats_gpu_runtime
        def train_epoch(epoch,opt,scaler,runtime,deadline):
            items=sorted(groups["train"],key=case_id);random.shuffle(items)
            seeds=np.random.randint(0,2**31-1,len(items));losses=[];dices=[]
            prepared=bounded_prefetch(zip(items,seeds),prepare,workers=c["VOXEL_PREFETCH_WORKERS"] if c["device"].type=="cuda" else 0,
                                      depth=max(2,c["VOXEL_EFFECTIVE_BATCH_SIZE"]))
            try:
                for group in patient_groups(prepared,c["VOXEL_EFFECTIVE_BATCH_SIZE"]):
                    deadline.check();loss,batch_dices=train_patient_group(head,group,opt,scaler,criterion,runtime)
                    losses.extend([loss]*len(group));dices.extend(batch_dices)
            finally:
                prepared.close()
            return dict(train_loss=float(np.mean(losses)),train_patch_dice=float(np.mean(dices)))
        def validate(deadline):
            from brats_gpu import amp_context
            scores=[];losses=[]
            for item in groups["val"]:
                deadline.check();meta=c["load_meta"](item[1])
                probs=prior(item)
                pred=sliding_window_predict(head,meta,probs,roi=c["VOXEL_MAX_SIZE"],overlap=c["INFERENCE_OVERLAP"])
                scores.append(mean_region_dice(c["postprocess_prediction_auto"](pred),meta["seg"]))
                seed_patch=int(hashlib.sha256(case_id(item).encode()).hexdigest()[:8],16)
                x,y=make_training_patch(meta,c['VOXEL_MAX_SIZE'],probs,rng=np.random.RandomState(seed_patch),tumour_probability=0.)
                with torch.no_grad(),amp_context(head.brats_gpu_runtime,c['device']):
                    losses.append(float(criterion(head(torch.from_numpy(x).unsqueeze(0).float().to(c['device'])),
                        torch.from_numpy(y).unsqueeze(0).long().to(c['device']))))
            return dict(val_dice=float(np.mean(scores)),val_loss=float(np.mean(losses)))
        state=train_phase(head,self.store,name,signature,epochs=c["VOXEL_EPOCHS"],patience=c["EARLY_STOP_PATIENCE"],lr=c["VOXEL_LR"],decay=c["WEIGHT_DECAY"],
            initial_precision="fp32",calibrate=calibrate,train_epoch=train_epoch,validate=validate,deadline=self.deadline,push_every=c["CKPT_PUSH_EVERY_EPOCHS"])
        return head,state,name

    def completion_name(self,spec,seed):
        return f"runs/{spec['name']}/seed_{seed}/trained.json"

    def completion(self,spec,seed):
        record=self.store.read(self.completion_name(spec,seed))
        if record is not None and (record["plan"]!=self.key or record["spec"]!=spec or record["seed"]!=seed):
            raise RuntimeError("Research completion identity mismatch")
        return record

    def train_run(self,spec,seed,groups):
        if self.completion(spec,seed) is not None:
            print(f"[research] already trained: {spec['name']} seed {seed}")
            return
        graph=head=None;checkpoints={};protocols={}
        try:
            if spec["backbone"]!="none":
                graph,state,name=self.fit_graph(spec,seed,groups)
                checkpoints["graph"]=dict(name=name,sha256=sha256_path(self.store.path(name)))
                protocols["graph"]=dict(stopping_reason=state.get("stopping_reason","imported_selection"), phase_seconds_used=state.get("phase_seconds_used"), best_epoch=state["best_epoch"],validation_dice=state["best_score"],history=state["history"],
                    reconstruction_parameters=state.get("reconstruction_parameters",0),
                    parameters=sum(p.numel() for p in graph.parameters()),trained_parameters=sum(p.numel() for p in graph.parameters() if p.requires_grad),
                    best_weights_sha256=model_digest(graph))
            if spec["voxel"]:
                head,state,name=self.fit_voxel(spec,seed,groups,graph)
                checkpoints["voxel"]=dict(name=name,sha256=sha256_path(self.store.path(name)))
                protocols["voxel"]=dict(stopping_reason=state.get("stopping_reason","imported_selection"), phase_seconds_used=state.get("phase_seconds_used"), best_epoch=state["best_epoch"],validation_dice=state["best_score"],history=state["history"],runtime=state["runtime"],
                    parameters=sum(p.numel() for p in head.parameters()),best_weights_sha256=model_digest(head))
            build_rows=[json.loads(p.read_text()) for p in self.store.path(f"graphs/slic_{spec['n_segments']}/marker").parent.glob('*.build.json')]
            graph_stats=[]
            for data,_ in groups["train"]:
                graph_stats.append(dict(nodes=sum(data[nt].num_nodes for nt in self.c["NODE_TYPES"]),
                    edges=sum(data[et].edge_index.shape[1] for et in data.edge_types)))
            self.store.save(self.completion_name(spec,seed),dict(plan=self.key,spec=spec,seed=seed,checkpoints=checkpoints,protocols=protocols,
                graph_size_mean={key:float(np.mean([v[key] for v in graph_stats])) for key in ("nodes","edges")},
                graph_construction=dict(measured_cases=len(build_rows), seconds_total=sum(r["seconds"] for r in build_rows), scope="fresh pilot graph builds" if self.settings.get("pilot") else "cached main graphs; consult Part 1 build receipt"), selection="validation only",test_evaluated=False),push=True)
        finally:
            del graph,head
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def require_complete_plan(self):
        missing=[f"{row['name']}/seed_{seed}" for row in self.plan for seed in self.seeds if self.completion(row,seed) is None]
        if missing:
            raise RuntimeError("Finish the frozen training plan before evaluating held-out test data. Missing: "+", ".join(missing))
        if self.manifest["debug"]:
            raise RuntimeError("Debug subsets cannot produce the final test report; use a full-cohort plan")

    def evaluation_complete(self,spec,seed,split):
        value=self.store.read(f"evaluation/{split}/{spec['name']}/seed_{seed}.json")
        if not value or not value.get("complete"):return False
        completed=self.completion(spec,seed)
        ids=[row["case_id"] for row in value["patients"]]
        if value["plan"]!=self.key or completed is None or value["checkpoints"]!=completed["checkpoints"] or len(ids)!=len(set(ids)) or set(ids)!=set(self.manifest["split"][split]):
            raise RuntimeError("Completed evaluation identity mismatch")
        return True

    def evaluate_run(self,spec,seed,groups,split):
        c=self.c;completed=self.completion(spec,seed)
        if completed is None:
            raise RuntimeError("Train this row before evaluation")
        for checkpoint in completed["checkpoints"].values():
            local=self.store.pull(checkpoint["name"])
            if local is None or sha256_path(local)!=checkpoint["sha256"]:
                raise RuntimeError("Completed-run checkpoint changed or is missing")
        prior_result=self.store.read(f"evaluation/{split}/{spec['name']}/seed_{seed}.json")
        if prior_result and prior_result.get("complete"):
            expected_ids=set(map(case_id,groups[split]))
            ids=[row["case_id"] for row in prior_result["patients"]]
            if prior_result["checkpoints"]!=completed["checkpoints"] or prior_result["plan"]!=self.key or len(ids)!=len(set(ids)) or set(ids)!=expected_ids:
                raise RuntimeError("Completed evaluation does not match the frozen cohort/checkpoints")
            return prior_result["patients"]
        graph=head=None
        if spec["backbone"]!="none":
            graph,_,_=self.fit_graph(spec,seed,groups)
        if spec["voxel"]:
            head,_,_=self.fit_voxel(spec,seed,groups,graph)
        name=f"evaluation/{split}/{spec['name']}/seed_{seed}.json"
        result=self.store.read(name) or dict(plan=self.key,checkpoints=completed["checkpoints"],patients=[])
        if result["plan"]!=self.key or result["checkpoints"]!=completed["checkpoints"]:
            raise RuntimeError("Evaluation resume identity mismatch")
        done={r["case_id"] for r in result["patients"]}
        wanted=set(map(case_id,groups[split]))
        if len(done)!=len(result["patients"]) or done-wanted:
            raise RuntimeError("Duplicate/unexpected evaluation patient")
        try:
            for item in groups[split]:
                if case_id(item) in done:
                    continue
                self.deadline.check()
                if c["device"].type=="cuda":
                    torch.cuda.synchronize(c["device"]);torch.cuda.reset_peak_memory_stats(c["device"])
                start=time.perf_counter();meta=c["load_meta"](item[1])
                probs=self.probabilities(graph,item) if graph is not None else None
                raw=(sliding_window_predict(head,meta,probs,roi=c["VOXEL_MAX_SIZE"],overlap=c["INFERENCE_OVERLAP"])
                     if head is not None else c["project_node_probs"](probs,meta))
                pred=c["postprocess_prediction_auto"](raw.copy())
                if c["device"].type=="cuda":
                    torch.cuda.synchronize(c["device"])
                seconds=time.perf_counter()-start
                wt=meta["seg"]>0
                from scipy.ndimage import binary_erosion
                characteristics=dict(WT_voxels=int(wt.sum()),ET_voxels=int(np.count_nonzero(meta["seg"]==3)),
                    WT_surface_fraction=float(np.count_nonzero(wt & ~binary_erosion(wt))/max(1,int(wt.sum()))))
                result["patients"].append(dict(experiment=spec["name"],seed=seed,case_id=case_id(item),
                    target_summary=characteristics,
                    metrics=segmentation_metrics(pred,meta["seg"]),raw_metrics=segmentation_metrics(raw,meta["seg"]),
                    cached_graph_inference_seconds=seconds,peak_cuda_bytes=torch.cuda.max_memory_allocated(c["device"]) if c["device"].type=="cuda" else 0))
                result["complete"]=len(result["patients"])==len(wanted)
                self.store.save(name,result,push=result["complete"] or len(result["patients"])%5==0)
        except BudgetPause:
            if self.store.path(name).exists():
                self.store.push(name)
            raise
        finally:
            del graph,head
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return result["patients"]

    def export_report(self,split):
        records=[];summaries=[];raw_summaries=[];table=[]
        for spec in self.plan:
            for seed in self.seeds:
                result=self.store.read(f"evaluation/{split}/{spec['name']}/seed_{seed}.json")
                if result is None or not result.get("complete"):
                    return None
                rows=result["patients"];records.extend(rows)
                stats=summarize_metric_rows([r["metrics"] for r in rows])
                summaries.append(dict(experiment=spec["name"],seed=seed,**stats))
                raw_summaries.append(dict(experiment=spec["name"],seed=seed,**summarize_metric_rows([r["raw_metrics"] for r in rows])))
                trained=self.completion(spec,seed)
                protocols=trained["protocols"]
                entry=dict(graph_stopping_reason=protocols.get("graph",{}).get("stopping_reason"), voxel_stopping_reason=protocols.get("voxel",{}).get("stopping_reason"), experiment=spec["name"],seed=seed,n_segments_per_partition=spec["n_segments"],n_patients=len(rows),
                    graph_parameters=protocols.get("graph",{}).get("parameters",0),
                    voxel_parameters=protocols.get("voxel",{}).get("parameters",0),
                    graph_best_epoch=protocols.get("graph",{}).get("best_epoch"),
                    voxel_best_epoch=protocols.get("voxel",{}).get("best_epoch"),
                    graph_construction_seconds=trained.get('graph_construction',{}).get('seconds_total'),
                    graph_nodes_mean=trained.get('graph_size_mean',{}).get('nodes'),
                    graph_edges_mean=trained.get('graph_size_mean',{}).get('edges'),
                    graph_training_seconds=sum(r.get('seconds',r.get('epoch_seconds',0)) for r in protocols.get('graph',{}).get('history',[])),
                    cached_graph_inference_seconds_mean=float(np.mean([r["cached_graph_inference_seconds"] for r in rows])),
                    inference_peak_cuda_bytes=max(r["peak_cuda_bytes"] for r in rows),
                    recorded_training_seconds=sum(h["seconds"] for p in protocols.values() for h in p["history"]),
                    **{f"mean_{key}":value for key,value in trained["graph_size_mean"].items()})
                for metric in rows[0]["metrics"]:
                    entry[metric+"_mean"],entry[metric+"_patient_sd"]=stats[metric],stats["std"][metric]
                table.append(entry)
        report=dict(plan=self.key,split=split,debug=self.manifest["debug"],runs=summaries,
            raw_runs=raw_summaries,
            comparisons=paired_summary(records),
            graph_only_controls=(paired_summary([r for r in records if r['experiment'] in ('no_voxel','graphsage_backbone','fixed_shared_hops')],reference='no_voxel')
                if {'no_voxel','graphsage_backbone','fixed_shared_hops'} <= {r['experiment'] for r in records} else None),runtime_scope="metadata read + graph/CNN inference + postprocess; excludes graph building and metrics")
        self.store.save(f"reports/{split}_report.json",report,push=True)
        csv_name=f"reports/{split}_patients.csv"
        fields=["experiment","seed","case_id","cached_graph_inference_seconds","peak_cuda_bytes"]+list(records[0]["metrics"])
        with open(self.store.path(csv_name),"w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for row in records:
                writer.writerow({**{k:row[k] for k in fields[:5]},**row["metrics"]})
        self.store.push(csv_name)
        table_name=f"reports/{split}_summary.csv"
        with open(self.store.path(table_name),"w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        self.store.push(table_name)
        self.plot_report(split,table)
        return report

    def plot_report(self,split,table):
        import matplotlib.pyplot as plt
        selected=[row for row in self.plan if row["name"]=="full" or row["name"].startswith("slic_")]
        if len(selected)>1:
            selected.sort(key=lambda row:row["n_segments"])
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            for region in ("WT","TC","ET"):
                values=[[r[f"Dice_{region}_mean"] for r in table if r["experiment"]==row["name"]] for row in selected]
                axes[0].errorbar([r["n_segments"] for r in selected],[np.mean(v) for v in values],
                    yerr=[np.std(v,ddof=1) if len(v)>1 else 0 for v in values],marker='o',label=region)
            axes[0].set(xlabel='Requested SLIC segments per partition',ylabel='Patient mean Dice (seed mean +/- SD)',title=split+' SLIC comparison')
            axes[0].legend()
            times=[[r['cached_graph_inference_seconds_mean'] for r in table if r['experiment']==row['name']] for row in selected]
            axes[1].plot([r['n_segments'] for r in selected],[np.mean(v) for v in times],marker='o')
            axes[1].set(xlabel='Requested SLIC segments per partition',ylabel='Cached-graph inference seconds/patient',title='Excludes SLIC construction')
            fig.tight_layout();name=f'reports/{split}_slic_performance_cost.png'
            fig.savefig(self.store.path(name),dpi=180);plt.close(fig);self.store.push(name)
        for spec in self.plan:
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            for seed in self.seeds:
                record=self.completion(spec,seed)
                for stage,protocol in record['protocols'].items():
                    history=protocol['history'];label=f'{stage} seed {seed}'
                    axes[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label=label)
                    axes[1].plot([r['epoch'] for r in history],[r['val_dice'] for r in history],label=label)
            axes[0].set(xlabel='Epoch',ylabel='Training objective',title=spec['name'])
            axes[1].set(xlabel='Epoch',ylabel='Full-volume validation Dice',title='Model selection uses validation only')
            axes[0].legend(fontsize=7);axes[1].legend(fontsize=7);fig.tight_layout()
            name=f"reports/curves/{spec['name']}.png";fig.savefig(self.store.path(name),dpi=180);plt.close(fig);self.store.push(name)

    def qualitative_examples(self):
        """Descriptive post-test selection, disclosed; never model selection."""
        spec=next(row for row in self.plan if row["name"]=="full")
        seed=self.seeds[0]  # Predeclared first seed, never chosen by test score.
        result=self.store.read(f"evaluation/test/full/seed_{seed}.json")
        if result is None or not result.get("complete"):
            return
        patients=result["patients"]
        score=lambda r:np.mean([r["metrics"]["Dice_"+region] for region in ("WT","TC","ET")])
        selection=[]
        def choose(rows,reason):
            for row in rows:
                if row["case_id"] not in {v["case_id"] for v in selection}:
                    selection.append(dict(case_id=row["case_id"],reason=reason));return
        choose(sorted(patients,key=score),"Lowest observed mean Dice")
        choose(sorted(patients,key=score,reverse=True),"Highest observed mean Dice")
        choose(sorted([r for r in patients if r["target_summary"]["ET_voxels"]>0],key=lambda r:r["target_summary"]["ET_voxels"]),"Small enhancing-tumour region")
        choose(sorted(patients,key=lambda r:r["target_summary"]["WT_voxels"],reverse=True),"Large whole-tumour region")
        choose(sorted(patients,key=lambda r:r["target_summary"]["WT_surface_fraction"],reverse=True),"High tumour surface-to-volume proxy")
        remaining=sorted(patients,key=lambda r:r["case_id"]);random.Random(self.c["SEED"]+990).shuffle(remaining)
        while len(selection)<min(10,len(patients)):
            choose(remaining,"Deterministic random example")
        self.store.save("reports/qualitative/selection.json",dict(seed=seed,selection=selection,
            policy="Descriptive post-test examples selected by disclosed morphology/score rules, then random fill; not a representative performance sample"),push=True)
        if all(self.store.pull(f"reports/qualitative/{row['case_id']}.png") is not None for row in selection):
            return
        groups=self.graphs(spec);lookup={case_id(item):item for item in groups["test"]}
        graph,_,_=self.fit_graph(spec,seed,groups);head,_,_=self.fit_voxel(spec,seed,groups,graph)
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap,BoundaryNorm
        cmap=ListedColormap(["black","#e15759","#59a14f","#f1ce63"])
        norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
        try:
            for chosen in selection:
                name=f"reports/qualitative/{chosen['case_id']}.png"
                if self.store.path(name).exists():continue
                self.deadline.check();item=lookup[chosen["case_id"]];meta=self.c["load_meta"](item[1])
                probs=self.probabilities(graph,item)
                gp=self.c["postprocess_prediction_auto"](self.c["project_node_probs"](probs,meta))
                hp=self.c["postprocess_prediction_auto"](sliding_window_predict(head,meta,probs,roi=self.c["VOXEL_MAX_SIZE"],overlap=self.c["INFERENCE_OVERLAP"]))
                slices=tuple(slice(int(lo),int(hi)) for lo,hi in zip(meta["lo"],meta["hi"]))
                target=meta["seg"][slices];gp=gp[slices];hp=hp[slices]
                z=int((target>0).sum((0,1)).argmax()) if np.any(target) else target.shape[2]//2
                fig,axes=plt.subplots(1,5,figsize=(16,4))
                views=[(meta["vis"]["t1ce"][:,:,z],"T1ce"),(target[:,:,z],"Reference"),(gp[:,:,z],"Graph only"),
                       (hp[:,:,z],"Graph + voxel"),((hp[:,:,z]!=target[:,:,z]).astype(np.uint8),"Voxel prediction error")]
                for i,(array,title) in enumerate(views):
                    axes[i].imshow(array.T,origin="lower",**({"cmap":"gray"} if i==0 else ({"cmap":"Reds","vmin":0,"vmax":1} if i==4 else {"cmap":cmap,"norm":norm})))
                    axes[i].set_title(title);axes[i].axis("off")
                fig.suptitle(f"{chosen['case_id']} | seed {seed} | {chosen['reason']}\nSlice chosen from reference for display only; inference covered the full brain",fontsize=10)
                fig.tight_layout(rect=(0,0,1,.88));fig.savefig(self.store.path(name),dpi=180);plt.close(fig);self.store.push(name)
        finally:
            del graph,head
            if torch.cuda.is_available():torch.cuda.empty_cache()

    def run(self):
        mode=self.settings["mode"]
        if mode not in ("train","validation","test"):
            raise ValueError("Research mode must be train, validation or test")
        jobs=self.settings["jobs"] or [r["name"] for r in self.plan]
        active_seeds=self.settings["active_seeds"] or self.seeds
        if set(jobs)-{r["name"] for r in self.plan} or set(active_seeds)-set(self.seeds):
            raise ValueError("Selected jobs/seeds must be members of the frozen plan")
        if mode=="test":
            self.require_complete_plan()
        print(f"Research {self.key}: {len(self.plan)} rows x {len(self.seeds)} seeds; mode={mode}")
        status=dict(plan=self.key,mode=mode,paused=False,finished_jobs=[])
        try:
            for spec in self.plan:
                if spec["name"] not in jobs:
                    continue
                self.deadline.check()
                if mode=="train" and all(self.completion(spec,seed) is not None for seed in active_seeds):
                    status["finished_jobs"].extend(dict(experiment=spec["name"],seed=seed) for seed in active_seeds)
                    continue
                if mode!="train" and all(self.evaluation_complete(spec,seed,"test" if mode=="test" else "val") for seed in active_seeds):
                    status["finished_jobs"].extend(dict(experiment=spec["name"],seed=seed) for seed in active_seeds)
                    continue
                groups=self.graphs(spec)
                for seed in active_seeds:
                    self.deadline.check()
                    if mode=="train":
                        self.train_run(spec,seed,groups)
                    else:
                        self.evaluate_run(spec,seed,groups,"val" if mode=="validation" else "test")
                    status["finished_jobs"].append(dict(experiment=spec["name"],seed=seed))
                del groups
            if mode!="train":
                status["report_ready"]=self.export_report("val" if mode=="validation" else "test") is not None
                if mode=="test" and status["report_ready"]:
                    self.qualitative_examples()
        except BudgetPause as exc:
            status.update(paused=True,message=str(exc))
            print(str(exc))
        status["scheduled_training_complete"]=all(self.completion(spec,seed) is not None
            for spec in self.plan if spec["name"] in jobs for seed in active_seeds)
        status["all_training_complete"]=all(self.completion(spec,seed) is not None for spec in self.plan for seed in self.seeds)
        self.store.save("session_status.json",status,push=True)
        return status


def run_research(context,settings):
    runner=ResearchRunner(context,settings)
    try:
        return runner.run()
    finally:
        if hasattr(runner.store,"flush"):
            runner.store.collect(runner.store.root/"priors","priors")
            runner.store.flush()
