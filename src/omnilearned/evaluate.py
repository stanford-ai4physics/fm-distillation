import torch
from omnilearned.network import PET2, DeepSets, MLPStudent, ACT_LAYERS
from omnilearned.dataloader import load_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omnilearned.utils import (
    is_master_node,
    ddp_setup,
    get_checkpoint_name,
    restore_checkpoint,
    pad_array,
    get_model_parameters,
    get_deepsets_parameters,
    get_mlp_parameters,
)
from omnilearned.diffusion import generate
import os
import time
import numpy as np
import h5py
from tqdm.auto import tqdm


def eval_model(
    model,
    test_loader,
    dataset,
    mode,
    use_event_loss,
    device="cpu",
    outdir="",
    save_tag="pretrain",
    rank=0,
    dataset_type="test",
    num_chunks=1,
    chunk_idx=0,
):
    chunk_suffix = f"_chunk{chunk_idx}of{num_chunks}" if num_chunks > 1 else ""
    cls_name = f"outputs_{save_tag}_{dataset}_{dataset_type}{chunk_suffix}_rank{rank}.npz"
    gen_name = f"generated_{save_tag}_{dataset}_{dataset_type}{chunk_suffix}_rank{rank}.h5"
    # For classifier mode we also collect sample_keys + raw logits so the
    # same outputs_*.npz file can serve as the teacher labels for KD.
    return_sample_keys = mode == "classifier"
    result = test_step(
        model, test_loader, mode, device, return_sample_keys=return_sample_keys
    )
    if return_sample_keys:
        prediction, cond, labels, sample_keys = result
    else:
        prediction, cond, labels = result

    if mode in ["classifier", "regression", "segmentation"]:
        if use_event_loss:
            np.savez(
                os.path.join(
                    outdir, cls_name
                ),
                prediction=prediction[:, :200].softmax(-1).cpu().numpy(),
                event_prediction=prediction[:, 200:].softmax(-1).cpu().numpy(),
                logits=prediction.cpu().numpy().astype(np.float16),
                sample_keys=sample_keys.cpu().numpy(),
                pid=labels.cpu().numpy(),
                cond=cond.cpu().numpy() if cond is not None else [],
            )
        else:
            if mode == "classifier":
                logits_np = prediction.cpu().numpy().astype(np.float16)
                prediction_np = prediction.softmax(-1).cpu().numpy()
                np.savez(
                    os.path.join(
                        outdir,
                        cls_name,
                    ),
                    prediction=prediction_np,
                    logits=logits_np,
                    sample_keys=sample_keys.cpu().numpy(),
                    pid=labels.cpu().numpy(),
                    cond=cond.cpu().numpy() if cond is not None else [],
                )
            else:
                np.savez(
                    os.path.join(
                        outdir,
                        cls_name,
                    ),
                    prediction=prediction.cpu().numpy(),
                    pid=labels.cpu().numpy(),
                    cond=cond.cpu().numpy() if cond is not None else [],
                )
    else:
        with h5py.File(
            os.path.join(
                outdir, gen_name
            ),
            "w",
        ) as fh5:
            fh5.create_dataset("data", data=prediction.cpu().numpy())
            fh5.create_dataset("global", data=cond.cpu().numpy())
            fh5.create_dataset("pid", data=labels.cpu().numpy() + 1)


def test_step(
    model,
    dataloader,
    mode,
    device,
    return_sample_keys=False,
):
    model.eval()

    # Inference precision is controlled by EVAL_AMP: "fp32" (default, no
    # autocast), "bf16", or "fp16". Logits are saved as float16 regardless, so
    # bf16 autocast gives a large speedup on tensor cores at negligible cost --
    # but the default stays fp32 until the bf16 path is validated A/B, so a
    # run can't silently produce unvalidated bf16 logits. Set EVAL_AMP=bf16 to
    # opt in per-dataset.
    amp_choice = os.environ.get("EVAL_AMP", "fp32").lower()
    amp_dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
    on_cuda = device != "cpu"
    use_amp = on_cuda and amp_choice in amp_dtypes
    amp_dtype = amp_dtypes.get(amp_choice)
    if is_master_node():
        print(f"[eval] forward precision: {'autocast ' + amp_choice if use_amp else 'fp32'}")
    fwd_time = 0.0

    preds = []
    labels = []
    conds = []
    sample_keys = [] if return_sample_keys else None

    for ib, batch in enumerate(
        tqdm(dataloader, desc="Iterating", total=len(dataloader))
        if is_master_node()
        else dataloader
    ):
        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        npart = X.shape[1]
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }

        with torch.no_grad():
            if mode in ["classifier", "regression", "segmentation"]:
                if on_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                if use_amp:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        outputs = model(X, y, **model_kwargs)
                else:
                    outputs = model(X, y, **model_kwargs)
                if on_cuda:
                    torch.cuda.synchronize()
                fwd_time += time.perf_counter() - t0
                output_name = (
                    "y_pred" if mode in ["classifier", "regression"] else "z_pred"
                )
                # Cast back to fp32: autocast outputs bf16/fp16, but downstream
                # softmax + numpy save need a numpy-representable float dtype.
                preds.append(outputs[output_name].float())

            elif mode == "generator":
                assert "cond" in model_kwargs, (
                    "ERROR, conditioning variables not passed to model"
                )
                preds.append(generate(model, y, X.shape, **model_kwargs))
        if mode == "segmentation":
            labels.append(batch["data_pid"].to(device))
        else:
            labels.append(y)

        conds.append(batch["cond"])
        if return_sample_keys:
            sample_keys.append(batch["sample_key"])
        if mode == "generator":
            if batch["pid"] is not None:
                preds[-1] = torch.cat(
                    [preds[-1], model_kwargs["pid"].unsqueeze(-1).float()], -1
                )
            if batch["add_info"] is not None:
                preds[-1] = torch.cat([preds[-1], model_kwargs["add_info"]], -1)

    if is_master_node() and mode in ["classifier", "regression", "segmentation"]:
        print(f"[eval] total forward time: {fwd_time:.2f}s over {len(preds)} batches "
              f"({1000 * fwd_time / max(len(preds), 1):.1f} ms/batch)")

    if mode == "generator":
        preds = pad_array(preds, npart)
    else:
        preds = torch.cat(preds).to(device)

    result = (
        preds,
        torch.cat(conds).to(device) if conds[0] is not None else None,
        torch.cat(labels).to(device),
    )
    if return_sample_keys:
        result = result + (torch.cat(sample_keys),)
    return result


def run(
    indir: str = "",
    outdir: str = "",
    save_tag: str = "",
    dataset: str = "top",
    path: str = "/pscratch/sd/v/vmikuni/datasets",
    num_feat: int = 4,
    model_size: str = "small",
    interaction: bool = False,
    local_interaction: bool = False,
    num_coord: int = 2,
    K: int = 10,
    interaction_type: str = "lhc",
    conditional: bool = False,
    num_cond: int = 3,
    use_pid: bool = False,
    pid_idx: int = -1,
    use_add: bool = False,
    num_add: int = 4,
    use_event_loss: bool = False,
    num_classes: int = 2,
    num_gen_classes: int = 1,
    mode: str = "classifier",
    batch: int = 64,
    num_workers: int = 16,
    clip_inputs: bool = False,
    dataset_type: str = "test",
    num_chunks: int = 1,
    chunk_idx: int = 0,
    arch: str = "pet2",
    energy_weighted_pool: bool = False,
    num_interaction_layers: int = 0,
    interaction_k: int = 0,
    act_layer: str = "gelu",
    deepsets_fixed_n: int = 0,
):
    if act_layer not in ACT_LAYERS:
        raise ValueError(
            f"--act-layer must be one of {list(ACT_LAYERS)}, got '{act_layer}'"
        )
    _act_layer_cls = ACT_LAYERS[act_layer]

    local_rank, rank, size = ddp_setup()

    # set up model
    if arch == "pet2":
        model_params = get_model_parameters(model_size)
        model = PET2(
            input_dim=num_feat,
            use_int=interaction,
            local_int=local_interaction,
            int_type=interaction_type,
            conditional=conditional,
            cond_dim=num_cond,
            pid=use_pid,
            add_info=use_add,
            add_dim=num_add,
            mode=mode,
            num_classes=num_classes,
            num_gen_classes=num_gen_classes,
            num_coord=num_coord,
            K=K,
            **model_params,
        )
    elif arch == "deep-sets":
        ds_params = get_deepsets_parameters(model_size)
        model = DeepSets(
            input_dim=num_feat,
            num_classes=num_classes,
            pid=use_pid,
            add_info=use_add,
            add_dim=num_add,
            conditional=conditional,
            cond_dim=num_cond,
            mode=mode,
            energy_weighted_pool=energy_weighted_pool,
            num_interaction_layers=num_interaction_layers,
            interaction_k=interaction_k,
            act_layer=_act_layer_cls,
            fixed_n=deepsets_fixed_n,
            **ds_params,
        )
    elif arch == "mlp":
        mlp_params = get_mlp_parameters(model_size)
        model = MLPStudent(
            input_dim=num_feat,
            num_classes=num_classes,
            mode=mode,
            **mlp_params,
        )
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: pet2, deep-sets, mlp")

    if rank == 0:
        d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("**** Setup ****")
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )
        print(f"Evaluating on device: {d}, with {size} GPUs")
        print("************")

    test_loader = load_data(
        dataset,
        dataset_type=dataset_type,
        use_cond=True,
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        path=path,
        batch=batch,
        num_workers=num_workers,
        rank=rank,
        size=size,
        clip_inputs=clip_inputs,
        mode=mode,
        shuffle=False,
        num_chunks=num_chunks,
        chunk_idx=chunk_idx,
    )
    if rank == 0:
        print("**** Setup ****")
        print(f"Train dataset len: {len(test_loader)}")
        print("************")

    if os.path.isfile(os.path.join(indir, get_checkpoint_name(save_tag))):
        if is_master_node():
            print(
                f"Loading checkpoint from {os.path.join(indir, get_checkpoint_name(save_tag))}"
            )

        restore_checkpoint(
            model,
            indir,
            get_checkpoint_name(save_tag),
            local_rank,
            is_main_node=is_master_node(),
            restore_ema_model=mode == "generator",
        )

    else:
        raise ValueError(
            f"Error loading checkpoint: {os.path.join(indir, get_checkpoint_name(save_tag))}"
        )

    # Transfer model to GPU if available
    kwarg = {}
    if torch.cuda.is_available():
        device = local_rank
        model.to(local_rank)
        kwarg["device_ids"] = [device]
    else:
        model.cpu()
        device = "cpu"

    model = DDP(
        model,
        **kwarg,
    )

    eval_model(
        model,
        test_loader,
        dataset,
        mode=mode,
        use_event_loss=use_event_loss,
        device=device,
        rank=rank,
        outdir=outdir,
        save_tag=save_tag,
        dataset_type=dataset_type,
        num_chunks=num_chunks,
        chunk_idx=chunk_idx,
    )
    dist.barrier()
    dist.destroy_process_group()
