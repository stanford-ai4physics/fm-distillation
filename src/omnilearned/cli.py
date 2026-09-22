import typer

from omnilearned.train import run as run_training
from omnilearned.evaluate import run as run_evaluation
from omnilearned.dataloader import load_data
from omnilearned.train_hl import run as run_training_hl
from omnilearned.evaluate_hl import run as run_evaluation_hl
from omnilearned.omnifold import run as run_omnifold

app = typer.Typer(
    help="OmniLearned: A unified deep learning approach for particle physics",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.command()
def train(
    # General Options
    outdir: str = typer.Option(
        "", "--output_dir", "-o", help="Directory to output best model"
    ),
    save_tag: str = typer.Option("", help="Extra tag for checkpoint model"),
    pretrain_tag: str = typer.Option(
        "", help="Tag given to pretrained checkpoint model"
    ),
    dataset: str = typer.Option("top", help="Dataset to load"),
    path: str = typer.Option("/pscratch/sd/v/vmikuni/datasets", help="Dataset path"),
    wandb: bool = typer.Option(False, help="use wandb logging"),
    fine_tune: bool = typer.Option(False, help="Fine tune the model"),
    resuming: bool = typer.Option(False, help="Resume training"),
    seed: int = typer.Option(
        -1, help="Base RNG seed (torch/numpy/random, +rank per DDP replica). "
                 "-1 (default) leaves runs unseeded / non-deterministic."
    ),
    # Model Options
    num_feat: int = typer.Option(
        4,
        help="Number of input kinematic features (not considering PID or additional features)",
    ),
    size: str = typer.Option("small", "--size", "-s", help="Model size"),
    interaction: bool = typer.Option(False, help="Use interaction matrix"),
    local_interaction: bool = typer.Option(False, help="Use local interaction matrix"),
    num_coord: int = typer.Option(
        2, help="Number of features for distance calculation"
    ),
    K: int = typer.Option(10, help="Number of k-neighbors"),
    interaction_type: str = typer.Option("lhc", help="Type of interaction"),
    conditional: bool = typer.Option(False, help="Use global conditional features"),
    num_cond: int = typer.Option(3, help="Number of global conditioning features"),
    use_pid: bool = typer.Option(False, help="Use particle ID for training"),
    pid_idx: int = typer.Option(4, help="Index of the PID in the input array"),
    pid_dim: int = typer.Option(9, help="Number of unique PIDs"),
    use_add: bool = typer.Option(
        False, help="Use additional features beyond kinematic information"
    ),
    num_add: int = typer.Option(4, help="Number of additional features"),
    zero_add: bool = typer.Option(
        False,
        help="Load the model with additional blocks but zero the inputs from the dataloader",
    ),
    use_clip: bool = typer.Option(False, help="Use CLIP loss during training"),
    use_event_loss: bool = typer.Option(
        False, help="Use additional classification loss between physics process"
    ),
    num_classes: int = typer.Option(
        2, help="Number of classes in the classification task"
    ),
    num_gen_classes: int = typer.Option(
        1, help="Number of classes in the particle classification task"
    ),
    mode: str = typer.Option(
        "classifier", help="Task to run: classifier, generator, pretrain"
    ),
    # Training options
    batch: int = typer.Option(64, help="Batch size"),
    iterations: int = typer.Option(-1, help="Number of iterations per pass"),
    epoch: int = typer.Option(10, help="Number of epochs"),
    warmup_epoch: int = typer.Option(0, help="Number of learning rate warmup epochs"),
    use_amp: bool = typer.Option(False, help="Use amp"),
    amp_dtype: str = typer.Option(
        "fp16", help="Autocast dtype when --use-amp is set: fp16 or bf16"
    ),
    clip_inputs: bool = typer.Option(
        False, help="Clip input dataset to be within R=0.8 and atl least 500 MeV"
    ),
    # Optimizer
    optim: str = typer.Option("lion", help="optimizer to use"),
    sched: str = typer.Option("cosine", help="lr scheduler to use"),
    b1: float = typer.Option(0.95, help="Lion b1"),
    b2: float = typer.Option(0.98, help="Lion b2"),
    lr: float = typer.Option(5e-5, help="Learning rate"),
    lr_factor: float = typer.Option(
        1.0, help="Learning rate factor for new layers during fine-tuning"
    ),
    wd: float = typer.Option(0.0, help="Weight decay"),
    nevts: int = typer.Option(-1, help="Maximum number of events to use"),
    # Model
    attn_drop: float = typer.Option(0.0, help="Dropout for attention layers"),
    mlp_drop: float = typer.Option(0.0, help="Dropout for mlp layers"),
    feature_drop: float = typer.Option(0.0, help="Dropout for input features"),
    num_workers: int = typer.Option(16, help="Number of workers for data loading"),
    # Distillation options (offline: requires pre-generated teacher logits)
    distill: bool = typer.Option(
        False, help="Enable offline KD against pre-saved teacher logits"
    ),
    teacher_labels_dir: str = typer.Option(
        "", help="Directory containing teacher outputs_*.npz files (saved by evaluate)"
    ),
    teacher_tag: str = typer.Option(
        "", help="Teacher save_tag; selects outputs_{tag}_{dataset}_{type}_*.npz"
    ),
    distill_alpha: float = typer.Option(0.5, help="Weight for task loss"),
    distill_beta: float = typer.Option(0.5, help="Weight for KL distillation"),
    distill_T: float = typer.Option(4.0, help="Temperature for KL distillation"),
    distill_teacher_slice: str = typer.Option(
        "",
        help="Column slice of teacher logits for KD, e.g. '2:12' to use columns "
             "2-11 from a 210-class pretrained teacher for a 10-class student. "
             "Default '' keeps all columns.",
    ),
    distill_standardize: bool = typer.Option(
        False,
        help="Per-sample Z-score of teacher and student logits before the KD "
             "softmax (Logit Standardization, CVPR 2024). Decouples "
             "teacher/student logit magnitude; helps under a large capacity gap.",
    ),
    distill_cls: bool = typer.Option(
        False,
        help="Also match the student's body-token embedding (outputs['x_body']) "
             "against a pre-saved teacher embedding via MSE (CLS-MSE feature "
             "distillation). Requires --arch pet2 and teacher companion files "
             "built with `tools/preprocess/build_teacher_h5.py --include-cls-embed`.",
    ),
    distill_gamma: float = typer.Option(
        0.5, help="Weight for the CLS-MSE feature distillation term"
    ),
    distill_cls_teacher_dim: int = typer.Option(
        1024,
        help="Teacher's base_dim, for sizing the student's cls_projector "
             "(default 1024 matches --size large)",
    ),
    arch: str = typer.Option("pet2", help="Student architecture: pet2, deep-sets, or mlp"),
    energy_weighted_pool: bool = typer.Option(
        False,
        help="DeepSets only: pool per-particle embeddings weighted by raw pT "
             "instead of a plain masked mean",
    ),
    num_interaction_layers: int = typer.Option(
        0,
        help="DeepSets only: number of particle-particle message-passing "
             "(GNN) layers inserted before the φ blocks. 0 (default) = plain "
             "Deep Sets, checkpoint-compatible.",
    ),
    interaction_k: int = typer.Option(
        0,
        help="DeepSets only: with --num-interaction-layers > 0, keep just the "
             "leading-pT K constituents before the body (0 = keep all). "
             "Caps the O(N²) message-passing cost.",
    ),
    act_layer: str = typer.Option(
        "gelu",
        help="DeepSets only: activation used in every MLP block. 'gelu' "
             "(default) or 'relu' (hls4ml-friendly; GELU has no ONNX/hls4ml "
             "lowering).",
    ),
    deepsets_fixed_n: int = typer.Option(
        0,
        help="DeepSets only: if >0, build the hls4ml-friendly body -- truncate "
             "to the leading-pT N constituents, drop the in-graph validity "
             "mask, and plain-mean-pool over all N slots. 0 (default) keeps "
             "the masked-mean behavior. Must match between train and evaluate.",
    ),
):
    run_training(
        outdir,
        save_tag,
        pretrain_tag,
        dataset,
        path,
        wandb,
        fine_tune,
        resuming,
        seed,
        num_feat,
        size,
        interaction,
        local_interaction,
        num_coord,
        K,
        interaction_type,
        conditional,
        num_cond,
        use_pid,
        pid_idx,
        pid_dim,
        use_add,
        num_add,
        zero_add,
        use_clip,
        use_event_loss,
        num_classes,
        num_gen_classes,
        mode,
        batch,
        iterations,
        epoch,
        warmup_epoch,
        use_amp,
        amp_dtype,
        optim,
        sched,
        b1,
        b2,
        lr,
        lr_factor,
        wd,
        nevts,
        attn_drop,
        mlp_drop,
        feature_drop,
        num_workers,
        clip_inputs=clip_inputs,
        distill=distill,
        teacher_labels_dir=teacher_labels_dir,
        teacher_tag=teacher_tag,
        distill_alpha=distill_alpha,
        distill_beta=distill_beta,
        distill_T=distill_T,
        distill_teacher_slice=distill_teacher_slice,
        distill_standardize=distill_standardize,
        distill_cls=distill_cls,
        distill_gamma=distill_gamma,
        distill_cls_teacher_dim=distill_cls_teacher_dim,
        arch=arch,
        energy_weighted_pool=energy_weighted_pool,
        num_interaction_layers=num_interaction_layers,
        interaction_k=interaction_k,
        act_layer=act_layer,
        deepsets_fixed_n=deepsets_fixed_n,
    )


@app.command()
def unfold(
    outdir: str = typer.Option(
        "", "--output_dir", "-o", help="Directory to output checkpoints"
    ),
    save_tag: str = typer.Option("", help="Extra tag for checkpoint models"),
    pretrain_tag: str = typer.Option(
        "", help="Tag given to pretrained checkpoint model (with --fine-tune)"
    ),
    path: str = typer.Option(
        "/pscratch/sd/t/twamorka/unfolding",
        help="Directory containing train_pythia.h5 / train_herwig.h5 "
        "(see tools/preprocess/preprocess_omnifold.py)",
    ),
    wandb: bool = typer.Option(False, help="use wandb logging"),
    fine_tune: bool = typer.Option(
        False, help="Warm-start iteration-0 step-1 model from --pretrain-tag"
    ),
    num_feat: int = typer.Option(
        13, help="Number of input per-particle features (13 for the OmniLearn "
        "tools/preprocess/preprocess_omnifold.py schema)"
    ),
    size: str = typer.Option("small", "--size", "-s", help="Model size"),
    interaction: bool = typer.Option(False, help="Use interaction matrix"),
    local_interaction: bool = typer.Option(False, help="Use local interaction matrix"),
    num_iter: int = typer.Option(5, help="Number of OmniFold iterations"),
    patience: int = typer.Option(
        3, help="Early-stopping patience (epochs) within each Step1/Step2 fit"
    ),
    batch: int = typer.Option(512, help="Batch size"),
    epoch: int = typer.Option(30, help="Max epochs per Step1/Step2 fit"),
    warmup_epoch: int = typer.Option(1, help="Number of learning rate warmup epochs"),
    lr: float = typer.Option(3e-5, help="Learning rate"),
    lr_factor: float = typer.Option(
        5.0, help="Learning rate factor for new layers when --fine-tune is set"
    ),
    wd: float = typer.Option(0.1, help="Weight decay"),
    b1: float = typer.Option(0.95, help="Lion b1"),
    b2: float = typer.Option(0.99, help="Lion b2"),
    optim: str = typer.Option("lion", help="optimizer to use"),
    sched: str = typer.Option("cosine", help="lr scheduler to use"),
    use_amp: bool = typer.Option(False, help="Use amp"),
    amp_dtype: str = typer.Option(
        "fp16", help="Autocast dtype when --use-amp is set: fp16 or bf16"
    ),
    num_workers: int = typer.Option(
        0,
        help="Number of DataLoader workers. Defaults to 0 (in-process, no fork): "
        "unlike train.py's single wandb.init(), unfold builds fresh DataLoaders "
        "on every Step1/Step2 call after wandb is already live, which would "
        "refork workers post-wandb.init() on every step and risk the documented "
        "wandb-forked-worker deadlock (see distill-lazy-teacher-progress memory, "
        "2026-08-08). Data is already fully in-memory here, so workers buy little "
        "anyway -- only raise this if you've verified it's safe for your run.",
    ),
):
    run_omnifold(
        outdir,
        save_tag,
        pretrain_tag,
        path,
        wandb,
        fine_tune,
        num_feat,
        size,
        interaction,
        local_interaction,
        num_iter,
        patience,
        batch,
        epoch,
        warmup_epoch,
        lr,
        lr_factor,
        wd,
        b1,
        b2,
        optim,
        sched,
        use_amp,
        amp_dtype,
        num_workers,
    )


@app.command()
def train_hl(
    # General Options
    outdir: str = typer.Option(
        "", "--output_dir", "-o", help="Directory to output best model"
    ),
    save_tag: str = typer.Option("", help="Extra tag for checkpoint model"),
    dataset: str = typer.Option("top", help="Dataset to load"),
    path: str = typer.Option("/pscratch/sd/v/vmikuni/datasets", help="Dataset path"),
    wandb: bool = typer.Option(False, help="use wandb logging"),
    resuming: bool = typer.Option(False, help="Resume training"),
    # Model Options
    num_feat: int = typer.Option(
        3,
        help="Number of input kinematic features (not considering PID or additional features)",
    ),
    conditional: bool = typer.Option(False, help="Use global conditional features"),
    num_cond: int = typer.Option(1, help="Number of global conditioning features"),
    # Training options
    batch: int = typer.Option(64, help="Batch size"),
    iterations: int = typer.Option(-1, help="Number of iterations per pass"),
    epoch: int = typer.Option(10, help="Number of epochs"),
    warmup_epoch: int = typer.Option(0, help="Number of learning rate warmup epochs"),
    # Optimizer
    optim: str = typer.Option("lion", help="optimizer to use"),
    b1: float = typer.Option(0.95, help="Lion b1"),
    b2: float = typer.Option(0.98, help="Lion b2"),
    lr: float = typer.Option(5e-5, help="Learning rate"),
    wd: float = typer.Option(0.0, help="Weight decay"),
    # Model
    mlp_drop: float = typer.Option(0.0, help="Dropout for mlp layers"),
    num_workers: int = typer.Option(16, help="Number of workers for data loading"),
):
    run_training_hl(
        outdir,
        save_tag,
        dataset,
        path,
        wandb,
        resuming,
        num_feat,
        conditional,
        num_cond,
        batch,
        iterations,
        epoch,
        warmup_epoch,
        optim,
        b1,
        b2,
        lr,
        wd,
        mlp_drop,
    )


@app.command()
def evaluate(
    # General Options
    indir: str = typer.Option(
        "", "--input_dir", "-i", help="Directory to input best model"
    ),
    outdir: str = typer.Option(
        "", "--output_dir", "-o", help="Directory to output evaluation results"
    ),
    save_tag: str = typer.Option("", help="Extra tag for checkpoint model"),
    dataset: str = typer.Option("top", help="Dataset to load"),
    path: str = typer.Option("/pscratch/sd/v/vmikuni/datasets", help="Dataset path"),
    # Model Options
    num_feat: int = typer.Option(
        4,
        help="Number of input kinematic features (not considering PID or additional features)",
    ),
    size: str = typer.Option("small", "--size", "-s", help="Model size"),
    interaction: bool = typer.Option(False, help="Use interaction matrix"),
    local_interaction: bool = typer.Option(False, help="Use local interaction matrix"),
    num_coord: int = typer.Option(
        2, help="Number of features for distance calculation"
    ),
    K: int = typer.Option(10, help="Number of k-neighbors"),
    interaction_type: str = typer.Option("lhc", help="Type of interaction"),
    conditional: bool = typer.Option(False, help="Use global conditional features"),
    num_cond: int = typer.Option(3, help="Number of global conditioning features"),
    use_pid: bool = typer.Option(False, help="Use particle ID for training"),
    pid_idx: int = typer.Option(4, help="Index of the PID in the input array"),
    use_add: bool = typer.Option(
        False, help="Use additional features beyond kinematic information"
    ),
    num_add: int = typer.Option(4, help="Number of additional features"),
    use_event_loss: bool = typer.Option(
        False, help="Use additional classification loss between physics process"
    ),
    num_classes: int = typer.Option(
        2, help="Number of classes in the classification task"
    ),
    num_gen_classes: int = typer.Option(
        1, help="Number of classes in the particle segmentation task"
    ),
    mode: str = typer.Option(
        "classifier", help="Task to run: classifier, generator, pretrain"
    ),
    # Training options
    batch: int = typer.Option(128, help="Batch size"),
    clip_inputs: bool = typer.Option(
        False, help="Clip input dataset to be within R=0.8 and atl least 500 MeV"
    ),
    num_workers: int = typer.Option(16, help="Number of workers for data loading"),
    dataset_type: str = typer.Option(
        "test",
        "--dataset-type",
        help="Which split to evaluate on: train, test, or val",
    ),
    num_chunks: int = typer.Option(
        1,
        "--num-chunks",
        help="Split the dataset into N chunks; run one chunk per job and concat after.",
    ),
    chunk_idx: int = typer.Option(
        0,
        "--chunk-idx",
        help="Which chunk (0..num_chunks-1) this invocation processes.",
    ),
    arch: str = typer.Option("pet2", help="Student architecture: pet2, deep-sets, or mlp"),
    energy_weighted_pool: bool = typer.Option(
        False,
        help="DeepSets only: pool per-particle embeddings weighted by raw pT "
             "instead of a plain masked mean",
    ),
    num_interaction_layers: int = typer.Option(
        0,
        help="DeepSets only: must match the value used at training time so the "
             "checkpoint loads (0 = plain Deep Sets).",
    ),
    interaction_k: int = typer.Option(
        0,
        help="DeepSets only: leading-pT constituent cap; must match training.",
    ),
    act_layer: str = typer.Option(
        "gelu",
        help="DeepSets only: 'gelu' (default) or 'relu'; must match training.",
    ),
    deepsets_fixed_n: int = typer.Option(
        0,
        help="DeepSets only: fixed leading-pT N / no-mask body; must match "
             "training (0 = masked-mean body).",
    ),
):
    run_evaluation(
        indir,
        outdir,
        save_tag,
        dataset,
        path,
        num_feat,
        size,
        interaction,
        local_interaction,
        num_coord,
        K,
        interaction_type,
        conditional,
        num_cond,
        use_pid,
        pid_idx,
        use_add,
        num_add,
        use_event_loss,
        num_classes,
        num_gen_classes,
        mode,
        batch,
        num_workers,
        clip_inputs=clip_inputs,
        dataset_type=dataset_type,
        num_chunks=num_chunks,
        chunk_idx=chunk_idx,
        arch=arch,
        energy_weighted_pool=energy_weighted_pool,
        num_interaction_layers=num_interaction_layers,
        interaction_k=interaction_k,
        act_layer=act_layer,
        deepsets_fixed_n=deepsets_fixed_n,
    )


@app.command()
def evaluate_hl(
    # General Options
    indir: str = typer.Option(
        "", "--input_dir", "-i", help="Directory to input best model"
    ),
    outdir: str = typer.Option("", "--output_dir", "-o", help="Output saved files"),
    save_tag: str = typer.Option("", help="Extra tag for checkpoint model"),
    dataset: str = typer.Option("top", help="Dataset to load"),
    path: str = typer.Option("/pscratch/sd/v/vmikuni/datasets", help="Dataset path"),
    # Model Options
    num_feat: int = typer.Option(
        3,
        help="Number of input kinematic features (not considering PID or additional features)",
    ),
    conditional: bool = typer.Option(False, help="Use global conditional features"),
    num_cond: int = typer.Option(3, help="Number of global conditioning features"),
    # Training options
    batch: int = typer.Option(128, help="Batch size"),
    num_workers: int = typer.Option(16, help="Number of workers for data loading"),
):
    run_evaluation_hl(
        indir,
        outdir,
        save_tag,
        dataset,
        path,
        num_feat,
        conditional,
        num_cond,
        batch,
        num_workers,
    )


@app.command()
def dataloader(
    dataset: str = typer.Option(
        "top", "--dataset", "-d", help="Dataset name to download"
    ),
    folder: str = typer.Option(
        "./", "--folder", "-f", help="Folder to save the dataset"
    ),
):
    for tag in ["train", "test", "val"]:
        print(tag)
        load_data(dataset, folder, dataset_type=tag, distributed=False)


if __name__ == "__main__":
    app()
