import argparse

def get_args():
    parser = argparse.ArgumentParser(description="args")
    
    parser.add_argument("--tau", default=0.015, type=float, help="temperature for ITSELF")
    parser.add_argument("--select_ratio", default=0.4, type=float, help="the ratio of selected tokens for ITSELF")
    parser.add_argument("--margin", default=0.1, type=float, help="margin + neg_score - pos_score for ITSELF")
    # parser.add_argument("--lambda1_weight", default=0.5, type=float) #TODO: please confirm whether those configs mean?
    # parser.add_argument("--lambda2_weight", default=3.5, type=float)

    ################################################ general settings ################################################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--output_dir", default="run_logs")
    parser.add_argument("--name", default="noname", help="experiment name to save")
    parser.add_argument("--log_period", default=50)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')
    parser.add_argument("--finetune", type=str, default="")
    parser.add_argument("--pretrain", type=str, default="")

    ################################################ model general settings ################################################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=True, action='store_true')
    parser.add_argument("--txt_aug", default=True, action='store_true')
    
    ################################################ loss settings ################################################
    parser.add_argument("--loss_names", default='tal+cid', help="which loss to use ['cid, tal']")

    ################################################ vision transformer settings ################################################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ################################################ text transformer settings ################################################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ################################################ solver ################################################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ################################################ scheduler ################################################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(45, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ################################################ dataset ################################################
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="identity", help="choose sampler from [identity, random]")
    parser.add_argument("--num_instance", type=int, default=2)
    parser.add_argument("--root_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    ################################################ GRAB ################################################
    parser.add_argument("--only_global", action='store_true')
    parser.add_argument("--return_all", action='store_true')
    parser.add_argument("--topk_type", type=str, default='mean', help='[mean, std, custom, layer_index]')
    parser.add_argument("--layer_index", type=int, default=-1, help='which layer attention to use: [0, 11]')
    parser.add_argument("--average_attn_weights", type=bool, default=True)
    parser.add_argument("--modify_k", action='store_true')
    
    #TODO: whether can train a CLIP baseline (not ITSELF) on this repo / --loss_names tal --only_global
    
    ################################################ new flag (no code yet) ################################################

    # reproducibility / experiment identity
    parser.add_argument("--seed", default=1, type=int,
                        help="random seed for reproducibility")

    # model / runtime mode
    parser.add_argument("--model_name", default="NONAME", help="top-level model name used by newer training pipeline")
    parser.add_argument("--model_variant", default="noname", help="variant name for logging / checkpoint naming")
    parser.add_argument("--training_mode", type=str, default="pas", help="training mode, e.g. pas or vanilla_clip")
    parser.add_argument("--host_type", type=str, default="clip", help="host branch type. use clip for CLIP host, itself for ITSELF-style host")

    # host loss control
    parser.add_argument("--use_host_loss", default=True, action="store_true",
                        help="enable host-side loss branch in newer pipeline")
    parser.add_argument("--lambda_host", type=float, default=1.0,
                        help="weight for host-side loss if enabled")

    # stage control
    parser.add_argument("--stage", "--training_stage", dest="training_stage",
                        type=str, default="stage1",
                        help="training stage, e.g. stage0 for host-only replication, stage1 for prototype training")

    # backbone / module freezing
    parser.add_argument("--freeze_host_projectors", default=False, action="store_true",
                        help="freeze host projector layers")
    parser.add_argument("--freeze_image_backbone", default=True, action="store_true",
                        help="freeze image backbone parameters")
    parser.add_argument("--freeze_text_backbone", default=True, action="store_true",
                        help="freeze text backbone parameters")
    parser.add_argument("--freeze_prototype_side", default=False, action="store_true",
                        help="freeze prototype-side parameters in prototype training stages")

    # mixed precision / numerical runtime
    parser.add_argument("--amp", default=False, action="store_true",
                        help="enable automatic mixed precision")
    parser.add_argument("--amp_dtype", type=str, default="fp16",
                        help="AMP dtype, usually fp16")
    parser.add_argument("--backbone_precision", type=str, default="fp16",
                        help="precision used by backbone forward path")
    parser.add_argument("--prototype_precision", type=str, default="fp32",
                        help="precision used by prototype-side modules")

    # training stability
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="gradient clipping max norm, 0 or negative usually means disabled depending on trainer")

    # checkpoint / eval runtime
    parser.add_argument("--checkpoint_path", "--checkpoint", dest="checkpoint", default="",
                        help="checkpoint path for evaluation or resume-like loading")
    parser.add_argument("--device", default="cuda",
                        help="runtime device")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="save checkpoint every N epochs")

    # logging / debugging
    parser.add_argument("--return_debug_outputs", default=False, action="store_true",
                        help="return extra debug tensors / statistics from model forward")
    parser.add_argument("--log_debug_metrics", default=True, action="store_true",
                        help="log extra debug metrics when supported by trainer")

    # wandb support
    parser.add_argument("--use_wandb", default=False, action="store_true",
                        help="enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="PAS",
                        help="wandb project name")
    parser.add_argument("--wandb_entity", default=None,
                        help="wandb entity / team name")
    parser.add_argument("--wandb_run_name", default=None,
                        help="explicit wandb run name")
    parser.add_argument("--wandb_group", default=None,
                        help="wandb group name for grouped experiments")
    parser.add_argument("--wandb_mode", default="online",
                        help="wandb mode, e.g. online or offline")
    parser.add_argument("--wandb_tags", nargs='*', default=[],
                        help="optional wandb tags")
    parser.add_argument("--wandb_notes", default=None,
                        help="optional wandb notes")
    parser.add_argument("--wandb_log_interval", type=int, default=50,
                        help="step interval for wandb logging")
    parser.add_argument("--wandb_log_code", default=False, action="store_true",
                        help="ask wandb to snapshot code")
    parser.add_argument("--nohup", default=False, action="store_true",
                        help="marker flag for nohup-based launching, mainly for bookkeeping")

    # retrieval / evaluation extensions from newer parser
    parser.add_argument("--retrieval_mode", type=str, default="surrogate_i2t",
                        help="retrieval scoring mode used by newer PAS pipeline")
    parser.add_argument("--retrieval_scorer", type=str, default="exact",
                        help="retrieval scorer backend used at evaluation time")
    parser.add_argument("--retrieval_metrics", nargs='+',
                        default=['R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum'],
                        help="retrieval metrics to report")
    parser.add_argument("--prototype_eval_image_chunk_size", type=int, default=32,
                        help="image chunk size for memory-safe evaluation")
    parser.add_argument("--prototype_eval_text_chunk_size", type=int, default=128,
                        help="text chunk size for memory-safe evaluation")

    # cross-domain / target-domain evaluation
    parser.add_argument("--cross_domain_generalization", default=False, action="store_true",
                        help="enable cross-domain generalization evaluation mode")
    parser.add_argument("--target_domain", default="RSTPReid",
                        help="target domain name for cross-domain evaluation")

    # newer optimizer naming compatibility
    parser.add_argument("--optimizer_type", type=str, default=None,
                        help="alias for newer codebases that refer to optimizer as optimizer_type")

    # newer scheduler naming compatibility
    parser.add_argument("--lr_scheduler", type=str, default=None,
                        help="alias for newer codebases that refer to scheduler as lr_scheduler")

    # prototype-side / PAS-side optional compatibility flags
    # Các flag này không thuộc ITSELF gốc, nhưng được thêm để parser cũ không bị lỗi
    # khi dùng config hoặc launch script từ codebase mới.
    parser.add_argument("--use_prototype_branch", default=False, action="store_true",
                        help="enable prototype branch when using PAS-style training")
    parser.add_argument("--use_prototype_bank", default=False, action="store_true",
                        help="enable prototype bank")
    parser.add_argument("--use_image_conditioned_pooling", default=False, action="store_true",
                        help="enable image-conditioned text pooling")
    parser.add_argument("--fusion_enabled", default=False, action="store_true",
                        help="enable host/prototype fusion")
    parser.add_argument("--fusion_coefficient", type=float, default=1.0,
                        help="fixed fusion coefficient if fusion is enabled")
    parser.add_argument("--fusion_coefficient_source", type=str, default="fixed",
                        help="how fusion coefficient is produced")

    # prototype bank configuration
    parser.add_argument("--prototype_num_prototypes", type=int, default=32,
                        help="number of prototypes in prototype bank")
    parser.add_argument("--prototype_dim", type=int, default=512,
                        help="prototype embedding dimension")
    parser.add_argument("--prototype_init", type=str, default="normalized_random",
                        help="prototype initialization strategy")
    parser.add_argument("--prototype_init_path", type=str, default="",
                        help="optional path to cached embeddings / centroids for prototype init")
    parser.add_argument("--prototype_init_hybrid_ratio", type=float, default=0.5,
                        help="hybrid init ratio when using hybrid prototype initialization")
    parser.add_argument("--prototype_init_max_iters", type=int, default=50,
                        help="max iterations for clustering-based prototype init")
    parser.add_argument("--prototype_init_tol", type=float, default=1e-4,
                        help="tolerance for clustering-based prototype init")
    parser.add_argument("--prototype_init_seed", type=int, default=None,
                        help="seed for prototype initialization when applicable")

    # prototype routing / contextualization
    parser.add_argument("--routing_similarity", type=str, default="cosine",
                        help="routing similarity type for prototype selection")
    parser.add_argument("--tau_p", type=float, default=0.07,
                        help="prototype routing temperature")
    parser.add_argument("--prototype_contextualization_enabled", default=False, action="store_true",
                        help="enable prototype contextualization")
    parser.add_argument("--prototype_contextualization_type", type=str, default="self_attention",
                        help="prototype contextualization module type")
    parser.add_argument("--prototype_contextualization_residual", default=True, action="store_true",
                        help="use residual connection in prototype contextualization")
    parser.add_argument("--normalize_for_self_interaction", default=True, action="store_true",
                        help="normalize prototype vectors before self interaction")
    parser.add_argument("--normalize_for_routing", default=True, action="store_true",
                        help="normalize features before routing")
    parser.add_argument("--prototype_dead_threshold", type=float, default=0.005,
                        help="threshold to mark prototypes as effectively dead")

    # prototype regularization
    parser.add_argument("--use_balancing_loss", default=False, action="store_true",
                        help="enable prototype balancing loss")
    parser.add_argument("--lambda_bal", type=float, default=0.0,
                        help="weight for prototype balancing loss")
    parser.add_argument("--use_diversity_loss", default=False, action="store_true",
                        help="enable prototype diversity loss")
    parser.add_argument("--lambda_div", type=float, default=0.01,
                        help="weight for prototype diversity loss")
    parser.add_argument("--use_loss_support", default=False, action="store_true",
                        help="enable support regularization loss on routing distribution")
    parser.add_argument("--lambda_support", type=float, default=0.0,
                        help="weight for support loss")
    parser.add_argument("--support_min", type=float, default=2.0,
                        help="minimum effective support target")

    # token scoring / pooling compatibility
    parser.add_argument("--token_policy", type=str, default="content_only",
                        help="token selection policy for prototype-side pooling")
    parser.add_argument("--token_similarity", type=str, default="cosine",
                        help="token scoring similarity type")
    parser.add_argument("--normalize_for_token_scoring", default=True, action="store_true",
                        help="normalize vectors before token scoring")
    parser.add_argument("--tau_t", type=float, default=0.07,
                        help="token pooling temperature")
    parser.add_argument("--error_on_empty_kept_tokens", default=True, action="store_true",
                        help="raise error if token selection removes all valid content tokens")

    # additional loss toggles from newer parser
    parser.add_argument("--use_loss_align", default=False, action="store_true",
                        help="enable alignment loss")
    parser.add_argument("--lambda_align", type=float, default=1.0,
                        help="weight for alignment loss")
    parser.add_argument("--use_loss_diag", default=False, action="store_true",
                        help="enable diagonal fidelity loss")
    parser.add_argument("--lambda_diag", type=float, default=1.0,
                        help="weight for diagonal fidelity loss")
    parser.add_argument("--use_loss_ret", default=False, action="store_true",
                        help="enable retrieval loss in newer PAS pipeline")
    parser.add_argument("--lambda_ret", type=float, default=1.0,
                        help="weight for retrieval loss")
    parser.add_argument("--lambda_proxy", type=float, default=1.0,
                        help="default shared weight for proxy losses")
    parser.add_argument("--lambda_proxy_image", type=float, default=None,
                        help="weight for image proxy loss")
    parser.add_argument("--lambda_proxy_text", type=float, default=None,
                        help="weight for text proxy loss")
    parser.add_argument("--lambda_proxy_text_exact", type=float, default=None,
                        help="weight for exact-text proxy loss")
    parser.add_argument("--use_loss_proxy_image", default=False, action="store_true",
                        help="enable image-side proxy loss")
    parser.add_argument("--use_loss_proxy_text", default=False, action="store_true",
                        help="enable text-side proxy loss")
    parser.add_argument("--use_loss_proxy_text_exact", default=False, action="store_true",
                        help="enable exact-text proxy loss")

    # finer learning rates for newer optimizer builders
    parser.add_argument("--lr_prototype_bank", type=float, default=None,
                        help="optional lr override for prototype bank")
    parser.add_argument("--lr_projectors", type=float, default=None,
                        help="optional lr override for projector modules")
    parser.add_argument("--lr_prototype_routing", type=float, default=None,
                        help="optional lr override for routing submodule")
    parser.add_argument("--lr_prototype_pooling", type=float, default=None,
                        help="optional lr override for pooling submodule")
    parser.add_argument("--lr_prototype_contextualization", type=float, default=None,
                        help="optional lr override for contextualization submodule")
    parser.add_argument("--lr_host_projectors", type=float, default=None,
                        help="optional lr override for host projector modules")
    parser.add_argument("--lr_image_backbone", type=float, default=None,
                        help="optional lr override for image backbone")
    parser.add_argument("--lr_text_backbone", type=float, default=None,
                        help="optional lr override for text backbone")
    parser.add_argument("--lr_class_proxies", type=float, default=None,
                        help="optional lr override for class proxy parameters")

    # finer weight decay for newer optimizer builders
    parser.add_argument("--weight_decay_prototype_bank", type=float, default=None,
                        help="optional weight decay override for prototype bank")
    parser.add_argument("--weight_decay_projectors", type=float, default=None,
                        help="optional weight decay override for projector modules")
    parser.add_argument("--weight_decay_prototype_routing", type=float, default=None,
                        help="optional weight decay override for routing submodule")
    parser.add_argument("--weight_decay_prototype_pooling", type=float, default=None,
                        help="optional weight decay override for pooling submodule")
    parser.add_argument("--weight_decay_prototype_contextualization", type=float, default=None,
                        help="optional weight decay override for contextualization submodule")
    parser.add_argument("--weight_decay_host_projectors", type=float, default=None,
                        help="optional weight decay override for host projectors")
    parser.add_argument("--weight_decay_class_proxies", type=float, default=None,
                        help="optional weight decay override for class proxy parameters")
    parser.add_argument("--weight_decay_image_backbone", type=float, default=None,
                        help="optional weight decay override for image backbone")
    parser.add_argument("--weight_decay_text_backbone", type=float, default=None,
                        help="optional weight decay override for text backbone")
    parser.add_argument("--optimizer_eps", type=float, default=1e-8,
                        help="epsilon used by optimizers such as Adam / AdamW")
    
    
    
    args = parser.parse_args()
    return args
