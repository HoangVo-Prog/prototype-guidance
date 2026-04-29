import copy
import logging
import os.path as op
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
import random
from utils.comm import get_world_size
import numpy as np
from .bases import ImageDataset, TextDataset, ImageTextDataset
from utils.repro import make_torch_generator, seed_worker

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid}


def _use_legacy_joint_sampler(args) -> bool:
    return (
        str(getattr(args, 'runtime_mode', '') or '').strip().lower() == 'joint_training'
        and bool(getattr(args, 'use_prototype_branch', False))
    )


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform



def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], (int, np.int64)):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict




def _resolve_image_path_from_anno(dataset, anno):
    relative_path = anno.get('img_path', anno.get('file_path'))
    if relative_path is None:
        raise KeyError('Expected annotation to contain `img_path` or `file_path` for evaluation loss monitoring.')
    if op.isabs(relative_path):
        return relative_path
    return op.join(dataset.img_dir, relative_path)



def _build_paired_records_from_annos(dataset, annos):
    records = []
    for image_id, anno in enumerate(annos):
        pid = int(anno['id'])
        img_path = _resolve_image_path_from_anno(dataset, anno)
        for caption in anno.get('captions', []):
            records.append((pid, image_id, img_path, caption))
    return records

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("pas.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    if args.dataset_name == 'Flickr' or args.dataset_name == 'MSCOCO':
        args.img_size = (224,224)
    num_classes = len(dataset.train_id_container)
    random.seed(42)
    ds1 = dataset.train
    all_text = []
    all_id = []
    for a_, b_, c_, text in ds1:
        all_id.append(a_)
        all_text.append(text)

    
    if args.training:
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)


        train_set = ImageTextDataset(dataset.train,args,
                            train_transforms,
                        text_length=args.text_length)

        reproducible = bool(getattr(args, 'repro_enabled', False))
        use_gen = reproducible and bool(getattr(args, 'repro_dataloader_generator', True))
        use_worker_seed = reproducible and bool(getattr(args, 'repro_dataloader_worker_seed', True))
        loader_generator = make_torch_generator(getattr(args, 'repro_seed', getattr(args, 'seed', 0))) if use_gen else None
        worker_init = seed_worker if use_worker_seed else None
        if args.sampler == 'identity':
            if reproducible:
                sampler_seed = int(getattr(args, 'repro_seed', getattr(args, 'seed', 0)))
            else:
                sampler_seed = None if _use_legacy_joint_sampler(args) else getattr(args, 'seed', 0)
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance, seed=sampler_seed)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)
                train_loader = DataLoader(train_set,
                                          batch_sampler=batch_sampler,
                                          num_workers=num_workers,
                                          collate_fn=collate,
                                          worker_init_fn=worker_init,
                                          generator=loader_generator)
            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance,
                                              seed=sampler_seed),
                                          num_workers=num_workers,
                                          collate_fn=collate,
                                          worker_init_fn=worker_init,
                                          generator=loader_generator)
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate,
                                      worker_init_fn=worker_init,
                                      generator=loader_generator)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

        eval_loss_loader = None
        preferred_eval_annos = 'val_annos' if args.val_dataset == 'val' else 'test_annos'
        fallback_eval_annos = 'test_annos' if preferred_eval_annos == 'val_annos' else 'val_annos'
        eval_annos = getattr(dataset, preferred_eval_annos, None)
        if eval_annos is None:
            eval_annos = getattr(dataset, fallback_eval_annos, None)
        if eval_annos:
            val_monitor_args = copy.copy(args)
            val_monitor_args.txt_aug = False
            val_monitor_args.img_aug = False
            eval_pair_records = _build_paired_records_from_annos(dataset, eval_annos)
            if eval_pair_records:
                eval_loss_set = ImageTextDataset(
                    eval_pair_records,
                    val_monitor_args,
                    val_transforms,
                    text_length=args.text_length,
                )
                eval_loss_loader = DataLoader(
                    eval_loss_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=collate,
                )
        train_loader.eval_loss_loader = eval_loss_loader
        if reproducible and bool(getattr(args, 'repro_proto_deterministic_recompute', True)):
            proto_args = copy.copy(args)
            proto_args.txt_aug = False
            proto_args.img_aug = False
            proto_set = ImageTextDataset(
                dataset.train,
                proto_args,
                val_transforms,
                text_length=args.text_length,
            )
            train_loader.proto_recompute_loader = DataLoader(
                proto_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate,
                worker_init_fn=worker_init,
                generator=loader_generator,
            )

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_img_loader.test_img_set = test_img_set
        test_txt_loader.test_txt_set = test_txt_set
        
        return test_img_loader, test_txt_loader, num_classes




