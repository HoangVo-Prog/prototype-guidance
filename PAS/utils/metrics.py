import logging

from prettytable import PrettyTable
import torch

from utils.precision import build_autocast_context, is_cuda_device


SUPPORTED_RETRIEVAL_METRICS = ('R1', 'R5', 'R10', 'mAP', 'mINP', 'rSum')


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(similarity, k=max_rank, dim=1, largest=True, sorted=True)
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)
    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.0) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def get_metrics(similarity, qids, gids, name):
    t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    return {
        'task': name,
        'R1': float(t2i_cmc[0]),
        'R5': float(t2i_cmc[4]),
        'R10': float(t2i_cmc[9]),
        'mAP': float(t2i_mAP),
        'mINP': float(t2i_mINP),
        'rSum': float(t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]),
    }


class Evaluator:
    def __init__(self, img_loader, txt_loader, args):
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.logger = logging.getLogger('pas.eval')
        self.args = args
        self.latest_metrics = {}
        requested_metrics = tuple(getattr(args, 'retrieval_metrics', SUPPORTED_RETRIEVAL_METRICS) or SUPPORTED_RETRIEVAL_METRICS)
        unknown_metrics = sorted(set(requested_metrics) - set(SUPPORTED_RETRIEVAL_METRICS))
        if unknown_metrics:
            raise ValueError(
                f'Unsupported evaluation.retrieval_metrics values: {unknown_metrics}. '
                f'Allowed values: {list(SUPPORTED_RETRIEVAL_METRICS)}'
            )
        self.requested_metrics = requested_metrics

    def _concat_feature_batches(self, batches):
        first = batches[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(batches, 0)
        if isinstance(first, dict):
            return {key: self._concat_feature_batches([batch[key] for batch in batches]) for key in first.keys()}
        raise TypeError(f'Unsupported feature batch type: {type(first)}')

    def _compute_similarity(self, model):
        model = model.eval()
        device = next(model.parameters()).device
        if bool(getattr(self.args, 'amp', False)) and not is_cuda_device(device):
            raise ValueError('training.amp=true requires a CUDA device.')

        text_ids, image_ids = [], []
        text_batches, image_batches = [], []

        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                with build_autocast_context(self.args, device):
                    text_features = model.encode_text_for_retrieval(caption)
            text_ids.append(pid.view(-1))
            text_batches.append({
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else {sub_key: sub_value.detach().cpu() for sub_key, sub_value in value.items()}
                for key, value in text_features.items()
            })

        for pid, image in self.img_loader:
            image = image.to(device)
            with torch.no_grad():
                with build_autocast_context(self.args, device):
                    image_features = model.encode_image_for_retrieval(image)
            image_ids.append(pid.view(-1))
            image_batches.append({key: value.detach().cpu() for key, value in image_features.items()})

        text_ids = torch.cat(text_ids, 0).cpu()
        image_ids = torch.cat(image_ids, 0).cpu()
        text_features = self._concat_feature_batches(text_batches)
        image_features = self._concat_feature_batches(image_batches)

        text_features = {
            key: value.to(device) if isinstance(value, torch.Tensor) else {sub_key: sub_value.to(device) for sub_key, sub_value in value.items()}
            for key, value in text_features.items()
        }
        image_features = {key: value.to(device) for key, value in image_features.items()}

        with torch.no_grad():
            with build_autocast_context(self.args, device):
                similarity = model.compute_retrieval_similarity(image_features, text_features).cpu()

        return similarity, text_ids, image_ids

    def eval(self, model):
        similarity, text_ids, image_ids = self._compute_similarity(model)
        metrics = get_metrics(similarity, text_ids, image_ids, 'pas-t2i')

        table = PrettyTable(['task'] + list(self.requested_metrics))
        table.add_row([metrics['task']] + [metrics[metric_name] for metric_name in self.requested_metrics])
        for metric_name in self.requested_metrics:
            table.custom_format[metric_name] = lambda _, value: f'{value:.2f}'

        self.latest_metrics = {
            f'val/pas/{metric_name}': metrics[metric_name]
            for metric_name in self.requested_metrics
        }
        self.latest_metrics['val/top1'] = metrics['R1']

        self.logger.info('\n' + str(table))
        self.logger.info('\nbest R1 = ' + str(metrics['R1']))
        return metrics['R1']
