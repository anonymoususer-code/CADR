import os
import time
import numpy as np
import torch
import arguments
import shutil 
from lib import image_caption, utils
from transformers import BertTokenizer
import logging
import tensorboard_logger as tb_logger
from lib import evaluation
from lib.vse import VSEModel, create_optimizer
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_attn_scores
from torch.nn.utils import clip_grad_norm_


def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):

    filepath = os.path.join(prefix, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(prefix, 'model_best.pth'))
def adjust_learning_rate(opt, optimizer, epoch):
    decay_rate = opt.decay_rate
    lr_schedules = opt.lr_schedules
    if epoch in lr_schedules:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate
            logger = logging.getLogger(__name__)
            logger.info(f"Decayed learning rate to {param_group['lr']} at epoch {epoch}")
def main():
    # Parse options
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    opt.model_name = opt.logger_name
    # Setup GPU
    if opt.multi_gpu:
        utils.init_distributed_mode(opt)
    else:
        torch.cuda.set_device(opt.gpu_id)
    if utils.is_main_process():
        if not os.path.exists(opt.model_name):
            os.makedirs(opt.model_name)
        logging.basicConfig(filename=os.path.join(opt.logger_name, 'train.log'), 
                            filemode='a', format='%(asctime)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    if utils.is_main_process():
        logger.info(opt)  
        arguments.save_parameters(opt, opt.logger_name)
        tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('data/bert-base-uncased')
    # Load data loaders
    train_loader = image_caption.get_train_loader(opt, opt.data_path, tokenizer, opt.batch_size, opt.workers, 'train')
    test_loader = image_caption.get_test_loader(opt, opt.data_path, tokenizer, opt.batch_size, opt.workers, 'test')
    # Initialize model
    model = VSEModel(opt).cuda()
    optimizer = create_optimizer(opt, model)
    best_rsum = 0
    start_epoch = 0

    # Load checkpoint if exists
    checkpoint_path = os.path.join(opt.model_name, 'checkpoint.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        best_rsum = checkpoint['best_rsum']
        
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")

    if opt.multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], output_device=opt.gpu, find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    for epoch in range(start_epoch, opt.num_epochs):
        if opt.multi_gpu:
            train_loader.sampler.set_epoch(epoch)

        adjust_learning_rate(opt, optimizer, epoch)
        train(opt, train_loader, model, model_without_ddp, optimizer, epoch)
        rsum = validate(opt, test_loader, model_without_ddp)

        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)

        if utils.is_main_process():
            logger.info(f"Epoch: [{epoch}], Best rsum: {best_rsum:.1f}")
            save_checkpoint({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_rsum': best_rsum,
                'Eiters': model_without_ddp.Eiters,
            }, is_best, filename='checkpoint.pth', prefix=opt.model_name)

        if opt.multi_gpu:
            torch.distributed.barrier()
            torch.cuda.empty_cache()

    # start eval
    if utils.is_main_process() and opt.eval:
        print('Evaluate the model now.')

        base = opt.logger_name
        logging.basicConfig(filename=os.path.join(base, 'eval.log'), filemode='w', 
                            format='%(asctime)s %(message)s', level=logging.INFO, force=True)

        logger = logging.getLogger()
        logger.info('Evaluating {}...'.format(base))

        model_path = os.path.join(base, 'model_best.pth')
        
        # Save the final results for computing ensemble results
        save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))

        if opt.dataset == 'coco':
            # Evaluate COCO 5-fold 1K
            evaluation.evalrank(model_path, model=model_without_ddp, split='testall', fold5=True)

            # Evaluate COCO 5K
            evaluation.evalrank(model_path, model=model_without_ddp, split='testall', fold5=False, save_path=save_path)

            if opt.evaluate_cxc:
                # Evaluate COCO-trained models on CxC
                evaluation.evalrank(model_path, model=model_without_ddp, split='testall', fold5=True, cxc=True)

        else:
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, model=model_without_ddp, split='test', fold5=False, save_path=save_path)

        logger.info('Evaluation finish!')    

def train(opt, train_loader, model, model_without_ddp, optimizer, epoch):

    # switch to train mode
    model.train()   

    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    if utils.is_main_process() and epoch == 0:
        logger.info('image encoder trainable parameters: {}M'.format(count_params(model_without_ddp.img_enc)))
        logger.info('txt encoder trainable parameters: {}M'.format(count_params(model_without_ddp.txt_enc)))
        logger.info('criterion trainable parameters: {}M'.format(count_params(model_without_ddp.criterion)))

    n_batch = len(train_loader) 

    end = time.time()

    for i, train_data in enumerate(train_loader):  
        
        optimizer.zero_grad()
        # warmup_alpha is [0, 1], loss = loss * warmup_alpha
        warmup_alpha = float(i) / n_batch if epoch == opt.embedding_warmup_epochs else 1. 

        # measure data loading time
        data_time.update(time.time() - end)
        images, captions, lengths, ids, img_ids = train_data
        
        # 将数据移到 GPU 上
        captions = captions.cuda(non_blocking=True)
        lengths = lengths.cuda(non_blocking=True)
        ids = ids.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        img_ids = img_ids.cuda(non_blocking=True) 
        
        # Convert captions from tensor to list of strings
        #captions_list = [tokenizer.decode(caption, skip_special_tokens=True) for caption in captions]
        
        loss = model(images, captions, lengths, img_ids=img_ids, warmup_alpha=warmup_alpha)
       
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.zeros([], requires_grad=True, device=images.device)
        loss.backward()
        if opt.grad_clip > 0:
            clip_grad_norm_(model.parameters(), opt.grad_clip)

        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()    
        model_without_ddp.logger = train_logger
        model_without_ddp.logger.update('Iter', model_without_ddp.Eiters)
        model_without_ddp.logger.update('lr', optimizer.param_groups[0]['lr'])   
        model_without_ddp.logger.update('Loss', loss.item(), opt.batch_size)
        model_without_ddp.Eiters += 1
        if utils.is_main_process():
  
            if model_without_ddp.Eiters % opt.log_step == 0:  
                if epoch == opt.embedding_warmup_epochs:
                    logging.info('The first epoch for training backbone, warmup alpha for loss is {}'.format(epoch, warmup_alpha))

                logging.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    '{e_log}\t'
                    'Batch-Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                        .format(epoch, i+1, n_batch, batch_time=batch_time, e_log=str(model_without_ddp.logger)))

            # Record logs in tensorboard
            tb_logger.log_value('epoch', epoch, step=model_without_ddp.Eiters)
            tb_logger.log_value('step', i, step=model_without_ddp.Eiters)
            tb_logger.log_value('batch_time', batch_time.val, step=model_without_ddp.Eiters)
            tb_logger.log_value('data_time', data_time.val, step=model_without_ddp.Eiters)

            model_without_ddp.logger.tb_log(tb_logger, step=model_without_ddp.Eiters)

        if i > n_batch:
            break

def validate(opt, val_loader, model):

    logger = logging.getLogger(__name__)
    
    model.eval()

    with torch.no_grad():
       img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)

    # have repetitive image features
    img_embs = img_embs[::5]

    start_time = time.time()

    if opt.multi_gpu:         
        sims = torch.zeros((len(img_embs), len(cap_embs))).cuda()
        
        num_tasks = utils.get_world_size()
        rank = utils.get_rank() 

        step = img_embs.size(0) // num_tasks + 1
        start = rank * step
        end = min(img_embs.size(0), start + step)

        sims_part = shard_attn_scores(model, img_embs[start:end], cap_embs, cap_lens, opt, gpu=True)
        sims[start:end] = sims_part

        # wait for synchronization 
        torch.distributed.barrier()
        # Aggregating results on different GPUs
        torch.distributed.all_reduce(sims, op=torch.distributed.ReduceOp.SUM) 
        sims = sims.cpu().numpy()
    else:
        sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt)      
        sims = sims.numpy()

    # compute metric
    if utils.is_main_process():
        
        logging.info("calculate similarity time: %.3f" % float(time.time() - start_time))

        npts = img_embs.shape[0]
        
        # caption retrieval
        (r1, r5, r10, medr, meanr) = i2t(npts, sims)
        logging.info("Image to text (R@1, R@5, R@10): %.1f, %.1f, %.1f" % (r1, r5, r10))

        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
        logging.info("Text to image (R@1, R@5, R@10): %.1f, %.1f, %.1f" % (r1i, r5i, r10i))

        # sum of recalls to be used for early stopping
        currscore = r1 + r5 + r10 + r1i + r5i + r10i
        logger.info('Current rsum is {}'.format(round(currscore, 1)))
            
        # record metrics in tensorboard
        tb_logger.log_value('r1', r1, step=model.Eiters)
        tb_logger.log_value('r5', r5, step=model.Eiters)
        tb_logger.log_value('r10', r10, step=model.Eiters)
        tb_logger.log_value('medr', medr, step=model.Eiters)
        tb_logger.log_value('meanr', meanr, step=model.Eiters)
        tb_logger.log_value('r1i', r1i, step=model.Eiters)
        tb_logger.log_value('r5i', r5i, step=model.Eiters)
        tb_logger.log_value('r10i', r10i, step=model.Eiters)
        tb_logger.log_value('medri', medri, step=model.Eiters)
        tb_logger.log_value('meanr', meanr, step=model.Eiters)
        tb_logger.log_value('rsum', currscore, step=model.Eiters)                     

        return currscore

def count_params(model):

    # The unit is M (million)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    params = round(params/(1024**2), 2)

    return params


if __name__ == '__main__':
    
    main()