# pip install ffmpeg-python ftfy regex
vid='somevideo/Bhxk-O1Y7Ho.mp4'
import time
start=time.time()
import torch

from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from utils.tensor_utils import pad_sequences_1d
from moment_detr.span_utils import span_cxw_to_xx
from utils.basic_utils import l2_normalize_np_array
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available()else "cpu") 

class MomentDETRPredictor:
    def __init__(self, ckpt_path, clip_model_name_or_path="ViT-B/32", device=device):
        self.clip_len = 2  # seconds
        self.device = device
        print("Loading feature extractors...")
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=True,
            model_name_or_path=clip_model_name_or_path, device=device
        )
        print("Loading trained Moment-DETR model...")
        self.model = build_inference_model(ckpt_path).to(self.device)

    @torch.no_grad() # =========就是forwrad函数.
    def localize_moment(self, video_path, query_list):
        """
        Args:
            video_path: str, path to the video file
            query_list: List[str], each str is a query for this video
        """
        # construct model inputs
        n_query = len(query_list)
        video_feats = self.feature_extractor.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)
        # add tef # 添加时间轴编码.
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames # 时间轴开始
        tef_ed = tef_st + 1.0 / n_frames # 时间轴结尾
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)  # (n_frames, 2) # 叠加为2维矩阵.
        video_feats = torch.cat([video_feats, tef], dim=1) #信息全cat上.
        if 0:
            assert n_frames <= 75, "The positional embedding of this pretrained MomentDETR only support video up " \
                                "to 150 secs (i.e., 75 2-sec clips) in length" # 视频太长不行.最大就150s.因为这个位置编码方案太大就不准了.
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)  # (#text, T, d)
        video_mask = torch.ones(n_query, n_frames).to(self.device)
        query_feats = self.feature_extractor.encode_text(query_list)  # #text * (L, d)
        query_feats, query_mask = pad_sequences_1d(
            query_feats, dtype=torch.float32, device=self.device, fixed_length=None) # 没有fixed_length,就不固定长度.跟局句子里面词数来.
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        model_inputs = dict(
            src_vid=video_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask
        )

        # decode outputs
        self.model.eval()
        outputs = self.model(**model_inputs)
        # #moment_queries refers to the positional embeddings in MomentDETR's decoder, not the input text query
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #moment_queries=10, #classes=2)
        scores = prob[..., 0]  # * (batch_size, #moment_queries)  foreground label is 0, we directly take it
        pred_spans = outputs["pred_spans"]  # (bsz, #moment_queries, 2)
        _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
        saliency_scores = []
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)): #shuchugaoguangshike
            _score = _saliency_scores[j, :int(valid_vid_lengths[j])].tolist()
            _score = [round(e, 4) for e in _score]
            saliency_scores.append(_score)

        # compose predictions #添加预测文本信息.
        predictions = []
        video_duration = n_frames * self.clip_len
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * video_duration #计算时间区间.
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                query=query_list[idx],  # str
                vid=video_path,
                pred_relevant_windows=cur_ranked_preds,  # List([st(float), ed(float), score(float)])
                pred_saliency_scores=saliency_scores[idx]  # List(float), len==n_frames, scores for each frame
            )
            predictions.append(cur_query_pred)

        return predictions


def run_example():
    # load example data
    from utils.basic_utils import load_jsonl
    video_path = vid
    query_path = "run_on_video/example/queries.jsonl"
    queries = load_jsonl(query_path) # anootaion doc:https://github.com/zhangbo2008/video_hightlight_detc/tree/main/data
    print(queries)
    query_text_list = [e["query"] for e in queries]
    ckpt_path = "run_on_video/moment_detr_ckpt/model_best.ckpt"

    # run predictions
    print("Build models...")
    clip_model_name_or_path = "ViT-B/32"
    # clip_model_name_or_path = "tmp/ViT-B-32.pt"
    moment_detr_predictor = MomentDETRPredictor(
        ckpt_path=ckpt_path,
        clip_model_name_or_path=clip_model_name_or_path,
        device=device
    )
    print("Run prediction...")
    predictions = moment_detr_predictor.localize_moment(
        video_path=video_path, query_list=query_text_list)

    # print data
    for idx, query_data in enumerate(queries):
        print("-"*30 + f"idx{idx}")
        print(f">> query: {query_data['query']}")
        print(f">> video_path: {video_path}")
        print(f">> GT moments: {query_data['relevant_windows']}")
        print(f">> Predicted moments ([start_in_seconds, end_in_seconds, score]): "
              f"{predictions[idx]['pred_relevant_windows']}")
        print(f">> GT saliency scores (only localized 2-sec clips): {query_data['saliency_scores']}")
        print('下面打印高光时刻的分数.')
        print(f">> Predicted saliency scores (for all 2-sec clip): "
              f"{predictions[idx]['pred_saliency_scores']}")


        houchuli=predictions[idx]['pred_saliency_scores']#================进行后处理.pred_saliency_scores	list(float), highlight prediction scores. The higher the better. This list should contain a score for each of the 2-second clip in the videos, and is ordered

        print(1)
        a=len(houchuli)


        print(1)
        import torch
        a=torch.tensor(houchuli)
        asdfasd=torch.argsort(a,descending=True)
        print(asdfasd)




        #========我们要抽取10秒.那就是5个.


        #=====视频全部分10秒块然后sum即可.

        minikuai=5
        maxi=-9999
        maxidex=0
        for i in range(len(houchuli))[:-minikuai+1]:
            print(i,'当前位置')
            tmp=houchuli[i:i+minikuai]
            if sum(tmp)>maxi:
                maxi=sum(tmp)
                maxidex=i
        print(maxi,maxidex)


        #========maxidex 抽取10秒即可.#=当前抽取mp4会bug,因为底层不是python实现的原因, 所以保存为gif. 再用其他软件转gif为mp4或者avi即可.
        if '/' in vid:
            outname=vid.split('/')[-1].split('.')[0]+'.gif'
        else:
            outname=vid.split('.')[0]+'.gif'
        print('输出的名字',outname)
        if 1:
            import ffmpeg
            ffmpeg.input(vid).trim(start=maxidex, duration=minikuai*2).output(outname).overwrite_output().run()
        print('使用时间',time.time()-start) #10分钟视频使用时间 71.93740630149841





        if 0:
            # import ffmpeg
            # in_filename = "somevideo/Bhxk-O1Y7Ho.mp4"
            # out_filename = "out.mp4"
            # prop = ffmpeg.probe(in_filename)
            # duration = float(prop["format"]["duration"])

            # stream = ffmpeg.input(in_filename)
            # v = stream.video.filter("trim", start=(maxidex),duration=10.0)
            # a = stream.audio.filter("atrim", start=(maxidex),duration=10.0)
            # out = ffmpeg.output(v, a, out_filename)
            # ffmpeg.run(out)

            import ffmpeg
            (ffmpeg
                .input(vid)
                .trim(start=165, duration=minikuai*2) #=====这地方底层不是python代码, 所以start里面不能写变量!
                .filter('fps', fps=25, round='up')
                .output('out.mp4')
                .run())

if __name__ == "__main__":
    run_example()
