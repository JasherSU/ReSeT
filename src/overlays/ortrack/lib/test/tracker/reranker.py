import torch
import torch.nn.functional as F


class HeuristicReranker:
    def __init__(self, lambda_cos=0.2, mu_disp=0.1, nu_edge=0.0,
                 ambig_low=0.25, ambig_high=0.45, margin_thresh=0.05, topk=5,
                 lens_z=64):
        self.lambda_cos = lambda_cos
        self.mu_disp = mu_disp
        self.nu_edge = nu_edge
        self.ambig_low = ambig_low
        self.ambig_high = ambig_high
        self.margin_thresh = margin_thresh
        self.topk = topk
        self.lens_z = lens_z

    def is_ambiguous(self, prev_score, top1_score, top2_score):
        if self.ambig_low <= prev_score <= self.ambig_high:
            return True
        if (top1_score - top2_score) < self.margin_thresh:
            return True
        return False

    def compute_template_cosine(self, before_norm, topk_idx, feat_sz):
        if before_norm is None:
            return [0.0] * topk_idx.shape[1]
        template_tok = before_norm[0, :self.lens_z, :].mean(dim=0)
        search_tokens = before_norm[0, self.lens_z:, :]
        cos_sims = []
        for ki in range(topk_idx.shape[1]):
            tok_idx = topk_idx[0, ki].item()
            if tok_idx < search_tokens.shape[0]:
                cos_sim = F.cosine_similarity(
                    template_tok.unsqueeze(0), search_tokens[tok_idx:tok_idx+1], dim=-1)
                cos_sims.append(cos_sim.item())
            else:
                cos_sims.append(0.0)
        return cos_sims

    def compute_displacement(self, candidates_bbox, prev_bbox):
        if prev_bbox is None:
            return [0.0] * len(candidates_bbox)
        prev_cx = prev_bbox[0] + 0.5 * prev_bbox[2]
        prev_cy = prev_bbox[1] + 0.5 * prev_bbox[3]
        prev_diag = (prev_bbox[2] ** 2 + prev_bbox[3] ** 2) ** 0.5
        if prev_diag < 1e-6:
            prev_diag = 1.0
        displacements = []
        for cand in candidates_bbox:
            cx = cand[0] + 0.5 * cand[2]
            cy = cand[1] + 0.5 * cand[3]
            dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
            displacements.append(dist / prev_diag)
        return displacements

    def compute_edge_penalty(self, topk_idx, feat_sz):
        center = (feat_sz - 1) / 2.0
        penalties = []
        for ki in range(topk_idx.shape[1]):
            idx = topk_idx[0, ki].item()
            row = idx // feat_sz
            col = idx % feat_sz
            dist_to_center = ((row - center) ** 2 + (col - center) ** 2) ** 0.5
            max_dist = (2 * center ** 2) ** 0.5
            penalties.append(dist_to_center / max(max_dist, 1.0))
        return penalties

    def rerank(self, topk_scores, topk_idx, candidates_bbox, prev_bbox,
               before_norm, feat_sz):
        raw_scores = topk_scores[0].tolist()
        cos_sims = self.compute_template_cosine(before_norm, topk_idx, feat_sz)
        displacements = self.compute_displacement(candidates_bbox, prev_bbox)
        edge_penalties = self.compute_edge_penalty(topk_idx, feat_sz)
        final_scores = []
        for i in range(len(raw_scores)):
            score = (raw_scores[i]
                     + self.lambda_cos * cos_sims[i]
                     - self.mu_disp * displacements[i]
                     - self.nu_edge * edge_penalties[i])
            final_scores.append(score)
        best_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])
        return best_idx, final_scores
