from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F

def euclidean_dist(x, y):
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist

def cosine_dist(x, y):
	bs1, bs2 = x.size(0), y.size(0)
	frac_up = torch.matmul(x, y.transpose(0, 1))
	frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
	            (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
	cosine = frac_up / frac_down
	return 1-cosine

def _batch_hard(mat_distance, mat_similarity, indice=False):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 0]
	hard_p_indice = positive_indices[:, 0]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n

class ContrastiveLoss(nn.Module): 
	def __init__(self, T, n_pos, config):
		super(ContrastiveLoss, self).__init__()
		self.T = T
		self.n_pos = n_pos
		self.ce = torch.nn.CrossEntropyLoss().cuda()

		self.proj = nn.Sequential(
			nn.Linear(256, 256), 
			nn.BatchNorm1d(256), 
			nn.ReLU(), 
			nn.Linear(256, 128), 
			nn.BatchNorm1d(128), 
		)

		if config == "random_positive": 
			self.forward_fn = self.forward_simclr_with_randompositive
		elif config == "hard_positive": 
			self.forward_fn = self.forward_simclr_with_hardpositive
		elif config == "nearest_positive": 
			self.forward_fn = self.forward_simclr_with_nearestpositive
		else:
			raise NotImplementedError
	
	"""
	def forward_jumpindex(self, emb, label):
		# emb = self.proj(emb)
		cos_dist = cosine_dist(emb, emb)
		cos_dist = (cos_dist-1)*(-1)
		loss = torch.Tensor([0.]).to(emb.device)
		for i in range(self.n_pos):
			tmp_dist = cos_dist[i::self.n_pos, ((i+1)%self.n_pos)::self.n_pos] # not the distane to itself
			tmp_dist /= self.T
			tmp_label = label[i::self.n_pos]
			# print(tmp_dist.shape, tmp_label.shape)
			tmp_loss = self.ce(tmp_dist, tmp_label)
			loss += tmp_loss
		loss /= self.n_pos
		return loss
		
	def forward(self, emb, label):

		# emb = F.normalize(emb)
		emb = self.proj(emb)
		cos_dist = cosine_dist(emb, emb) # (N, N)
		N = cos_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
		# NOTE: the mat_dist is the cosine distance, not similarity

		# get a matrix, with same label as 0, different labels as 1, 
		# and set the hard positive as 1; 
		# then indexing
		rev_mat_sim = 1.0 - mat_sim
		hard_p, _, hard_p_indice, _ = _batch_hard(cos_dist, mat_sim, indice=True)
		rev_mat_sim[torch.arange(N), hard_p_indice] = 1.0
		# except hard positive, and all negative; all other entries are set with 0
		
		# print(hard_p_indice[:100])
		hard_p_indice_ = hard_p_indice // self.n_pos
		
		col_indices = rev_mat_sim.nonzero()[:, 1]
		assert col_indices.size(0) == N*(N-self.n_pos+1)
		# print("Check contrastive loss. Column indices shape: {}".format(col_indices.size(0)))
		col_indices = col_indices.reshape(N, N-self.n_pos+1)
		
		cos_dist_hardp = torch.gather(cos_dist, 1, col_indices) # (N, N-self.n_pos+1)
		cos_dist_hardp = (cos_dist_hardp - 1) * (-1)
		# print("*")
		# print((hard_p[:10]-1)*(-1))
		# print(cos_dist[:10, :20])
		# print(cos_dist_hardp[:10, :20])
		cos_dist_hardp /= self.T

		target = hard_p_indice_ * self.n_pos # the new indices for the hard positive in cos_dist_hardp
		# print(target[:100])
		loss = self.ce(cos_dist_hardp, target)
		return loss
	"""

	def forward(self, emb, label, ref_pts=None):
		return self.forward_fn(emb, label, ref_pts)

	def forward_simclr_with_nearestpositive(self, emb, label, ref_pts): 
		"""
			emb: [N, emb_dim].
			label: [N, ]. 
			ref_pts: [N, 3].
		"""
		# select the positive pair by nearest distance

		emb = self.proj(emb)
		cos_sim = 1. - cosine_dist(emb, emb) # (N, N)
		N = cos_sim.size(0)
		label = label.expand(N, N).eq(label.expand(N, N).t()).float() # (N, N)

		eucli_dist = euclidean_dist(ref_pts, ref_pts) # (N, N)
		# print("Check euclidean dist: {}.".format(eucli_dist.shape))
		
		mask = torch.eye(label.shape[0], dtype=torch.bool) # (N, N)
		
		label = label[~mask].view(label.shape[0], -1) # (N, N-1)
		cos_sim = cos_sim[~mask].view(cos_sim.shape[0], -1) # (N, N-1)
		eucli_dist = eucli_dist[~mask].view(eucli_dist.shape[0], -1) # (N, N-1)

		all_positives = cos_sim[label.bool()].view(label.shape[0], -1) # (N, n_pos-1)
		all_negatives = cos_sim[~label.bool()].view(label.shape[0], -1) # (N, N-n_pos)

		# select the positive with **nearest** euclidean distance
		pos_eucli_dist = eucli_dist[label.bool()].view(label.shape[0], -1) # (N, n_pos-1)
		_, idx = torch.sort(pos_eucli_dist, dim=1) # from small to large sort
		all_positives = all_positives[torch.arange(N), idx[:, 0]] # (N, )
		# print("Check shape in nearest positive. all_positives: {}.".format(all_positives.shape))

		
		logits = torch.cat([all_positives[:, None], all_negatives], dim=1) # (N, 1+N-n_pos)
		labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

		logits = logits / self.T
		return self.ce(logits, labels)
	
	def forward_simclr_with_hardpositive(self, emb, label, ref_pts): 
		# hard positive, select the largest distance positive pair

		emb = self.proj(emb)
		cos_sim = 1. - cosine_dist(emb, emb) # (N, N)
		N = cos_sim.size(0)
		label = label.expand(N, N).eq(label.expand(N, N).t()).float() # (N, N)
		
		mask = torch.eye(label.shape[0], dtype=torch.bool) # (N, N)
		
		label = label[~mask].view(label.shape[0], -1) # (N, N-1)
		cos_sim = cos_sim[~mask].view(cos_sim.shape[0], -1) # (N, N-1)

		all_positives = cos_sim[label.bool()].view(label.shape[0], -1) # (N, n_pos-1)
		all_negatives = cos_sim[~label.bool()].view(label.shape[0], -1) # (N, N-n_pos)

		# select the positive with **smallest** cosine similarity
		all_positives, _ = torch.sort(all_positives, dim=1)
		all_positives = all_positives[:, 0] # (N, )
		
		logits = torch.cat([all_positives[:, None], all_negatives], dim=1) # (N, 1+N-n_pos)
		labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

		logits = logits / self.T
		return self.ce(logits, labels)

	def forward_simclr_with_randompositive(self, emb, label, ref_pts):
		# randomly select a positive

		emb = self.proj(emb)
		cos_sim = 1. - cosine_dist(emb, emb) # (N, N)
		N = cos_sim.size(0)
		label = label.expand(N, N).eq(label.expand(N, N).t()).float() # (N, N)
		
		mask = torch.eye(label.shape[0], dtype=torch.bool) # (N, N)
		
		label = label[~mask].view(label.shape[0], -1) # (N, N-1)
		cos_sim = cos_sim[~mask].view(cos_sim.shape[0], -1) # (N, N-1)

		all_positives = cos_sim[label.bool()].view(label.shape[0], -1) # (N, n_pos-1)
		all_negatives = cos_sim[~label.bool()].view(label.shape[0], -1) # (N, N-n_pos)

		# print("Check shape. label: {}. cos_sim: {}. all_pos: {}. all_neg: {}".format(label.shape, cos_sim.shape, all_positives.shape, all_negatives.shape))
		
		# randomly select a positive
		rand_idx = torch.randint(0, all_positives.shape[1], (N, ))
		all_positives = all_positives[torch.arange(N), rand_idx] # (N, )
		# print(all_positives.shape)
		
		logits = torch.cat([all_positives[:, None], all_negatives], dim=1) # (N, 1+N-n_pos)
		labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

		logits = logits / self.T
		return self.ce(logits, labels)
		
class TripletLoss(nn.Module):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''
	def __init__(self, margin, normalize_feature=False):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

	def forward(self, emb, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb = F.normalize(emb)
		# mat_dist = euclidean_dist(emb, emb)
		mat_dist = cosine_dist(emb, emb)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
		assert dist_an.size(0)==dist_ap.size(0)
		y = torch.ones_like(dist_ap)
		loss = self.margin_loss(dist_an, dist_ap, y)
		prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		return loss, prec