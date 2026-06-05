import os
import sys
import csv
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import scipy
from sklearn import metrics
from tqdm import tqdm

from networks.base_model import BaseModel
from networks.dcase2023t2_ae.domain_cae_network import DomainCAENet
from networks.criterion.mahala import (
    cov_v,
    loss_function_mahala,
    calc_inv_cov,
)
from tools.plot_anm_score import AnmScoreFigData
from tools.plot_loss_curve import csv_to_figdata


class DCASE2023T2DomainCAE(BaseModel):
    """
    Thesis model = Domain-Conditioned Autoencoder (PART 1)
                 + Domain/Condition-Aware Mahalanobis scoring (PART 2).

    Training loss : L = recon_MSE + lambda_domain * BCE(domain_logits, domain_label)
                    -- MSE is ONLY the AE training loss, NOT the anomaly score.

    Anomaly score :
        DOMAIN_MAHALA + weighted : score = (1-p_t) * D_source + p_t * D_target
        DOMAIN_MAHALA + hard_min : score = min(D_source, D_target)   (baseline behavior)
        MAHALA                   : score = min(D_source, D_target)   (legacy baseline)
        MSE                      : score = mean MSE                  (sanity only)
    """

    def __init__(self, args, train, test):
        super().__init__(args=args, train=train, test=test)
        parameter_list = [{"params": self.model.parameters()}]
        self.optimizer = optim.Adam(parameter_list, lr=self.args.learning_rate)

        # one score-distribution file per scoring mode so MSE/MAHALA/DOMAIN_MAHALA
        # do NOT clobber each other.
        self.mse_score_distr_file_path = (
            self.model_dir
            / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mse.pickle"
        )
        self.mahala_score_distr_file_path = (
            self.model_dir
            / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mahala.pickle"
        )
        self.domain_mahala_score_distr_file_path = (
            self.model_dir
            / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_domain_mahala.pickle"
        )

        self.lambda_domain = float(getattr(self.args, "lambda_domain", 0.05))
        self.domain_scoring_mode = getattr(self.args, "domain_scoring_mode", "weighted")
        assert self.domain_scoring_mode in ("weighted", "hard_min"), (
            f"Unknown --domain_scoring_mode '{self.domain_scoring_mode}'"
        )

        self.domain_prob_clip = float(getattr(self.args, "domain_prob_clip", 0.05))
        assert 0.0 <= self.domain_prob_clip < 0.5, (
            f"--domain_prob_clip must be in [0, 0.5); got {self.domain_prob_clip}"
        )

        self._print_architecture_summary()

    def _clip_p_target(self, p_target):
        """Clamp predicted target probability into [clip, 1-clip].

        Keeps the weighted DOMAIN_MAHALA score from collapsing to D_source or
        D_target when the classifier becomes over-confident. With clip=0 this
        is a no-op.
        """
        if self.domain_prob_clip <= 0.0:
            return p_target
        lo = self.domain_prob_clip
        hi = 1.0 - self.domain_prob_clip
        return p_target.clamp(min=lo, max=hi)

    # ------------------------------------------------------------------ #
    # Setup helpers                                                      #
    # ------------------------------------------------------------------ #
    def init_model(self):
        self.block_size = self.data.height  # = n_mels
        return DomainCAENet(
            input_dim=self.data.input_dim,
            block_size=self.block_size,
            n_mels=self.args.n_mels,
            frames=self.args.frames,
        )

    def get_log_header(self):
        self.column_heading_list = [
            ["loss"],
            ["val_loss"],
            ["recon_loss"],
            ["bce_loss"],
            ["domain_acc"],
            ["recon_loss_source", "recon_loss_target"],
        ]
        return "loss,val_loss,recon_loss,bce_loss,domain_acc,recon_loss_source,recon_loss_target"

    def _print_architecture_summary(self):
        n_total = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("==================== ARCHITECTURE ====================")
        print(self.model)
        print("------------------------------------------------------")
        print(f"Total parameters     : {n_total:,}")
        print(f"Trainable parameters : {n_trainable:,}")
        print(f"lambda_domain        : {self.lambda_domain}")
        print(f"domain_scoring_mode  : {self.domain_scoring_mode}")
        print(f"domain_prob_clip     : {self.domain_prob_clip}  "
              f"(p_target clamped to [{self.domain_prob_clip}, {1.0 - self.domain_prob_clip}])")
        print("======================================================")

    # ------------------------------------------------------------------ #
    # Loss utilities                                                     #
    # ------------------------------------------------------------------ #
    def loss_reduction_1d(self, score):
        return torch.mean(score, dim=1)

    def loss_reduction(self, score, n_loss):
        if n_loss == 0:
            return torch.tensor(0.0, device=score.device if torch.is_tensor(score) else self.device)
        return torch.sum(score) / n_loss

    def loss_fn(self, recon_x, x):
        # element-wise MSE; per-sample reduction is handled by callers
        return F.mse_loss(recon_x, x.view(recon_x.shape), reduction="none")

    @staticmethod
    def _domain_labels_from_basenames(basenames, device):
        is_target = [1.0 if "target" in name else 0.0 for name in basenames]
        return torch.tensor(is_target, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Train                                                              #
    # ------------------------------------------------------------------ #
    def train(self, epoch):
        if epoch <= self.epoch:
            return
        torch.autograd.set_detect_anomaly(True)

        train_loss = 0.0
        train_recon_loss = 0.0
        train_bce_loss = 0.0
        train_recon_loss_source = 0.0
        train_recon_loss_target = 0.0
        domain_correct = 0
        domain_total = 0
        y_pred = []

        train_loader = self.train_loader

        if epoch == self.args.epochs + 1:
            print("\n============== CALCULATE COVARIANCE ==============")
            is_calc_cov = True
            self.model.eval()
            torch.set_grad_enabled(False)
            cov_x_source = torch.zeros(self.block_size, self.block_size, device=self.device).float()
            cov_x_target = cov_x_source.clone().detach()
            num_source = 0
            num_target = 0
            epoch = self.args.epochs
        else:
            self.model.train()
            is_calc_cov = False
            torch.set_grad_enabled(True)

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            data = batch[0].to(self.device).float()
            if data.shape[0] <= 1:
                continue
            data_name_list = batch[3]

            is_target_list = ["target" in data_name for data_name in data_name_list]
            is_source_list = [not t for t in is_target_list]
            n_source = is_source_list.count(True)
            n_target = is_target_list.count(True)

            domain_label = self._domain_labels_from_basenames(data_name_list, self.device)

            if not is_calc_cov:
                self.optimizer.zero_grad()

            # During training (when computing loss) we condition the decoder on the
            # TRUE binary domain label. During cov-calc we just use the predicted
            # probability (since self.model.eval() => model.training == False).
            recon_batch, _z, domain_logits = self.model(data, domain_label=domain_label)

            if is_calc_cov:
                score_2d, cov_diff_source, cov_diff_target = loss_function_mahala(
                    recon_x=recon_batch,
                    x=data,
                    block_size=self.block_size,
                    update_cov=True,
                    reduction=False,
                    is_source_list=is_source_list,
                    is_target_list=is_target_list,
                )
                cov_x_source_batch = cov_v(diff=cov_diff_source, num=1)
                cov_x_source += cov_x_source_batch.clone().detach()
                num_source += n_source
                if n_target > 0:
                    cov_x_target_batch = cov_v(diff=cov_diff_target, num=1)
                    cov_x_target += cov_x_target_batch.clone().detach()
                    num_target += n_target
            else:
                score_2d = self.loss_fn(recon_batch, data)

            n_loss = len(score_2d)
            score = self.loss_reduction_1d(score=score_2d)

            recon_loss = self.loss_reduction(score=score, n_loss=n_loss)
            recon_loss_source = self.loss_reduction(score=score[is_source_list], n_loss=n_source)
            if n_target > 0:
                recon_loss_target = self.loss_reduction(score=score[is_target_list], n_loss=n_target)
            else:
                recon_loss_target = torch.tensor(0.0, device=self.device)

            # Domain BCE (always computed for monitoring; only applied to grad when training)
            bce_loss = F.binary_cross_entropy_with_logits(
                domain_logits.view(-1), domain_label.view(-1)
            )

            # Domain accuracy
            with torch.no_grad():
                preds = (torch.sigmoid(domain_logits.view(-1)) >= 0.5).float()
                domain_correct += int((preds == domain_label.view(-1)).sum().item())
                domain_total += int(domain_label.numel())

            if is_calc_cov:
                # cov-only pass, no backward
                self.loss = recon_loss
            else:
                self.loss = recon_loss + self.lambda_domain * bce_loss
                self.loss.backward()
                self.optimizer.step()

            train_loss += float(self.loss)
            train_recon_loss += float(recon_loss)
            train_bce_loss += float(bce_loss)
            train_recon_loss_source += float(recon_loss_source)
            train_recon_loss_target += float(recon_loss_target)

            y_pred.append(float(self.loss.item()))

            if (batch_idx % self.args.log_interval == 0) and not is_calc_cov:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Recon: {:.6f}  BCE: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        float(self.loss.item()),
                        float(recon_loss.item()),
                        float(bce_loss.item()),
                    )
                )

        # ----- end of training epoch / cov pass -----
        if is_calc_cov:
            cov_x_source /= max(num_source - 1, 1)
            if num_target == 0:
                cov_x_target = cov_x_source.clone().detach()
            else:
                cov_x_target /= max(num_target - 1, 1)
            self.model.cov_source.data = cov_x_source
            self.model.cov_target.data = cov_x_target

            inv_cov_source, inv_cov_target = calc_inv_cov(model=self.model, device=self.device)

            # fit both MAHALA (hard-min) and DOMAIN_MAHALA score distributions
            # using validation/train normal data only (no test labels are ever touched).
            y_pred_mahala = []
            y_pred_domain_mahala = []
            for _, batch in enumerate(tqdm(train_loader, desc="cov-fit-train")):
                y_pred_mahala, y_pred_domain_mahala = self._collect_mahala_scores(
                    batch=batch,
                    y_pred_mahala=y_pred_mahala,
                    y_pred_domain_mahala=y_pred_domain_mahala,
                    inv_cov_source=inv_cov_source,
                    inv_cov_target=inv_cov_target,
                )
            for _, batch in enumerate(tqdm(self.valid_loader, desc="cov-fit-valid")):
                y_pred_mahala, y_pred_domain_mahala = self._collect_mahala_scores(
                    batch=batch,
                    y_pred_mahala=y_pred_mahala,
                    y_pred_domain_mahala=y_pred_domain_mahala,
                    inv_cov_source=inv_cov_source,
                    inv_cov_target=inv_cov_target,
                )
            self.fit_anomaly_score_distribution(
                y_pred=y_pred_mahala,
                score_distr_file_path=self.mahala_score_distr_file_path,
            )
            self.fit_anomaly_score_distribution(
                y_pred=y_pred_domain_mahala,
                score_distr_file_path=self.domain_mahala_score_distr_file_path,
            )

        # ---- validation ----
        val_loss = 0.0
        val_domain_correct = 0
        val_domain_total = 0
        with torch.no_grad():
            self.model.eval()
            for _, batch in enumerate(self.valid_loader):
                data = batch[0].to(self.device).float()
                data_name_list = batch[3]
                domain_label = self._domain_labels_from_basenames(data_name_list, self.device)

                # At validation/inference we never feed the true label to the decoder.
                recon_batch, _z, domain_logits = self.model(data, domain_label=None)
                score = self.loss_fn(recon_batch, data)
                loss = score.mean()
                val_loss += float(loss)

                preds = (torch.sigmoid(domain_logits.view(-1)) >= 0.5).float()
                val_domain_correct += int((preds == domain_label.view(-1)).sum().item())
                val_domain_total += int(domain_label.numel())

                y_pred.append(float(loss.item()))

        train_domain_acc = (domain_correct / domain_total) if domain_total > 0 else 0.0
        val_domain_acc = (val_domain_correct / val_domain_total) if val_domain_total > 0 else 0.0

        if not is_calc_cov:
            print(
                "====> Epoch: {} Avg loss: {:.4f}  Recon: {:.4f}  BCE: {:.4f}  "
                "DomainAcc(train): {:.3f}  DomainAcc(val): {:.3f}  ValLoss: {:.4f}".format(
                    epoch,
                    train_loss / max(len(train_loader), 1),
                    train_recon_loss / max(len(train_loader), 1),
                    train_bce_loss / max(len(train_loader), 1),
                    train_domain_acc,
                    val_domain_acc,
                    val_loss / max(len(self.valid_loader), 1),
                )
            )
            with open(self.log_path, "a") as log:
                np.savetxt(
                    log,
                    [
                        "{0},{1},{2},{3},{4},{5},{6}".format(
                            train_loss / max(len(train_loader), 1),
                            val_loss / max(len(self.valid_loader), 1),
                            train_recon_loss / max(len(train_loader), 1),
                            train_bce_loss / max(len(train_loader), 1),
                            train_domain_acc,
                            train_recon_loss_source / max(len(train_loader), 1),
                            train_recon_loss_target / max(len(train_loader), 1),
                        )
                    ],
                    fmt="%s",
                )
            try:
                csv_to_figdata(
                    file_path=self.log_path,
                    column_heading_list=self.column_heading_list,
                    ylabel="loss",
                    fig_count=len(self.column_heading_list),
                    cut_first_epoch=True,
                )
            except Exception as e:
                print(f"[Domain-CAE] log plot skipped: {e}")

            # Fit MSE score distribution (kept for compatibility; not used as the
            # thesis anomaly score, only available via --score MSE for sanity).
            self.fit_anomaly_score_distribution(
                y_pred=y_pred,
                score_distr_file_path=self.mse_score_distr_file_path,
            )

        torch.save(self.model.state_dict(), self.model_path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            self.checkpoint_path,
        )

    # ------------------------------------------------------------------ #
    # Per-batch Mahalanobis distance helpers                             #
    # ------------------------------------------------------------------ #
    def _per_sample_mahalanobis_means(self, recon_data, data, inv_cov):
        """
        Returns one Mahalanobis distance per *input vector* (per row of the
        original 640-D feature batch), averaged over the `frames` blocks.

        recon_data/data: shape (B, 640) where 640 = n_mels * frames.
        We view as (B*frames, n_mels), compute the diagonal of
        delta @ inv_cov @ delta.T, then average the `frames` blocks back to
        get a (B,) per-vector distance that aligns with p_target shape.
        """
        u = recon_data.view(-1, self.block_size)         # (B*frames, n_mels)
        v = data.view(-1, self.block_size)               # (B*frames, n_mels)
        delta = u - v
        m_per_block = (delta @ inv_cov * delta).sum(dim=1)  # (B*frames,)
        # group back to (B, frames) and average → (B,)
        m_per_vec = m_per_block.view(-1, self.args.frames).mean(dim=1)
        # divide by block_size to keep scale comparable to baseline averaging
        return m_per_vec / self.block_size

    def _collect_mahala_scores(
        self,
        batch,
        y_pred_mahala,
        y_pred_domain_mahala,
        inv_cov_source,
        inv_cov_target,
    ):
        data = batch[0].to(self.device).float()
        data_name_list = batch[3]
        recon_data, _z, domain_logits = self.model(data, domain_label=None)
        p_target = torch.sigmoid(domain_logits.view(-1))

        d_source = self._per_sample_mahalanobis_means(recon_data, data, inv_cov_source)
        d_target = self._per_sample_mahalanobis_means(recon_data, data, inv_cov_target)

        # Legacy MAHALA: file-level hard-min over the whole batch (matches baseline
        # which feeds one file's vectors per batch during test; during fit we use
        # the train/valid loaders so we keep per-batch aggregation consistent).
        d_source_file = d_source.mean()
        d_target_file = d_target.mean()
        y_pred_mahala.append(float(min(d_source_file.item(), d_target_file.item())))

        if self.domain_scoring_mode == "hard_min":
            domain_score = torch.min(d_source, d_target).mean()
        else:  # weighted
            p_eff = self._clip_p_target(p_target)
            domain_score = ((1.0 - p_eff) * d_source + p_eff * d_target).mean()
        y_pred_domain_mahala.append(float(domain_score.item()))
        return y_pred_mahala, y_pred_domain_mahala

    # ------------------------------------------------------------------ #
    # Test                                                               #
    # ------------------------------------------------------------------ #
    def test(self):
        anm_score_figdata = AnmScoreFigData()
        mode = self.data.mode

        csv_lines = []
        block_size = self.data.height
        if mode:
            performance_over_all = []
            performance = []

        print("============== MODEL LOAD ==============")
        if not os.path.exists(self.model_path):
            print(f"model not found -> {self.model_path}")
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        if self.args.score == "DOMAIN_MAHALA":
            decision_threshold = self.calc_decision_threshold(
                score_distr_file_path=self.domain_mahala_score_distr_file_path
            )
        elif self.args.score == "MAHALA":
            decision_threshold = self.calc_decision_threshold(
                score_distr_file_path=self.mahala_score_distr_file_path
            )
        else:
            decision_threshold = self.calc_decision_threshold(
                score_distr_file_path=self.mse_score_distr_file_path
            )

        dir_name = "test"
        inv_cov_source, inv_cov_target = calc_inv_cov(model=self.model, device=self.device)

        for idx, test_loader_tmp in enumerate(self.test_loader):
            section_name = f"section_{self.data.section_id_list[idx]}"
            result_dir = self.result_dir if self.args.dev else self.eval_data_result_dir

            anomaly_score_csv = (
                result_dir
                / f"anomaly_score_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            )
            anomaly_score_list = []

            decision_result_csv = (
                result_dir
                / f"decision_result_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            )
            decision_result_list = []

            domain_list = [] if mode else None

            print("\n============== BEGIN TEST FOR A SECTION ==============")
            y_pred = []
            y_true = []
            test_loader = test_loader_tmp

            with torch.no_grad():
                y_pred, anomaly_score_list, decision_result_list, domain_list = self.eval(
                    test_loader=test_loader,
                    y_pred=y_pred,
                    anomaly_score_list=anomaly_score_list,
                    decision_result_list=decision_result_list,
                    domain_list=domain_list,
                    y_true=y_true,
                    decision_threshold=decision_threshold,
                    mode=mode,
                    inv_cov_source=inv_cov_source,
                    inv_cov_target=inv_cov_target,
                )

            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            print(f"anomaly score result ->  {anomaly_score_csv}")
            save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
            print(f"decision result ->  {decision_result_csv}")

            if mode:
                y_true_s_auc = [y_true[i] for i in range(len(y_true)) if domain_list[i] == "source" or y_true[i] == 1]
                y_pred_s_auc = [y_pred[i] for i in range(len(y_true)) if domain_list[i] == "source" or y_true[i] == 1]
                y_true_t_auc = [y_true[i] for i in range(len(y_true)) if domain_list[i] == "target" or y_true[i] == 1]
                y_pred_t_auc = [y_pred[i] for i in range(len(y_true)) if domain_list[i] == "target" or y_true[i] == 1]

                y_true_s = [y_true[i] for i in range(len(y_true)) if domain_list[i] == "source"]
                y_pred_s = [y_pred[i] for i in range(len(y_true)) if domain_list[i] == "source"]
                y_true_t = [y_true[i] for i in range(len(y_true)) if domain_list[i] == "target"]
                y_pred_t = [y_pred[i] for i in range(len(y_true)) if domain_list[i] == "target"]

                auc_s = metrics.roc_auc_score(y_true_s_auc, y_pred_s_auc)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
                p_auc_s = metrics.roc_auc_score(y_true_s, y_pred_s, max_fpr=self.args.max_fpr)
                tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(
                    y_true_s, [1 if x > decision_threshold else 0 for x in y_pred_s]
                ).ravel()
                prec_s = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
                recall_s = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
                f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, sys.float_info.epsilon)

                anm_score_figdata.append_figdata(
                    anm_score_figdata.anm_score_to_figdata(
                        scores=[[t, p] for t, p in zip(y_true_s, y_pred_s)],
                        title=f"{section_name}_source_AUC{auc_s}",
                    )
                )

                print("AUC (source) : {}".format(auc_s))
                print("pAUC : {}".format(p_auc))
                print("pAUC (source) : {}".format(p_auc_s))
                print("precision (source) : {}".format(prec_s))
                print("recall (source) : {}".format(recall_s))
                print("F1 score (source) : {}".format(f1_s))

                if len(y_true_t) > 0:
                    auc_t = metrics.roc_auc_score(y_true_t_auc, y_pred_t_auc)
                    p_auc_t = metrics.roc_auc_score(y_true_t, y_pred_t, max_fpr=self.args.max_fpr)
                    tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(
                        y_true_t, [1 if x > decision_threshold else 0 for x in y_pred_t]
                    ).ravel()
                    prec_t = tp_t / np.maximum(tp_t + fp_t, sys.float_info.epsilon)
                    recall_t = tp_t / np.maximum(tp_t + fn_t, sys.float_info.epsilon)
                    f1_t = 2.0 * prec_t * recall_t / np.maximum(prec_t + recall_t, sys.float_info.epsilon)
                    if len(csv_lines) == 0:
                        csv_lines.append(self.result_column_dict["source_target"])
                    csv_lines.append(
                        [
                            section_name.split("_", 1)[1],
                            auc_s, auc_t, p_auc, p_auc_s, p_auc_t,
                            prec_s, prec_t, recall_s, recall_t, f1_s, f1_t,
                        ]
                    )
                    performance.append([auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
                    performance_over_all.append([auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
                    anm_score_figdata.append_figdata(
                        anm_score_figdata.anm_score_to_figdata(
                            scores=[[t, p] for t, p in zip(y_true_t, y_pred_t)],
                            title=f"{section_name}_target_AUC{auc_t}",
                        )
                    )
                    print("AUC (target) : {}".format(auc_t))
                    print("pAUC (target) : {}".format(p_auc_t))
                    print("precision (target) : {}".format(prec_t))
                    print("recall (target) : {}".format(recall_t))
                    print("F1 score (target) : {}".format(f1_t))
                else:
                    if len(csv_lines) == 0:
                        csv_lines.append(self.result_column_dict["single_domain"])
                    csv_lines.append([section_name.split("_", 1)[1], auc_s, p_auc, prec_s, recall_s, f1_s])
                    performance.append([auc_s, p_auc, prec_s, recall_s, f1_s])
                    performance_over_all.append([auc_s, p_auc, prec_s, recall_s, f1_s])

            print("\n============ END OF TEST FOR A SECTION ============")

        if mode:
            amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["arithmetic mean"] + list(amean_performance))
            hmean_performance = scipy.stats.hmean(
                np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon), axis=0
            )
            csv_lines.append(["harmonic mean"] + list(hmean_performance))
            csv_lines.append([])

            anm_score_figdata.show_fig(
                title=self.args.model + "_" + self.args.dataset + self.model_name_suffix + self.eval_suffix + "_anm_score",
                export_dir=result_dir,
            )
        else:
            return

        result_path = (
            result_dir
            / f"result_{self.args.dataset}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}_roc.csv"
        )
        print(f"results -> {result_path}")
        save_csv(save_file_path=result_path, save_data=csv_lines)

    def eval(
        self,
        test_loader,
        y_pred,
        anomaly_score_list,
        decision_result_list,
        domain_list,
        y_true,
        decision_threshold,
        mode,
        inv_cov_source,
        inv_cov_target,
    ):
        # When using DOMAIN_MAHALA we emit a richer CSV with per-file debug stats.
        # Prepend a header row exactly once so analysis tools know the schema while
        # MAHALA/MSE outputs remain identical to the baseline (basename, score).
        write_domain_csv = (self.args.score == "DOMAIN_MAHALA")
        if write_domain_csv and len(anomaly_score_list) == 0:
            anomaly_score_list.append(
                ["basename", "final_score", "p_target_mean", "D_source_mean", "D_target_mean"]
            )

        for j, batch in enumerate(test_loader):
            data = batch[0].to(self.device).float()
            y_true.append(batch[1][0].item())
            basename = batch[3][0]

            recon_data, _z, domain_logits = self.model(data, domain_label=None)
            p_target = torch.sigmoid(domain_logits.view(-1))

            if self.args.score == "DOMAIN_MAHALA":
                d_source = self._per_sample_mahalanobis_means(recon_data, data, inv_cov_source)
                d_target = self._per_sample_mahalanobis_means(recon_data, data, inv_cov_target)
                if self.domain_scoring_mode == "hard_min":
                    file_score = torch.min(d_source, d_target).mean()
                else:  # weighted
                    p_eff = self._clip_p_target(p_target)
                    file_score = ((1.0 - p_eff) * d_source + p_eff * d_target).mean()
                y_pred.append(float(file_score.item()))

                # Per-file debug stats. p_target_mean uses the *unclipped*
                # classifier output so the CSV shows how confident the head
                # actually is.
                p_target_mean = float(p_target.mean().item())
                d_source_mean = float(d_source.mean().item())
                d_target_mean = float(d_target.mean().item())
                anomaly_score_list.append([
                    basename, y_pred[-1], p_target_mean, d_source_mean, d_target_mean,
                ])
            elif self.args.score == "MAHALA":
                loss_source, num = loss_function_mahala(
                    recon_x=recon_data, x=data, block_size=self.block_size,
                    cov=inv_cov_source, use_precision=True, reduction=False,
                )
                loss_source = self.loss_reduction(score=self.loss_reduction_1d(loss_source), n_loss=num)
                loss_target, num = loss_function_mahala(
                    recon_x=recon_data, x=data, block_size=self.block_size,
                    cov=inv_cov_target, use_precision=True, reduction=False,
                )
                loss_target = self.loss_reduction(score=self.loss_reduction_1d(loss_target), n_loss=num)
                y_pred.append(min(loss_target.item(), loss_source.item()))
                anomaly_score_list.append([basename, y_pred[-1]])
            else:  # MSE
                y_pred.append(self.loss_fn(recon_x=recon_data, x=data).mean().item())
                anomaly_score_list.append([basename, y_pred[-1]])

            if y_pred[-1] > decision_threshold:
                decision_result_list.append([basename, 1])
            else:
                decision_result_list.append([basename, 0])

            if mode:
                domain_list.append("target" if "target" in basename else "source")
        return y_pred, anomaly_score_list, decision_result_list, domain_list


def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(save_data)
