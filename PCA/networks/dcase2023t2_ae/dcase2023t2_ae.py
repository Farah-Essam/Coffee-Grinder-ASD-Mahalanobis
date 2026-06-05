import os
import sys
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import scipy
from sklearn import metrics
import csv
from tqdm import tqdm

from networks.base_model import BaseModel
from networks.dcase2023t2_ae.network import AENet
from networks.criterion.mahala import cov_v, cov_v_diff, loss_function_mahala, calc_inv_cov
from tools.plot_anm_score import AnmScoreFigData
from tools.plot_loss_curve import csv_to_figdata

class DCASE2023T2AE(BaseModel):
    def __init__(self, args, train, test):
        super().__init__(
            args=args,
            train=train,
            test=test
        )
        parameter_list = [{"params":self.model.parameters()}]
        self.optimizer = optim.Adam(parameter_list, lr=self.args.learning_rate)
        self.mse_score_distr_file_path = self.model_dir/f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mse.pickle"
        self.mahala_score_distr_file_path = self.model_dir/f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mahala.pickle"
        self.mahala_pca_score_distr_file_path = self.model_dir/f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mahala_pca.pickle"
        self.mahala_pca_aux_path = self.model_dir/f"{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mahala_pca.pt"
        self._mahala_pca_mean = None
        self._mahala_pca_W = None
        self._mahala_pca_inv_source = None
        self._mahala_pca_inv_target = None


    def init_model(self):
        self.block_size = self.data.height
        return AENet(input_dim=self.data.input_dim, block_size=self.block_size)

    def get_log_header(self):
        self.column_heading_list=[
                ["loss"],
                ["val_loss"],
                ["recon_loss"], 
                ["recon_loss_source", "recon_loss_target"],
        ]
        return "loss,val_loss,recon_loss,recon_loss_source,recon_loss_target"
    
    def train(self, epoch):
        if epoch <= self.epoch:
            return
        torch.autograd.set_detect_anomaly(True)
        train_loss = 0
        train_recon_loss = 0
        train_recon_loss_source = 0
        train_recon_loss_target = 0
        y_pred = []

        train_loader = self.train_loader

        if epoch == self.args.epochs + 1:
            print("\n============== CALCULATE COVARIANCE ==============")
            is_calc_cov = True
            self.model.eval()
            torch.set_grad_enabled(False)
            # initialize mahalanobis param
            cov_x_source = np.zeros((self.block_size, self.block_size))
            cov_x_source = torch.from_numpy(cov_x_source)
            cov_x_source = cov_x_source.to(self.device).float()
            cov_x_target = cov_x_source.clone().detach()
            num_source = 0
            num_target = 0
            epoch = self.args.epochs
        else:
            self.model.train()
            is_calc_cov = False

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            data = batch[0]
            data = data.to(self.device).float()
            if data.shape[0] <= 1:
                continue
            data_name_list = batch[3]
            machine_id = torch.argmax(batch[2], dim=1).long()
            machine_id = machine_id.to(self.device)

            is_target_list = ["target" in data_name for data_name in data_name_list]
            is_source_list = np.logical_not(is_target_list).tolist()
            n_source = is_source_list.count(True)
            n_target = is_target_list.count(True)

            if not is_calc_cov:
                self.optimizer.zero_grad()
            recon_batch, z = self.model(data)

            if is_calc_cov and self.args.score != "MAHALA_PCA":
                score_2d, cov_diff_source, cov_diff_target = loss_function_mahala(
                    recon_x=recon_batch,
                    x=data,
                    block_size=self.block_size,
                    update_cov=True,
                    reduction=False,
                    is_source_list=is_source_list,
                    is_target_list=is_target_list
                )
                cov_x_source_batch = cov_v(
                    diff=cov_diff_source,
                    num=1
                )
                cov_x_source += cov_x_source_batch.clone().detach()
                num_source += n_source
                if n_target > 0:
                    cov_x_target_batch = cov_v(
                        diff=cov_diff_target,
                        num=1
                    )
                    cov_x_target += cov_x_target_batch.clone().detach()
                    num_target += n_target
            else:
                score_2d = self.loss_fn(
                    recon_batch,
                    data
                )
            
            # loss reduction
            n_loss = len(score_2d)
            score = self.loss_reduction_1d(score=score_2d)

            recon_loss = self.loss_reduction(score=score, n_loss=n_loss)
            recon_loss_source = self.loss_reduction(score=score[is_source_list], n_loss=n_source)
            if n_target > 0:
                recon_loss_target = self.loss_reduction(score=score[is_target_list], n_loss=n_target)
            else:
                recon_loss_target = 0
            
            self.loss = recon_loss
            if not is_calc_cov:
                self.loss.backward()
                self.optimizer.step()
            train_loss += float(self.loss)
            train_recon_loss += float(recon_loss)
            train_recon_loss_source += float(recon_loss_source)
            train_recon_loss_target += float(recon_loss_target)

            
            # calculate y_pred for fitting anomaly score distribution
            y_pred.append(self.loss.item())

            if batch_idx % self.args.log_interval == 0 and not is_calc_cov:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    self.loss.item()))

        if is_calc_cov:
            if self.args.score == "MAHALA_PCA":
                self._run_mahala_pca_covariance_phase(train_loader)
            else:
                # save cov_x (original MAHALA / legacy covariance pass)
                cov_x_source /= num_source - 1
                if num_target == 0:
                    cov_x_target = cov_x_source.clone().detach()
                else:
                    cov_x_target /= num_target - 1
                self.model.cov_source.data = cov_x_source
                self.model.cov_target.data = cov_x_target

                inv_cov_source, inv_cov_target = calc_inv_cov(
                    model=self.model,
                    device=self.device
                )
                y_pred_mahala = []
                for batch_idx, batch in enumerate(tqdm(train_loader)):
                    y_pred_mahala = self.calc_valid_mahala_score(
                        data=batch[0],
                        y_pred=y_pred_mahala,
                        inv_cov_source=inv_cov_source,
                        inv_cov_target=inv_cov_target,
                    )
                for batch_idx, batch in enumerate(self.valid_loader):
                    y_pred_mahala = self.calc_valid_mahala_score(
                        data=batch[0],
                        y_pred=y_pred_mahala,
                        inv_cov_source=inv_cov_source,
                        inv_cov_target=inv_cov_target,
                    )
                self.fit_anomaly_score_distribution(
                    y_pred=y_pred_mahala,
                    score_distr_file_path=self.mahala_score_distr_file_path
                )

        # validation test
        val_loss = 0
        with torch.no_grad():
            self.model.eval()
            for batch_idx, batch in enumerate(self.valid_loader):
                data = batch[0]
                data = data.to(self.device).float()

                recon_batch, _ = self.model(data)
                score = self.loss_fn(
                    recon_batch,
                    data
                )
                loss = score.mean()

                val_loss += float(loss)
                
                # calculate y_pred for fitting anomaly score distribution
                y_pred.append(loss.item())

        if not is_calc_cov:
            # plot train log
            print('====> Epoch: {} Average loss: {:.4f} Validation loss: {:.4f}'.format(
                  epoch, 
                  train_loss / len(train_loader),
                  val_loss / len(self.valid_loader)))
            with open(self.log_path, 'a') as log:
                np.savetxt(log,["{0},{1},{2},{3},{4}".format(
                    train_loss / len(train_loader),
                    val_loss / len(self.valid_loader),
                    train_recon_loss / len(train_loader),
                    train_recon_loss_source / len(train_loader),
                    train_recon_loss_target / len(train_loader),
                    )],fmt="%s")
            csv_to_figdata(
                file_path=self.log_path,
                column_heading_list=self.column_heading_list,
                ylabel="loss",
                fig_count=len(self.column_heading_list),
                cut_first_epoch=True
            )
            self.fit_anomaly_score_distribution(
                y_pred=y_pred,
                score_distr_file_path=self.mse_score_distr_file_path
            )

        # save model
        torch.save(self.model.state_dict(), self.model_path)
        torch.save({
            'epoch':epoch,
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'loss':self.loss
        }, self.checkpoint_path)

    def _fit_pca_numpy(self, residuals_np, k):
        """Fit PCA on training residuals only: center, SVD, top-k directions. W is (D, k)."""
        n, d = residuals_np.shape
        k_eff = min(k, d, max(n - 1, 1))
        if k_eff < k:
            print(f"[MAHALA_PCA] Warning: reducing PCA dim from {k} to {k_eff} (N={n}, D={d}).")
        mean = residuals_np.mean(axis=0)
        xc = residuals_np - mean
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        w = vt[:k_eff].T.astype(np.float32)
        return mean.astype(np.float32), w, k_eff

    def _run_mahala_pca_covariance_phase(self, train_loader):
        print("\n============== MAHALA_PCA: PCA enabled ==============")
        d = self.block_size
        k_req = self.args.mahala_pca_dim
        print(f"[MAHALA_PCA] PCA target dimension (requested): {k_req}")
        print(f"[MAHALA_PCA] Residual shape before PCA (per row): (batch_rows, {d})")

        self.model.eval()
        torch.set_grad_enabled(False)
        parts = []
        with torch.no_grad():
            for batch in tqdm(train_loader):
                data = batch[0].to(self.device).float()
                if data.shape[0] <= 1:
                    continue
                recon_batch, _ = self.model(data)
                diff = data - recon_batch
                parts.append(diff.view(-1, d).cpu().numpy())
        residuals_np = np.concatenate(parts, axis=0)
        print(f"[MAHALA_PCA] Collected training residuals for PCA: N={residuals_np.shape[0]}, D={d} (train_loader only)")

        mean_np, w_np, k = self._fit_pca_numpy(residuals_np, k_req)
        print(f"[MAHALA_PCA] PCA dimension (effective): {k}")
        print(f"[MAHALA_PCA] Residual shape after PCA projection: (batch_rows, {k})")

        pca_mean = torch.from_numpy(mean_np).to(self.device).float()
        pca_w = torch.from_numpy(w_np).to(self.device).float()
        self._mahala_pca_mean = pca_mean
        self._mahala_pca_W = pca_w

        cov_x_source = torch.zeros((k, k), device=self.device, dtype=torch.float32)
        cov_x_target = torch.zeros((k, k), device=self.device, dtype=torch.float32)
        num_source = 0
        num_target = 0

        with torch.no_grad():
            for batch in tqdm(train_loader):
                data = batch[0].to(self.device).float()
                if data.shape[0] <= 1:
                    continue
                data_name_list = batch[3]
                is_target_list = ["target" in data_name for data_name in data_name_list]
                is_source_list = np.logical_not(is_target_list).tolist()
                n_source = is_source_list.count(True)
                n_target = is_target_list.count(True)

                recon_batch, _ = self.model(data)
                diff = data - recon_batch

                diff_s = diff[is_source_list].contiguous().view(-1, d)
                proj_s = torch.matmul(diff_s - pca_mean.view(1, -1), pca_w)
                cov_diff_source, _ = cov_v_diff(in_v=proj_s)
                cov_x_source += cov_v(diff=cov_diff_source, num=1)
                num_source += n_source
                if n_target > 0:
                    diff_t = diff[is_target_list].contiguous().view(-1, d)
                    proj_t = torch.matmul(diff_t - pca_mean.view(1, -1), pca_w)
                    cov_diff_target, _ = cov_v_diff(in_v=proj_t)
                    cov_x_target += cov_v(diff=cov_diff_target, num=1)
                    num_target += n_target

        cov_x_source /= max(num_source - 1, 1)
        if num_target == 0:
            cov_x_target = cov_x_source.clone().detach()
        else:
            cov_x_target /= max(num_target - 1, 1)

        self._mahala_pca_inv_source = torch.inverse(cov_x_source).to(self.device).float()
        self._mahala_pca_inv_target = torch.inverse(cov_x_target).to(self.device).float()

        torch.save(
            {
                "pca_mean": self._mahala_pca_mean.cpu(),
                "pca_W": self._mahala_pca_W.cpu(),
                "inv_cov_source": self._mahala_pca_inv_source.cpu(),
                "inv_cov_target": self._mahala_pca_inv_target.cpu(),
                "mahala_pca_dim": k,
            },
            self.mahala_pca_aux_path,
        )
        print(f"[MAHALA_PCA] Saved PCA + inverse covariances -> {self.mahala_pca_aux_path}")

        inv_cov_source = self._mahala_pca_inv_source
        inv_cov_target = self._mahala_pca_inv_target
        y_pred_mahala = []
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            y_pred_mahala = self.calc_valid_mahala_score(
                data=batch[0],
                y_pred=y_pred_mahala,
                inv_cov_source=inv_cov_source,
                inv_cov_target=inv_cov_target,
            )
        for batch_idx, batch in enumerate(self.valid_loader):
            y_pred_mahala = self.calc_valid_mahala_score(
                data=batch[0],
                y_pred=y_pred_mahala,
                inv_cov_source=inv_cov_source,
                inv_cov_target=inv_cov_target,
            )
        self.fit_anomaly_score_distribution(
            y_pred=y_pred_mahala,
            score_distr_file_path=self.mahala_pca_score_distr_file_path
        )

    def calc_valid_mahala_score(self, data, y_pred, inv_cov_source, inv_cov_target):
        data = data.to(self.device).float()
        recon_data, _ = self.model(data)
        if self.args.score == "MAHALA_PCA":
            loss_source, num = self._mahala_pca_quad_form(recon_data, data, inv_cov_source)
            loss_source = self.loss_reduction(score=self.loss_reduction_1d(loss_source), n_loss=num)

            loss_target, num = self._mahala_pca_quad_form(recon_data, data, inv_cov_target)
            loss_target = self.loss_reduction(score=self.loss_reduction_1d(loss_target), n_loss=num)
        else:
            loss_source, num = loss_function_mahala(
                recon_x=recon_data,
                x=data,
                block_size=self.block_size,
                cov=inv_cov_source,
                use_precision=True,
                reduction=False
            )
            loss_source = self.loss_reduction(score=self.loss_reduction_1d(loss_source), n_loss=num)

            loss_target, num = loss_function_mahala(
                recon_x=recon_data,
                x=data,
                block_size=self.block_size,
                cov=inv_cov_target,
                use_precision=True,
                reduction=False
            )
            loss_target = self.loss_reduction(score=self.loss_reduction_1d(loss_target), n_loss=num)
        y_pred.append(min(loss_target.item(), loss_source.item()))
        return y_pred

    def _mahala_pca_quad_form(self, recon_x, x, inv_cov):
        diff = (x - recon_x).view(-1, self.block_size).float()
        delta = torch.matmul(diff - self._mahala_pca_mean.view(1, -1), self._mahala_pca_W)
        tmp = torch.matmul(delta, inv_cov)
        m_mat = torch.matmul(tmp, delta.t())
        return m_mat, delta.shape[0]

    def _load_mahala_pca_aux_for_eval(self):
        if not self.mahala_pca_aux_path.exists():
            raise FileNotFoundError(
                f"MAHALA_PCA auxiliary state not found: {self.mahala_pca_aux_path}. "
                "Train with --score MAHALA_PCA first."
            )
        try:
            state = torch.load(self.mahala_pca_aux_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(self.mahala_pca_aux_path, map_location="cpu")
        self._mahala_pca_mean = state["pca_mean"].to(self.device).float()
        self._mahala_pca_W = state["pca_W"].to(self.device).float()
        self._mahala_pca_inv_source = state["inv_cov_source"].to(self.device).float()
        self._mahala_pca_inv_target = state["inv_cov_target"].to(self.device).float()
        k = int(state.get("mahala_pca_dim", self._mahala_pca_W.shape[1]))
        print(f"[MAHALA_PCA] Loaded PCA aux from {self.mahala_pca_aux_path} (K={k}).")

    def loss_reduction_1d(self, score):
        return torch.mean(score, dim=1)

    def loss_reduction(self, score, n_loss):
        return torch.sum(score) / n_loss

    def loss_fn(self,recon_x, x):
        ### MSE loss ###
        loss = F.mse_loss(recon_x, x.view(recon_x.shape), reduction="none")
        return loss

    def test(self):
        anm_score_figdata = AnmScoreFigData()
        # check mode
        # "development": mode == True
        # "evaluation": mode == False
        mode = self.data.mode

        # initialize lines in csv for AUC and pAUC
        csv_lines = []

        block_size = self.data.height
        if mode:
            performance_over_all = []
            performance = []

        print("============== MODEL LOAD ==============")
        # load model file
        if not os.path.exists(self.model_path):
            print(f"model not found -> {self.model_path} ")
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        if self.args.score == "MAHALA":
            decision_threshold = self.calc_decision_threshold(score_distr_file_path=self.mahala_score_distr_file_path)
        elif self.args.score == "MAHALA_PCA":
            decision_threshold = self.calc_decision_threshold(score_distr_file_path=self.mahala_pca_score_distr_file_path)
        else:
            decision_threshold = self.calc_decision_threshold(score_distr_file_path=self.mse_score_distr_file_path)
        
        dir_name = "test"
        if self.args.score == "MAHALA_PCA":
            self._load_mahala_pca_aux_for_eval()
            inv_cov_source = self._mahala_pca_inv_source
            inv_cov_target = self._mahala_pca_inv_target
        else:
            inv_cov_source, inv_cov_target = calc_inv_cov(
                model=self.model,
                device=self.device
            )
        for idx, test_loader_tmp in enumerate(self.test_loader):
            section_name = f"section_{self.data.section_id_list[idx]}"
            result_dir = self.result_dir if self.args.dev else self.eval_data_result_dir

            # setup anomaly score file path
            anomaly_score_csv = result_dir/f"anomaly_score_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            anomaly_score_list = []

            # setup decision result file path
            decision_result_csv = result_dir/f"decision_result_{self.args.dataset}_{section_name}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            decision_result_list = []

            domain_list = None
            if mode:
                domain_list = []

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

            # output anomaly scores
            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            print("anomaly score result ->  {}".format(anomaly_score_csv))

            # output decision results
            save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
            print("decision result ->  {}".format(decision_result_csv))

            if mode:
                # extract scores for calculation of AUC (source) and AUC (target)
                y_true_s_auc = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx]=="source" or y_true[idx]==1]
                y_pred_s_auc = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx]=="source" or y_true[idx]==1]
                y_true_t_auc = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx]=="target" or y_true[idx]==1]
                y_pred_t_auc = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx]=="target" or y_true[idx]==1]

                # extract scores for calculation of precision, recall, F1 score for each domain
                y_true_s = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx]=="source"]
                y_pred_s = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx]=="source"]
                y_true_t = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx]=="target"]
                y_pred_t = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx]=="target"]


                # calculate AUC, pAUC, precision, recall, F1 score 
                auc_s = metrics.roc_auc_score(y_true_s_auc, y_pred_s_auc)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
                p_auc_s = metrics.roc_auc_score(y_true_s, y_pred_s, max_fpr=self.args.max_fpr)
                tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(y_true_s, [1 if x > decision_threshold else 0 for x in y_pred_s]).ravel()
                prec_s = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
                recall_s = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
                f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, sys.float_info.epsilon)

                anm_score_figdata.append_figdata(anm_score_figdata.anm_score_to_figdata(
                    scores=[[t, p] for t, p in zip(y_true_s, y_pred_s)],
                    title=f"{section_name}_source_AUC{auc_s}"
                ))

                print("AUC (source) : {}".format(auc_s))
                print("pAUC : {}".format(p_auc))
                print("pAUC (source) : {}".format(p_auc_s))
                print("precision (source) : {}".format(prec_s))
                print("recall (source) : {}".format(recall_s))
                print("F1 score (source) : {}".format(f1_s))

                if len(y_true_t) > 0:
                    auc_t = metrics.roc_auc_score(y_true_t_auc, y_pred_t_auc)
                    p_auc_t = metrics.roc_auc_score(y_true_t, y_pred_t, max_fpr=self.args.max_fpr)
                    tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(y_true_t, [1 if x > decision_threshold else 0 for x in y_pred_t]).ravel()
                    prec_t = tp_t / np.maximum(tp_t + fp_t, sys.float_info.epsilon)
                    recall_t = tp_t / np.maximum(tp_t + fn_t, sys.float_info.epsilon)
                    f1_t = 2.0 * prec_t * recall_t / np.maximum(prec_t + recall_t, sys.float_info.epsilon)
                    if len(csv_lines) == 0:
                        csv_lines.append(self.result_column_dict["source_target"])
                    csv_lines.append([section_name.split("_", 1)[1],
                                    auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])

                    performance.append([auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
                    performance_over_all.append([auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])

                    anm_score_figdata.append_figdata(anm_score_figdata.anm_score_to_figdata(
                        scores=[[t, p] for t, p in zip(y_true_t, y_pred_t)],
                        title=f"{section_name}_target_AUC{auc_t}"
                    ))
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
            # calculate averages for AUCs and pAUCs
            amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["arithmetic mean"] + list(amean_performance))
            hmean_performance = scipy.stats.hmean(np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon), axis=0)
            csv_lines.append(["harmonic mean"] + list(hmean_performance))
            csv_lines.append([])

            # output results
            anm_score_figdata.show_fig(
                title=self.args.model+"_"+self.args.dataset+self.model_name_suffix+self.eval_suffix+"_anm_score",
                export_dir=result_dir
            )
        else:
            return
        
        result_path = result_dir/f"result_{self.args.dataset}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}_roc.csv"
        print("results -> {}".format(result_path))
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
        for j, batch in enumerate(test_loader):
            data = batch[0]
            data = data.to(self.device).float()
            y_true.append(batch[1][0].item())
            basename = batch[3][0]

            recon_data, _ = self.model(data)

            if self.args.score == "MAHALA":
                loss_source, num = loss_function_mahala(
                    recon_x=recon_data,
                    x=data,
                    block_size=self.block_size,
                    cov=inv_cov_source,
                    use_precision=True,
                    reduction=False
                )
                loss_source = self.loss_reduction(score=self.loss_reduction_1d(loss_source), n_loss=num)

                loss_target, num = loss_function_mahala(
                    recon_x=recon_data,
                    x=data,
                    block_size=self.block_size,
                    cov=inv_cov_target,
                    use_precision=True,
                    reduction=False
                )
                loss_target = self.loss_reduction(score=self.loss_reduction_1d(loss_target), n_loss=num)
                y_pred.append(min(loss_target.item(), loss_source.item()))
            elif self.args.score == "MAHALA_PCA":
                loss_source, num = self._mahala_pca_quad_form(recon_data, data, inv_cov_source)
                loss_source = self.loss_reduction(score=self.loss_reduction_1d(loss_source), n_loss=num)
                loss_target, num = self._mahala_pca_quad_form(recon_data, data, inv_cov_target)
                loss_target = self.loss_reduction(score=self.loss_reduction_1d(loss_target), n_loss=num)
                y_pred.append(min(loss_target.item(), loss_source.item()))
            else:
                y_pred.append(self.loss_fn(recon_x=recon_data, x=data).mean().item())
            
            # store anomaly scores
            anomaly_score_list.append([basename, y_pred[-1]])

            # store decision results
            if y_pred[-1] > decision_threshold:
                decision_result_list.append([basename, 1])
            else:
                decision_result_list.append([basename, 0])

            if mode:
                domain_list.append("target" if "target" in basename else "source")
        return y_pred, anomaly_score_list, decision_result_list, domain_list

def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)
