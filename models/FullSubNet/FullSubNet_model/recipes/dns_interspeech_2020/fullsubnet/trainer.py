import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import gc#################
import time###########



from audio_zen.acoustics.feature import drop_band
from audio_zen.trainer.base_trainer import BaseTrainer
from audio_zen.acoustics.mask import build_complex_ideal_ratio_mask, decompress_cIRM

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(self, dist, rank, config, resume, only_validation, model, loss_function, optimizer, train_dataloader, validation_dataloader):
        super().__init__(dist, rank, config, resume, only_validation, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for noisy, clean in tqdm(self.train_dataloader, desc="Training") if self.rank == 0 else self.train_dataloader:
            self.optimizer.zero_grad()

            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)

            noisy_mag, noisy_phase, noisy_real, noisy_imag = self.torch_stft(noisy)
            _, _, clean_real, clean_imag = self.torch_stft(clean)
            cIRM = build_complex_ideal_ratio_mask(noisy_real, noisy_imag, clean_real, clean_imag)  # [B, F, T, 2]
            cIRM = drop_band(
                cIRM.permute(0, 3, 1, 2),  # [B, 2, F ,T]
                self.model.module.num_groups_in_drop_band
            ).permute(0, 2, 3, 1)

            with autocast(enabled=self.use_amp):
                # [B, F, T] => [B, 1, F, T] => model => [B, 2, F, T] => [B, F, T, 2]
                noisy_mag = noisy_mag.unsqueeze(1)
                cRM = self.model(noisy_mag)
                cRM = cRM.permute(0, 2, 3, 1)
                loss = self.loss_function(cIRM, cRM)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total +=float( loss.item()) #خودم

        del cRM##############
        del cIRM##############
        del noisy_mag
        del noisy
        del clean
        if self.rank == 0:
            self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)
        torch.cuda.empty_cache() ###############
        print('cuda.memory_summary_1=',torch.cuda.memory_summary())##############
        time.sleep(1)#############
        
# ########
#         print('current_device2=',torch.cuda.current_device())
#         print('cuda.memory_allocated2=',torch.cuda.memory_allocated())
#         print('cuda.max_memory_allocated2',torch.cuda.max_memory_allocated())
#         print('cuda.memory_reserved2=',torch.cuda.memory_reserved())
#         print('cuda.max_memory_reserved2=',torch.cuda.max_memory_reserved())
#         print('cuda.memory_stats2=',torch.cuda.memory_stats())
#         print('cuda.memory_snapshot2=',torch.cuda.memory_snapshot())
#         print('cuda.memory_summary2=',torch.cuda.memory_summary())
        #torch.cuda.empty_cache()
#############
 

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        
        

        torch.cuda.empty_cache()# ###########
        print('cuda.memory_summary_2=',torch.cuda.memory_summary())
        self.model.eval()#خودم
        
        visualization_n_samples = self.visualization_config["n_samples"]
        visualization_num_workers = self.visualization_config["num_workers"]
        visualization_metrics = self.visualization_config["metrics"]

        loss_total = 0.0
        loss_list = {"With_reverb": 0.0, "No_reverb": 0.0, }
        item_idx_list = {"With_reverb": 0, "No_reverb": 0, }
        noisy_y_list = {"With_reverb": [], "No_reverb": [], }
        clean_y_list = {"With_reverb": [], "No_reverb": [], }
        enhanced_y_list = {"With_reverb": [], "No_reverb": [], }
        validation_score_list = {"With_reverb": 0.0, "No_reverb": 0.0}

        # speech_type in ("with_reverb", "no_reverb")
        for i, (noisy, clean, name, speech_type) in tqdm(enumerate(self.valid_dataloader), desc="Validation"):
            assert len(name) == 1, "The batch size for the validation stage must be one."
            name = name[0]
            speech_type = speech_type[0]
            
            print('cuda.memory_reserved111=',torch.cuda.memory_reserved())
            noisy = noisy.to(self.rank)
            clean = clean.to(self.rank)
            print('cuda.memory_reserved222=',torch.cuda.memory_reserved())

            noisy_mag, noisy_phase, noisy_real, noisy_imag = self.torch_stft(noisy)
            _, _, clean_real, clean_imag = self.torch_stft(clean)
            cIRM = build_complex_ideal_ratio_mask(noisy_real, noisy_imag, clean_real, clean_imag)  # [B, F, T, 2]

            print('cuda.memory_reserved1111=',torch.cuda.memory_reserved())
            noisy_mag = noisy_mag.unsqueeze(1)
            cRM = self.model(noisy_mag)
            cRM = cRM.permute(0, 2, 3, 1)

            loss = self.loss_function(cIRM, cRM)

            cRM = decompress_cIRM(cRM)
            print('cuda.memory_reserved2222=',torch.cuda.memory_reserved())

            enhanced_real = cRM[..., 0] * noisy_real - cRM[..., 1] * noisy_imag
            enhanced_imag = cRM[..., 1] * noisy_real + cRM[..., 0] * noisy_imag
            enhanced = self.torch_istft((enhanced_real, enhanced_imag), length=noisy.size(-1), input_type="real_imag")

            print('cuda.memory_reserved11=',torch.cuda.memory_reserved())
            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enhanced = enhanced.detach().squeeze(0).cpu().numpy()
            print('cuda.memory_reserved22=',torch.cuda.memory_reserved())

            assert len(noisy) == len(clean) == len(enhanced)
            loss_total += float(loss) ##################

            # Separated loss
            loss_list[speech_type] += float(loss) ##############
            item_idx_list[speech_type] += 1

            if item_idx_list[speech_type] <= visualization_n_samples:
                self.spec_audio_visualization(noisy, enhanced, clean, name, epoch, mark=speech_type)

            print('cuda.memory_reserved1=',torch.cuda.memory_reserved())
            noisy_y_list[speech_type].append(noisy)
            clean_y_list[speech_type].append(clean)
            enhanced_y_list[speech_type].append(enhanced)
            print('cuda.memory_reserved2=',torch.cuda.memory_reserved())
            gc.collect()##########################

            torch.cuda.empty_cache()#####################################
            print('cuda.memory_summary_1=',torch.cuda.memory_summary())##############
#             time.sleep(1)#############
#             torch.cuda.synchronize()##############
#             print('cuda.memory_summary_2=',torch.cuda.memory_summary())##############

         
        del enhanced_real###########
        del enhanced_imag###############
        del enhanced
        del cRM
        del noisy
        del clean
        ##############
        self.writer.add_scalar(f"Loss/Validation_Total", loss_total / len(self.valid_dataloader), epoch)

        for speech_type in ("With_reverb", "No_reverb"):
            self.writer.add_scalar(f"Loss/{speech_type}", loss_list[speech_type] / len(self.valid_dataloader), epoch)

            validation_score_list[speech_type] = self.metrics_visualization(
                noisy_y_list[speech_type], clean_y_list[speech_type], enhanced_y_list[speech_type],
                visualization_metrics, epoch, visualization_num_workers, mark=speech_type
            )

        return validation_score_list["No_reverb"]
