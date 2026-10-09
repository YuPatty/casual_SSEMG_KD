# test_causality.py
"""
驗證 student 網路是否真的 causal：
  把第 t0 幀（含）之後的輸入換成亂數，檢查「第 t0 幀之前」的 mask / phase 輸出有沒有改變。
  因果網路 → 差異應為 0（浮點誤差內）；非因果網路 → 差異 > 0（拿來確認這個測試抓得到洩漏）。

用法：
    python test_causality.py --config config/student_16ch_1blk_causal.yaml
    python test_causality.py --config config/student_16ch_1blk.yaml      # 預期失敗（原版非因果）
"""
import os, sys, argparse, yaml, torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))
from models.StudentNet import StudentSSEMGNet


@torch.no_grad()
def body(model, mag, pha):
    """跟 forward_spectrogram 同一段主幹（encoder → TFConv → decoders）。mag/pha: [B,T,F]"""
    x = torch.stack([mag, pha], dim=1)                 # [B,2,T,F]
    f = model.dense_encoder(x)
    for blk in model.TFConv:
        f = blk(f)
    mask = model.mask_decoder(f)                       # [B,1,T,F]
    pha_o = model.phase_decoder(f) if model.phase_decoder is not None else None
    return mask, pha_o


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/student_16ch_1blk_causal.yaml')
    p.add_argument('--frames', type=int, default=64)
    p.add_argument('--tol', type=float, default=1e-5)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding='utf-8'))
    torch.manual_seed(0)
    model = StudentSSEMGNet(cfg).eval()
    # 隨機化所有參數（含 norm 的 weight/bias），避免「全部是預設值剛好看不出洩漏」
    for prm in model.parameters():
        prm.data.add_(0.1 * torch.randn_like(prm))

    Fb = cfg['model']['n_fft'] // 2 + 1
    T = a.frames
    mag = torch.rand(1, T, Fb) * 2
    pha = (torch.rand(1, T, Fb) * 2 - 1) * 3.14159
    m_ref, p_ref = body(model, mag, pha)

    print(f"config={a.config} | causal={model.causal} | T={T}, F={Fb}")
    worst = 0.0
    for t0 in (1, 5, T // 2, T - 3):
        mag2, pha2 = mag.clone(), pha.clone()
        mag2[:, t0:] = torch.rand_like(mag2[:, t0:]) * 2
        pha2[:, t0:] = (torch.rand_like(pha2[:, t0:]) * 2 - 1) * 3.14159
        m2, p2 = body(model, mag2, pha2)
        d_mask = (m2[:, :, :t0] - m_ref[:, :, :t0]).abs().max().item()
        d_pha = 0.0 if p_ref is None else (p2[:, :, :t0] - p_ref[:, :, :t0]).abs().max().item()
        worst = max(worst, d_mask, d_pha)
        print(f"  t0={t0:3d}: 過去幀 mask 最大差 = {d_mask:.3e} | phase 最大差 = {d_pha:.3e}")

    ok = worst <= a.tol
    print("RESULT:", "✔ causal（未來輸入不影響過去輸出）" if ok else "✘ NOT causal（過去輸出被未來輸入改變）")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
