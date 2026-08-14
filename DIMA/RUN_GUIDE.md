# 🚀 Hướng Dẫn Chạy Huấn Luyện: DIMA (Diffusion) & FIMA (Flow Matching) trên MATE

Tài liệu này hướng dẫn chi tiết cách chạy thử nghiệm huấn luyện cho cả **Baseline DIMA (Diffusion World Model)** và **FIMA (Flow Matching World Model)** trên môi trường multi-agent MATE.

---

## 📋 1. Chuẩn Bị Môi Trường

Kích hoạt môi trường Conda và di chuyển vào thư mục `DIMA`:

```bash
conda activate dima
cd D:\Diffusion-inspired-Multi-Agent-on-MATE\Diffusion-inspired-Multi-Agent-on-MATE\DIMA
```

---

## ⚡ 2. Câu Lệnh Chạy Huấn Luyện

### 🟢 2.1. Huấn luyện FIMA (Flow Matching World Model - Mô Hình Mới)

Sử dụng tham số `--world_model_type flow_matching`:

```bash
python train.py \
    --env mate \
    --env_name MATE-4v8-9 \
    --policy_class discrete \
    --mate_levels 5 \
    --mate_episode_limit 200 \
    --world_model_type flow_matching \
    --steps 10000000 \
    --n_workers 2 \
    --mode disabled
```

---

### 🔵 2.2. Huấn luyện DIMA Gốc (Diffusion World Model - Baseline Đối Chứng)

Sử dụng tham số `--world_model_type diffusion` (hoặc bỏ qua cờ này, mặc định hệ thống sẽ dùng `diffusion`):

```bash
python train.py \
    --env mate \
    --env_name MATE-4v8-9 \
    --policy_class discrete \
    --mate_levels 5 \
    --mate_episode_limit 200 \
    --world_model_type diffusion \
    --steps 10000000 \
    --n_workers 2 \
    --mode disabled
```

---

## 🛠️ 3. Giải Thích Các Tham Số Cốt Lõi

| Tham số | Ý nghĩa / Giá trị | Ghi chú |
| :--- | :--- | :--- |
| `--env` | `mate` | Tên môi trường huấn luyện |
| `--env_name` | `MATE-4v8-9` | Kịch bản 4 camera quan sát vs 8 đối tượng di chuyển |
| `--policy_class` | `discrete` | Loại hành động rời rạc (Discrete Camera Control) |
| `--mate_levels` | `5` | Số mức góc của camera ($5 \times 5 = 25$ không gian hành động) |
| `--mate_episode_limit`| `200` | Số bước tối đa cho 1 episode |
| **`--world_model_type`** | **`flow_matching`** / **`diffusion`** | **Cờ rẽ nhánh lựa chọn World Model backend** |
| `--steps` | `10000000` (10M) | Tổng số bước tương tác môi trường |
| `--n_workers` | `2` | Số worker Ray lấy kinh nghiệm song song |
| `--mode` | `disabled` | Tắt WandB (có thể đổi thành `online` để log lên WandB) |

---

## ⚙️ 4. Điều Chỉnh Siêu Tham Số Trực Tiếp Trong Code

### 4.1. Siêu tham số World Model (FIMA)
- **Tốc độ học (Learning Rate)**: Sửa `self.FM_LR = 1e-4` trong `DIMA/configs/dreamer/mate/MATELearnerConfig.py`.
- **Số bước lấy mẫu (Sampling Steps $M$)**: Sửa `self.fm_num_sampling_steps = 4` trong `DIMA/configs/dreamer/DreamerAgentConfig.py`.
- **Horizon ($H$)**: Sửa `self.horizon = 15` trong `DreamerAgentConfig.py`.

### 4.2. Khái niệm Codebook & Tokenizer (State Decoder)
- **Codebook (Từ điển mã)**: Tập hợp các vector đại diện rời rạc dùng để nén trạng thái liên tục $s_t \in \mathbb{R}^{220}$ thành dạng Token rời rạc trước khi giải mã ra quan sát `joint_obs` của các camera.
- **`vq_type`**: `'fsq'` (Finite Scalar Quantization - khuyến nghị) hoặc `'vq'` (Vector Quantization truyền thống).
- **`nums_obs_token`**: `12` (Số lượng token biểu diễn cho 1 trạng thái).
- **`OBS_VOCAB_SIZE`**: `128` (Kích thước từ điển Codebook khi dùng VQ).
- **`levels`**: `[8, 6, 5]` (Các mức phân giải của FSQ $\implies 8 \times 6 \times 5 = 240$ trạng thái mã hóa).

---

## 📊 5. Xem Kết Quả & TensorBoard

Log và checkpoint sẽ được lưu tự động tại:
`D:\Diffusion-inspired-Multi-Agent-on-MATE\Diffusion-inspired-Multi-Agent-on-MATE\DIMA\0814_results\mate\MATE-4v8-9\`

Mở TensorBoard để theo dõi trực quan:
```bash
tensorboard --logdir 0814_results/
```
