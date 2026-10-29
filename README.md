# DS201 - Lab 2: Convolutional Neural Networks

Lab 2 thực hiện bài toán phân loại 21 món ăn Việt Nam trên bộ dữ liệu
VinaFood21 bằng bốn kiến trúc:

1. LeNet-5 với Average Pooling và Max Pooling.
2. GoogLeNet/Inception huấn luyện từ đầu.
3. ResNet-18 huấn luyện từ đầu.
4. ResNet-50 pretrained với Feature Extraction và Fine-tuning.

Checkpoint tốt nhất của mỗi thí nghiệm được chọn theo **Macro-F1 trên tập
validation**. Tập test chỉ được dùng để đánh giá mô hình sau khi huấn luyện.

## 1. Cấu trúc thư mục

```text
Lab2/
|-- README.md
|-- VinaFood21/
|   `-- VinaFood21/
|       |-- train/
|       `-- test/
|-- bai1_lenet/
|   |-- data.py
|   |-- engine.py
|   |-- model.py
|   |-- train.py
|   |-- evaluate.py
|   |-- outputs_avg/
|   `-- outputs_max/
|-- bai2_googlenet/
|   |-- data.py
|   |-- engine.py
|   |-- model.py
|   |-- train.py
|   |-- evaluate.py
|   `-- outputs_googlenet/
|-- bai3_resnet18/
|   |-- data.py
|   |-- engine.py
|   |-- model.py
|   |-- train.py
|   |-- evaluate.py
|   `-- outputs_resnet18/
`-- bai4_resnet50/
    |-- data.py
    |-- engine.py
    |-- model.py
    |-- train.py
    |-- evaluate.py
    |-- outputs_feature/
    |-- outputs_finetune/
    |-- summary.csv
    `-- experiment_config.json
```

Bộ dữ liệu không được đưa lên GitHub. `data_root` phải là thư mục trực tiếp
chứa `train/` và `test/`. Nếu không có thư mục `val/`, chương trình tách cố
định 10% dữ liệu train làm validation với seed 42.

Mỗi lớp được tổ chức theo định dạng `ImageFolder`:

```text
train/
|-- banh-can/
|-- banh-hoi/
|-- ...
`-- thit-kho-trung/
```

## 2. Cài đặt môi trường

Yêu cầu Python 3.10 trở lên. Có thể cài các thư viện bằng lệnh:

```bash
pip install torch torchvision scikit-learn matplotlib pillow transformers
```

Kiểm tra PyTorch và GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Các lệnh bên dưới giả sử terminal đang đứng trong thư mục của từng bài và dữ
liệu nằm tại `../VinaFood21/VinaFood21`. Hãy thay `--data_root` nếu dữ liệu
được đặt ở vị trí khác.

## 3. Bài 1 - LeNet-5

Hai thí nghiệm sử dụng cùng seed, learning rate và số epoch; chỉ thay đổi lớp
pooling để bảo đảm so sánh công bằng.

### Average Pooling

```bash
cd bai1_lenet
python train.py --data_root "../VinaFood21/VinaFood21" --pool avg --img_size 64 --epochs 20 --batch_size 32 --lr 0.001 --num_workers 2 --seed 42 --out_dir outputs_avg
python evaluate.py --data_root "../VinaFood21/VinaFood21" --ckpt "outputs_avg/best.pt" --batch_size 32 --num_workers 2
```

### Max Pooling

```bash
python train.py --data_root "../VinaFood21/VinaFood21" --pool max --img_size 64 --epochs 20 --batch_size 32 --lr 0.001 --num_workers 2 --seed 42 --out_dir outputs_max
python evaluate.py --data_root "../VinaFood21/VinaFood21" --ckpt "outputs_max/best.pt" --batch_size 32 --num_workers 2
```

## 4. Bài 2 - GoogLeNet

Mô hình GoogLeNet được xây dựng từ các Inception block và huấn luyện từ đầu
bằng Adam.

```bash
cd ../bai2_googlenet
python train.py --data_root "../VinaFood21/VinaFood21" --img_size 64 --epochs 20 --batch_size 32 --lr 0.001 --num_workers 2 --seed 42 --out_dir outputs_googlenet
python evaluate.py --data_root "../VinaFood21/VinaFood21" --ckpt "outputs_googlenet/best.pt" --batch_size 32 --num_workers 2
```

## 5. Bài 3 - ResNet-18

ResNet-18 được xây dựng với cấu hình residual block `[2, 2, 2, 2]` và huấn
luyện từ đầu bằng Adam.

```bash
cd ../bai3_resnet18
python train.py --data_root "../VinaFood21/VinaFood21" --img_size 112 --epochs 20 --batch_size 32 --lr 0.001 --num_workers 2 --seed 42 --out_dir outputs_resnet18
python evaluate.py --data_root "../VinaFood21/VinaFood21" --ckpt "outputs_resnet18/best.pt" --batch_size 32 --num_workers 2
```

## 6. Bài 4 - Pretrained ResNet-50

Mô hình `microsoft/resnet-50` được tải bằng Transformers và chạy ở hai chế độ:

- `feature`: đóng băng backbone, chỉ huấn luyện classifier.
- `finetune`: cập nhật toàn bộ mô hình với learning rate nhỏ hơn.

Lần chạy đầu cần kết nối Internet để tải pretrained weights.

### Feature Extraction

```bash
cd ../bai4_resnet50
python train.py --data_root "../VinaFood21/VinaFood21" --mode feature --img_size 224 --epochs 10 --batch_size 16 --lr 0.001 --num_workers 2 --seed 42 --out_dir outputs_feature
python evaluate.py --data_root "../VinaFood21/VinaFood21" --ckpt "outputs_feature/best.pt" --batch_size 16 --num_workers 2
```

### Fine-tuning

```bash
python train.py --data_root "../VinaFood21/VinaFood21" --mode finetune --img_size 224 --epochs 15 --batch_size 16 --lr 0.0001 --num_workers 2 --seed 42 --out_dir outputs_finetune
python evaluate.py --data_root "../VinaFood21/VinaFood21" --ckpt "outputs_finetune/best.pt" --batch_size 16 --num_workers 2
```

## 7. Kết quả thực nghiệm

Các số liệu dưới đây được tính trên 6.682 ảnh test. Precision, Recall và F1
được lấy theo macro average trên 21 lớp.

| Bài | Mô hình/cấu hình | Best epoch | Accuracy | Precision-macro | Recall-macro | F1-macro |
|---|---|---:|---:|---:|---:|---:|
| 1 | LeNet-5 + AvgPool | 18/20 | 0.3237 | 0.3233 | 0.3140 | 0.3085 |
| 1 | LeNet-5 + MaxPool | 20/20 | 0.3298 | 0.3422 | 0.3190 | 0.3213 |
| 2 | GoogLeNet | 20/20 | 0.4133 | 0.4307 | 0.3867 | 0.3920 |
| 3 | ResNet-18, 112 x 112 | 20/20 | 0.4994 | 0.5475 | 0.4959 | 0.4895 |
| 4 | ResNet-50 Feature Extraction | 8/10 | 0.7291 | 0.7513 | 0.7405 | 0.7399 |
| 4 | ResNet-50 Fine-tuning | 15/15 | **0.8583** | **0.8586** | **0.8634** | **0.8590** |

### Nhận xét

- Max Pooling cho kết quả nhỉnh hơn Average Pooling trong LeNet-5.
- GoogLeNet và ResNet-18 huấn luyện từ đầu cải thiện dần theo độ sâu của mô
  hình, nhưng vẫn bị giới hạn bởi kích thước dữ liệu và thời gian huấn luyện.
- ResNet-50 pretrained vượt rõ rệt các mô hình huấn luyện từ đầu.
- Fine-tuning toàn bộ ResNet-50 cho kết quả tốt hơn Feature Extraction, cho
  thấy pretrained features thích nghi tốt hơn khi được cập nhật trên
  VinaFood21.

## 8. Các file kết quả

Mỗi thư mục output chứa:

| File | Ý nghĩa |
|---|---|
| `best.pt` | Checkpoint có validation Macro-F1 tốt nhất |
| `history.csv` | Loss, Accuracy và Macro-F1 theo từng epoch |
| `loss_curve.png` | Đường train loss và validation loss |
| `classification_report.txt` | Kết quả test tổng thể và theo từng lớp |
| `confusion_matrix.png` | Ma trận nhầm lẫn trên tập test |

Bài 4 có thêm `summary.csv` để so sánh trực tiếp Feature Extraction và
Fine-tuning, cùng `experiment_config.json` lưu cấu hình thực nghiệm.

## 9. Ghi chú tái lập kết quả

- Các thí nghiệm sử dụng seed 42.
- Optimizer sử dụng là Adam.
- Dữ liệu train có augmentation; validation và test không augmentation.
- Không sử dụng tập test để chọn checkpoint hoặc điều chỉnh siêu tham số.
- Kết quả có thể chênh lệch nhẹ giữa các phiên bản PyTorch, CUDA và GPU.
- Không commit thư mục dataset, `__pycache__` hoặc các file tạm của notebook.
- Các checkpoint pretrained có dung lượng lớn; có thể dùng Git LFS nếu cần
  lưu chúng trên GitHub.
