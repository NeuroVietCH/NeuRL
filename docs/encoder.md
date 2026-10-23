# Encoder trong NeuRL

Tài liệu này mô tả đúng trạng thái code hiện tại của ba encoder chính:

- 3D-Neuro-SimCLR ResNet-18;
- NeuroVFM;
- Neuro-JEPA-Large.

`architecture_v3.md` và `Phân-công_v3.md` là tài liệu thiết kế/kế hoạch để tham
khảo. Các shape, command và trạng thái "đã chạy" trong file này chỉ được ghi là
đã implement khi chúng khớp với source, config và test hiện tại.

Mục tiêu chung là biến một MRI T1 3D thành nhiều vector đặc trưng có vị trí để
Reader/Agent có thể chọn bằng chứng. `Local` trong tài liệu này chỉ nói về phạm
vi thông tin của token, không có nghĩa token đã tương ứng với một vùng giải phẫu
được đặt tên.

## 1. Workflow chung

```text
MRI T1 thô
  -> preprocessing riêng của encoder
  -> tensor/patches đúng không gian đầu vào
  -> frozen encoder có pretrained weights
  -> native region/token embeddings
  -> tọa độ + mask hợp lệ
  -> cache
  -> adapter về chiều chung (ví dụ 128/256)
  -> Reader / Agent / classifier
```

Không được dùng cùng một preprocessing cho mọi encoder. Mỗi checkpoint học trên
một quy trình khác nhau; đổi orientation, spacing, skull stripping hay intensity
normalization có thể làm phân phối đầu vào lệch khỏi lúc pretrain.

## 2. Bảng so sánh

| Encoder | Input vào model | Native output dùng trong NeuRL | Vị trí | Local hay global? | Code |
|---|---|---|---|---|---|
| 3D-Neuro-SimCLR ResNet-18 | Tensor `[B,1,D,H,W]` | Layer 3: `[B,N,256]` | Lưới layer 3 ngầm định | Spatial/convolutional; receptive field chồng lấn, không độc lập | `src/neuro_evidence/resnet3d.py` |
| NeuroVFM | Patch tensor tạo từ scan, patch mặc định `[4,16,16]` | `[N,768]` | `coords [N,3]` | Token cuối đã trao đổi context qua Transformer; không thuần local | `src/neuro_evidence/encoder.py` |
| Neuro-JEPA-Large | `[1,1,96,108,96]` sau pipeline MNI | `[576,1024]` | `coords [576,3]`, grid `8x9x8` | Token ViT cuối có context; không thuần local | `src/neuro_evidence/neurojepa.py` |

`PlaceholderEncoder` trong `src/models/encoders.py` chỉ là random projection để
debug. `CachedTokenEncoder` chỉ đọc cache đã tạo trước; hai class này không phải
foundation encoder để so sánh chất lượng biểu diễn.

## 3. 3D-Neuro-SimCLR ResNet-18

### Pipeline và shape

Backbone giữ nguyên kiến trúc của checkpoint: ảnh xám một channel, `conv1` kernel
7 và stride 1, max-pool stride 2, sau đó các residual stage. Wrapper dừng tại
`layer3`; `layer4` và SimCLR projector không tham gia output.

```text
Input                       [B,1,D,H,W]
  -> conv1 + BN + ReLU
  -> maxpool
  -> layer1
  -> layer2
  -> layer3 feature map     [B,256,d,h,w]
  -> flatten spatial
  -> transpose              [B,d*h*w,256]
```

Ví dụ với `[B,1,96,112,96]`:

```text
layer3 feature map = [B,256,12,14,12]
region embeddings  = [B,2016,256]
```

Con số `2016 = 12*14*12` khác thiết kế MedicalNet stride lớn hơn có 252 vị trí.
Checkpoint SimCLR này dùng stem stride 1, nên không được hardcode 252.

### SimCLR ResNet khác "ResNet thường" ở đâu?

Tên **SimCLR** chủ yếu mô tả cách pretrain, không phải một loại residual block
mới. Backbone trong checkpoint vẫn là 3D ResNet-18 với `conv1`, bốn residual
stage, skip connections, batch normalization và ReLU như ResNet-18 tương ứng.

| Thành phần | ResNet supervised thông thường | ResNet trong 3D-Neuro-SimCLR |
|---|---|---|
| Backbone | ResNet | Vẫn là ResNet |
| Input của bản này | Tùy cấu hình | 3D MRI, 1 channel |
| Mục tiêu pretrain | Thường cross-entropy từ nhãn | Contrastive learning từ hai augmented views, không cần nhãn chẩn đoán |
| Head lúc pretrain | Classification head | Projection MLP sau global ResNet feature |
| Weight học được | Tối ưu cho task có nhãn | Tối ưu để hai view cùng scan có representation gần nhau |
| Dùng trong NeuRL | Có thể lấy feature nếu kiến trúc khớp | Chỉ lấy `encoder.*`; bỏ projection MLP |

Vì vậy, nếu tạo **đúng cùng 3D ResNet-18 config** rồi train supervised hoặc để
random initialization, cấu trúc tensor có thể giống hệt và điểm khác chính là
weights, BatchNorm running statistics, training objective và head dùng lúc
train. Nhưng không được nói mọi "normal ResNet" chỉ khác weights:

- torchvision ResNet-18 phổ biến là 2D, RGB 3 channel, nên không cùng kiến trúc
  input với model MRI 3D này;
- MedicalNet/ResNet3D khác có thể dùng `conv1` stride, shortcut type hoặc
  preprocessing khác, làm grid output khác;
- checkpoint và model config phải khớp tuyệt đối mới được strict-load.

Trong pretraining SimCLR, global feature sau layer 4 được đưa qua projection MLP
để tính contrastive loss. Trong inference NeuRL hiện tại, projection MLP không
được load và cũng không chạy; output lấy sớm ở layer 3 để giữ lưới không gian.

### Local và global

- Mỗi row `[256]` gắn với một vị trí trên lưới layer 3.
- Đây không phải patch độc lập. Receptive field lý thuyết ở layer 3 rộng 109
  voxel mỗi trục, trong khi tâm hai vị trí kề nhau cách 8 voxel. Các vùng nhìn
  thấy chồng lấn mạnh và ở volume nhỏ có thể bao phủ phần lớn ảnh.
- `forward()` chỉ trả region embeddings `[B,N,256]` theo yêu cầu hiện tại.
- Nếu downstream cần một vector tóm tắt đơn giản, có thể dùng
  `embeddings.mean(dim=1)` để được `[B,256]`. Đây là mean pooling do downstream
  định nghĩa, không phải global head đã được huấn luyện riêng.
- Global feature nguyên bản của ResNet có thể lấy từ layer 4 + average pooling
  thành `[B,512]`, nhưng wrapper hiện tại cố ý không chạy đường này.
- Do receptive field chồng lấn, implementation hiện tại **chưa thỏa** yêu cầu
  "mỗi local token được encode độc lập" trong `Phân-công_v3.md`. Muốn thỏa yêu
  cầu đó phải encode crop/window riêng hoặc có locality mechanism được kiểm
  chứng; chỉ gọi layer-3 cell là local token là chưa đủ.

### Weight nằm ở đâu?

Default:

```text
checkpoints/3d_neuro_simclr/simclr_3d_brain_foundation.tar
```

Tên `.tar` hơi gây nhầm: đây là PyTorch ZIP checkpoint và `torch.load()` đọc trực
tiếp được. File chứa epoch 150, `model_state_dict`, optimizer state, encoder và
projector. Loader chỉ lấy 123 tensor `encoder.*`, bỏ projector, rồi nạp backbone
với `strict=True`.

Không cần giữ thêm thư mục `checkpoint_150/`. Thư mục đó chỉ là nội dung giải nén
trùng với file trên và đã được dọn khỏi project. Cần giữ **một trong hai**:

1. file checkpoint mặc định trong project; hoặc
2. một file checkpoint ở shared storage/vị trí khác và truyền đường dẫn bằng
   `checkpoint_path=` khi gọi loader.

Nếu xóa file mặc định mà không cung cấp path khác, encoder không có pretrained
weights và sẽ dừng bằng `FileNotFoundError`; smoke test cũng không chạy.
`checkpoints/` đã nằm trong `.gitignore`, vì vậy weight lớn không được commit Git.

### Smoke test bắt buộc dùng weight

Chạy từ repository root:

```bash
PYTHONPATH=src .venv/bin/python tests/test_resnet3d_smoke.py
```

Test thực hiện theo thứ tự:

1. tìm checkpoint;
2. `torch.load(..., weights_only=True)`;
3. xác nhận epoch 150;
4. strict-load encoder weights;
5. tạo hai MRI phantom 3D có ellipsoid mô, hai cavity đối xứng, bias field,
   texture, correlated noise và fine noise;
6. chạy layer 3;
7. kiểm tra shape, finite, non-constant, repeatability và hai sample tạo output
   khác nhau.

### Gọi bằng Python và lấy embedding

```python
import torch

from neuro_evidence.resnet3d import load_pretrained_simclr_resnet18
from tests.test_resnet3d_smoke import make_fake_brain_batch

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = load_pretrained_simclr_resnet18(device=device)
fake_mri = make_fake_brain_batch(
    batch_size=2,
    shape=(96, 112, 96),
).to(device)

with torch.inference_mode():
    embeddings = encoder(fake_mri)

print(fake_mri.shape)    # [2, 1, 96, 112, 96]
print(embeddings.shape)  # [2, 2016, 256]
```

Nếu weight nằm ngoài project:

```python
encoder = load_pretrained_simclr_resnet18(
    checkpoint_path="/path/to/simclr_3d_brain_foundation.tar",
    device=device,
)
```

Phantom chỉ kiểm tra plumbing và numerical behavior. Nó không chứng minh encoder
hoạt động tốt trên MRI thật. Inference nghiên cứu phải tái tạo preprocessing mà
3D-Neuro-SimCLR dùng khi pretrain và phải audit overlap dữ liệu; checkpoint này
đã học từ tập có ADNI nên không nên tự động coi kết quả ADNI là held-out sạch.

## 4. NeuroVFM

### Pipeline

```text
NIfTI/DICOM
  -> NeuroVFM load_image
  -> reorient/resample theo pipeline chính thức
  -> clip percentile + scale intensity
  -> patch mặc định [4,16,16]
  -> foreground/token rule
  -> NeuroVFM encoder
  -> embeddings [N,768] + coords [N,3]
```

`NeuroVFMEncoder.extract()` nhận đường dẫn scan, không nhận trực tiếp một tensor
`[B,1,D,H,W]`. Output là `TokenBatch` gồm embedding, coordinate, grid/volume
metadata, brain fraction và cờ foreground. `N` thay đổi theo kích thước scan và
quy tắc lọc background/mask.

### Local và global

Patch ban đầu local, nhưng embedding cuối đã đi qua các block có self-attention,
vì vậy một token có thể chứa thông tin từ token khác trong cùng scan. Coordinate
vẫn cho biết token bắt nguồn từ patch nào, không bảo đảm nội dung chỉ đến từ
patch đó. Code hiện tại không trả dedicated global vector; downstream có thể
pool các token nhưng phải ghi rõ rule pooling.

## 5. Neuro-JEPA-Large

### Pipeline

```text
T1 MRI
  -> RAS + drop neck + N4
  -> affine registration MNI152 1 mm
  -> skull strip
  -> MONAI transforms
  -> tensor [1,1,96,108,96]
  -> patches [12,12,12]
  -> grid [8,9,8] = 576 tokens
  -> Neuro-JEPA-Large
  -> embeddings [576,1024]
```

Ngoài embeddings, `extract()` trả `coords [576,3]`, `grid_shape [8,9,8]` và
`brain_fraction [576]`. Vì grid cố định sau MNI preprocessing, số token cố định.

### Local và global

Mỗi token có patch origin rõ ràng trên lưới MNI, nhưng output từ backbone ViT là
contextual representation. Nó phù hợp để so sánh vị trí chuẩn giữa bệnh nhân,
nhưng không được gọi là bằng chứng local độc lập nếu chưa trích representation
trước context mixing hoặc chưa chạy từng crop riêng.

## 6. Chọn encoder nào cho thí nghiệm nào?

| Câu hỏi | Encoder phù hợp để bắt đầu | Lý do/cảnh báo |
|---|---|---|
| Cần region grid từ CNN, ít sửa backbone | SimCLR ResNet layer 3 | Dễ flatten thành `[B,N,256]`; receptive fields vẫn overlap |
| Cần pipeline clinical scan linh hoạt và token metadata | NeuroVFM | Có preprocessing/token filtering sẵn; token cuối mang global context |
| Cần grid cố định trong MNI và embedding lớn | Neuro-JEPA | 576 vị trí cố định; preprocessing registration nặng |
| Cần local evidence thật sự không nhìn ngoài vùng | Chưa encoder nào ở trên bảo đảm | Phải crop/tile trước encoder hoặc trích trước bước context mixing, rồi kiểm chứng |

Để so sánh công bằng, phải cố định subject split, nhãn, downstream Reader,
budget token và metric. Không so AUC giữa các cache được tạo từ preprocessing hay
cohort khác nhau rồi kết luận encoder nào tốt hơn.
