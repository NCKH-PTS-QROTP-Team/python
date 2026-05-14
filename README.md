# Face Recognition Service

Microservice xác thực khuôn mặt cho hệ thống điểm danh, hỗ trợ **phát hiện khuôn mặt**, **đăng ký khuôn mặt** và **chống lừa đảo (anti-spoofing)**. Tài liệu này hỗ trợ viết NCKH (nghiên cứu khoa học) về công nghệ nhận diện khuôn mặt.

---

## 1. Tổng quan

Service sử dụng các mô hình deep learning chạy **local** (không phụ thuộc API bên ngoài), triển khai dưới dạng REST API (FastAPI). Hỗ trợ GPU (CUDA) hoặc CPU.

| Chức năng            | Mô tả                                                                 |
|----------------------|-----------------------------------------------------------------------|
| Phát hiện khuôn mặt  | Phát hiện vị trí mặt và 5 điểm landmark (mắt, mũi, miệng)             |
| Đăng ký khuôn mặt    | Trích xuất embedding 512 chiều từ ảnh khuôn mặt                       |
| Xác thực khuôn mặt   | So sánh embedding với dữ liệu đã đăng ký (cosine similarity)          |
| Chống lừa đảo        | Phân biệt mặt thật / ảnh in / video replay (liveness detection)       |

---

## 2. Định dạng khuôn mặt (Face Format)

### 2.1 Luồng xử lý ảnh

1. **Input**: Ảnh base64 (JPEG/PNG)
2. **Decode**: PIL/OpenCV → mảng BGR (OpenCV format)
3. **Detection**: Phát hiện bounding box `[x1, y1, x2, y2, confidence]` và 5 landmarks
4. **Alignment**: Căn chỉnh khuôn mặt về 112×112 px dựa trên landmarks
5. **Embedding**: Trích xuất vector 512 chiều (L2-normalized)

### 2.2 Định dạng bounding box và landmarks

- **Bounding box**: `[x1, y1, x2, y2, confidence]` – tọa độ góc và điểm tin cậy (0–1)
- **5 landmarks**: 2 mắt, mũi, 2 mép miệng – dùng cho face alignment (norm_crop)

### 2.3 Định dạng embedding

- **Kích thước**: 512 chiều (float32)
- **Chuẩn hóa**: L2-norm (vector đơn vị)
- **So sánh**: Cosine similarity = tích vô hướng (vì đã normalize)

---

## 3. Đăng ký khuôn mặt (Face Registration)

### 3.1 Quy trình

1. Người dùng chụp ảnh (1 khuôn mặt duy nhất trong ảnh)
2. Service phát hiện mặt → lấy bounding box và landmarks
3. Face alignment: căn chỉnh về 112×112 px (insightface `norm_crop`)
4. **Face enhancement** (tùy chọn): khử noise, cân bằng sáng (CLAHE), sharpening (Unsharp Mask)
5. ArcFace trích xuất embedding 512D
6. L2-normalize và lưu (Backend lưu trong DB dưới dạng mảng float)

### 3.2 Enhancement trước khi trích xuất

- **Bilateral denoise**: Giảm nhiễu, giữ chi tiết
- **Gamma correction**: Tự động cân bằng độ sáng
- **CLAHE + Unsharp Mask**: Tăng độ nét thích ứng theo độ mờ ảnh (Laplacian variance)

### 3.3 API đăng ký

```
POST /api/extract-encoding
Body: { "base64Image": "data:image/jpeg;base64,..." }
Response: { "faceEncoding": [0.12, -0.34, ...], "bbox": {...}, "processedImageBase64": "..." }
```

---

## 4. Chống lừa đảo (Anti-Spoofing)

Service sử dụng **nhiều lớp** để hạn chế gian lận:

### 4.1 Mô hình phát hiện liveness

- **Tên mô hình**: MiniFASNet (antispoof_80x80.onnx)
- **Nguồn**: Minivision, từ `checkin_face_anti_spoofing`
- **Input**: Ảnh khuôn mặt 80×80 px (RGB)
- **Output**: Xác suất 3 lớp: **Real** (mặt thật), **Print** (ảnh in), **Replay** (video replay)
- **Ngưỡng**: `real_prob >= 0.55` → coi là mặt thật

### 4.2 Quality gates (chặn ảnh giả / kém chất lượng)

| Cổng                 | Ngưỡng        | Mục đích                                           |
|----------------------|---------------|----------------------------------------------------|
| **Kích thước mặt**   | `min_side >= 120px` | Mặt quá nhỏ → dễ là ảnh chụp màn hình, kém chất lượng |
| **Độ nét (blur)**    | `Laplacian variance >= 250` | Ảnh mờ → khó đánh giá liveness                    |
| **Liveness score**   | `real_prob >= 0.55` | Ngăn ảnh in, video replay                        |

### 4.3 Phát hiện blur (Laplacian variance)

```python
# Kênh Y (YCrCb) + Gaussian blur → Laplacian → variance
lap_var = cv2.Laplacian(y_channel, cv2.CV_64F).var()
# lap_var < 250 → ảnh quá mờ → reject
```

Ảnh mờ thường dùng trong tấn công replay (chụp màn hình, in ảnh) nên bị loại sớm.

### 4.4 Chuỗi xử lý anti-spoof

1. Phát hiện mặt → crop (có pad 50px)
2. Kiểm tra kích thước (`min_side >= 120`)
3. Kiểm tra độ nét (`lap_var >= 250`)
4. **Face enhancement** (denoise, gamma, CLAHE, sharpening)
5. Resize về 80×80 → đưa vào mô hình MiniFASNet
6. Lấy xác suất lớp **Real** → so sánh với ngưỡng 0.55

---

## 5. Công nghệ sử dụng

### 5.1 Mô hình

| Thành phần       | Mô hình                    | File                        | Chức năng              |
|------------------|----------------------------|-----------------------------|------------------------|
| Detection        | SCRFD                      | `det_10g.onnx`             | Phát hiện mặt + landmarks |
| Recognition      | ArcFace (ResNet50)         | `w600k_r50.onnx`           | Trích xuất embedding   |
| Anti-spoof       | MiniFASNet                 | `antispoof_80x80.onnx`     | Liveness detection     |

### 5.2 Thư viện

- **InsightFace**: Detection (SCRFD), alignment, ArcFace
- **ONNX Runtime**: Chạy mô hình ONNX (CPU/GPU)
- **OpenCV**: Xử lý ảnh (blur, Laplacian, CLAHE, …)
- **FastAPI**: REST API

### 5.3 Fallback

- Nếu không có InsightFace → OpenCV Haar Cascade (chỉ detection, không embedding)
- Nếu không có GPU → chạy trên CPU

---

## 6. API Endpoints

| Method | Endpoint              | Mô tả                              |
|--------|------------------------|------------------------------------|
| GET    | `/`                   | Thông tin service và models        |
| POST   | `/api/detect`         | Phát hiện mặt trong ảnh            |
| POST   | `/api/extract-encoding` | Đăng ký: trích xuất embedding    |
| POST   | `/api/verify`         | Xác thực: so sánh ảnh với encoding |
| POST   | `/api/anti-spoof`     | Kiểm tra liveness (mặt thật/giả)   |

---

## 7. Tham số cấu hình

| Tham số           | Giá trị mặc định | Ý nghĩa                                      |
|-------------------|------------------|----------------------------------------------|
| `DET_THRESH`      | 0.6              | Ngưỡng confidence cho detection              |
| `VERIFY_THRESHOLD`| 0.7              | Ngưỡng cosine similarity để coi là “trùng”   |
| `ANTI_SPOOF_THRESHOLD` | 0.55      | Ngưỡng xác suất mặt thật                     |
| `MIN_FACE_SIZE`   | 120              | Cạnh ngắn nhất (px) của mặt chấp nhận       |
| `BLUR_VAR_THR`    | 250.0            | Ngưỡng Laplacian variance (dưới = mờ)        |

---

## 8. Cấu trúc thư mục `trained_models/`

```
trained_models/
├── detection/
│   └── det_10g.onnx           # SCRFD – phát hiện mặt
├── recognition/
│   └── w600k_r50.onnx         # ArcFace – embedding
└── face_anti_spoofing/
    └── weights/
        └── antispoof_80x80.onnx  # MiniFASNet – liveness
```

Copy từ repo `checkin_face_anti_spoofing` hoặc tải từ nguồn tương thích.

---

## 9. Cài đặt và chạy

```bash
# Virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Cài đặt
pip install -r requirements.txt

# Chạy
python main.py
# hoặc: uvicorn main:app --host 0.0.0.0 --port 8110 --reload
```

Service chạy tại: `http://localhost:8110`

---

## 10. Tài liệu tham khảo (cho NCKH)

- **SCRFD**: Sample and Computation Redistribution for Efficient Face Detection
- **ArcFace**: Additive Angular Margin Loss for Deep Face Recognition
- **MiniFASNet**: MiniFASNet: Minimal Face Anti-Spoofing Network (Minivision)
- **InsightFace**: https://github.com/deepinsight/insightface
- **ONNX Runtime**: https://onnxruntime.ai/

---

## 11. Tóm tắt cho NCKH

Hệ thống nhận diện khuôn mặt gồm:

1. **Phát hiện và căn chỉnh**: SCRFD + 5 landmarks → ảnh 112×112
2. **Trích xuất đặc trưng**: ArcFace ResNet50 → vector 512D L2-normalized
3. **Xác thực**: Cosine similarity giữa embedding mới và embedding đã lưu
4. **Chống giả mạo**: MiniFASNet + quality gates (kích thước, blur) → loại ảnh in, video replay

Các biện pháp chống lừa đảo:

- Mô hình liveness (Real/Print/Replay)
- Kiểm tra kích thước mặt tối thiểu
- Kiểm tra độ nét (Laplacian variance)
- Face enhancement trước khi đánh giá
