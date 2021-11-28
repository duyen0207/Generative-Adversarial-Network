# Generative-Adversarial-Network
## Ý tưởng chính
Huấn luyện đồng thời hai mô hình
* Generator G: Học cách sinh dữ liệu giống thật nhất để đánh lừa discriminator
* Discriminator D: đánh giá và phân biệt đâu là dữ liệu thực và đâu là dữ liệu giả được tạo ra bởi generator

Quá trình huấn luyện sẽ diễn ra cho đến khi Discriminator D không thể phân biệt được thật giả, xác suất D bằng 1/2 ở mọi nơi.
![GAN] (./assets/gan_pipeline.png)


## Quá trình cài đặt
### Chuẩn bị dữ liệu training
Sử dụng tập dữ liệu MNIST hoặc Fashion MNIST cho huấn luyện mô hình.
Trước khi đưa vào training dữ liệu ảnh được transform sang dạng float tensor.
### Discriminator
* Xây dựng tất cả lớp ẩn là Linear layer
* Sử dụng kĩ thuật Dropout với hệ số 0.3 để bỏ qua một số neural giúp tăng khả năng học của mô hình và tránh overfitting
* Dữ liệu hình ảnh x đã được flatten đem forward qua các layer
* Sử dụng Leaky ReLU làm hàm kích hoạt, trừ lớp cuối cùng không sử dụng bất kì hàm kích hoạt nào.

Discriminator khi có input đầu vào sẽ cho kết quả là 0 hoặc 1 cho biết hình ảnh là thật hay giả.
### Generator
Tương tự khi xây dựng Discriminator nhưng layer cuối sử dụng hàm tanh làm hàm kích hoạt để tỉ lệ output nằm trong khoảng [-1,1]

## Quá trình huấn luyện
### Hàm loss của Discriminator và Generator


#### Discriminator
Mục tiêu: mong muốn D(real_images) = 1 và D(fake_images) = 0
* Tổng loss của Discriminator là `d_loss = d_real_loss + d_fake_loss`. 
* Sử dụng BCEWithLoss (kết hợp Binary cross entropy và kích hoạt sigmoid)
* Làm mịn nhãn nếu tham số smooth=true bằng việc giảm từ 1.0 xuống 0.9.


#### Generator

Mục tiêu: làm cho D(fake_image) = 1
Các nhãn được lật thể hiện generator đang cố gắng đánh lừa discriminator với ảnh mà nó tạo ra

Huấn luyện xen kẽ k lần Discriminator và 1 lần Generator. Ở đây k = 1.

### Huấn luyện Discriminator
1. Tính loss của discriminator trên ảnh thật : d_real_loss      
2. Tạo ảnh giả bằng cách truyền một nhiễu z ngẫu nhiên qua Generator
3. Tính loss của discriminator trên ảnh được tạo bởi Generator: d_fake_loss 
4. Cộng d_real_loss và d_fake_loss
5. Sử dụng backpropagation và một bước tối ưu hóa để cập nhật trọng số của discriminator

### Huấn luyện Generator
1. Tạo ảnh giả bằng cách truyền một nhiễu z ngẫu nhiên qua Generator
2. Tính loss của discriminator trên ảnh giả được tạo bởi Generator, sử dụng các nhãn **flipped**
3. Sử dụng backpropagation và một bước tối ưu hóa để cập nhật trọng số của generator

Sau mỗi bước huấn luyện, loss được lưu lại cho mục đích thống kê

## Tham khảo
GAN tutorial: https://github.com/udacity/deep-learning-v2-pytorch
Paper Generative Adversarial Nets - Ian J. Goodfellow at al: https://arxiv.org/pdf/1406.2661.pdf
