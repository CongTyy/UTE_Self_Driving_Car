                                                # UTE_Self_Driving_Car

Hướng dẫn sử dụng:

1. Chuẩn bị dataset với định dạng:
      ```angular2html
      Folder dataset
            |
            |--> folder ảnh      
            |      |--> abc.png/.jpg
            |      |--> xyz.png/ .jpg
            |--> folder label
                  |--> abc.png
                  |--> xyz.png

2. Sử dụng file train_UNET.ipynb để train mô hình trên google colab hoặc sử dụng file train.py nếu train trên máy local. 
3. Lưu ý: 
      1. Chỉnh sửa lại đường dẫn dataset trong các file train_UNET.ipynb và train.py.
      2. Có 2 tập dữ liệu train và test, 2 tập này phải khác nhau.
      3. file .pth là file output của mô hình khi huấn luyện. File này sử dụng khi test và inference.
      4. Dựa vào mIoU để kiểm tra độ chính xác của mô hình ( mIoU càng cao càng tốt)

  
