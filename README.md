
## Automated generation of adaptive perturbed images based on GAN for motivated adversaries on deep learning models.
 
#    Duy Trung Pham
    Department of Information Security -
    Academy of Cryptography Techniques
    Ha Noi, Vietnam trungpd@actvn.edu.vn

#    Cong Thanh Nguyen
    Department of Information Security -
    Academy of Cryptography Techniques
    Ho Chi Minh, Vietnam thanhnc1212@gmail.com
  
#    Phi Ho Truong∗
    Department of Information Security -
    Academy of Cryptography Techniques
    Ha Noi, Vietnam hotp.gvm@actvn.edu.vn

#    Nhat Hai Nguyen
    School of Information and Communication Technology -
    Hanoi University of Science and Technology
    Ha Noi, Vietnam hainn@soict.hust.edu.vn

# Install
  - Vui lòng cài đặt biến môi trường theo requirements.txt
# Using & train model
  - cifar_main.py
  - AIGAN.py (Tham khảo)
# General image fakes & Evaluate model
  - test_adversarial_examples-cifar10.py
  - Model Target Evaluate
    - resnes56

    ![alt text](https://github.com/congthanh96/Deepfake_Adversarial/blob/main/test/resnes56.PNG)

    - mobilenetv2_x1_4

    ![alt text](https://github.com/congthanh96/Deepfake_Adversarial/blob/main/test/mobilenetv2_x1_4.PNG)

    - repvgg_a2

    ![alt text](https://github.com/congthanh96/Deepfake_Adversarial/blob/main/test/repvgg_a2.PNG)

    - shufflenetv2

    ![alt text](https://github.com/congthanh96/Deepfake_Adversarial/blob/main/test/shufflenetv2.PNG)

    - vgg19bn

    ![alt text](https://github.com/congthanh96/Deepfake_Adversarial/blob/main/test/vgg19bn.PNG)


# Test & Evaluate
  - usingresnes56.ipynb
  - usingmobilenet.ipynb
  - usingrevgg_a2.ipynb
  - usingshufflenetv2.ipynb
  - usingvgg19bn.ipynb
