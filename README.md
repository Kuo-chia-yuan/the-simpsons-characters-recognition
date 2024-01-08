# the-simpsons-characters-recognition

## 介紹
影像辨識 "辛普森家庭" 50種卡通角色
 - input：一張角色圖片
 - output：正確角色名稱

## 模型架構
- 主架構：2層 CNN + MaxPool + 1層 FC
- 激勵函數：ReLU
```
class CNNModel(nn.Module):
    def __init__(self, num_classes=50):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
```

## 參數設定
- epoch = 2
- leanring rate = 0.01
- batch size = 32
- validation_size = 0.2
- loss：CrossEntropyLoss
- optimizer：SGD

## 額外優化
- 改變 train data，讓它們產生與 test data 類似的10種隨機噪音，以模擬各種圖片的扭曲、變形、縮放等雜音效果
```
transform = T.Compose([
    T.Resize((224, 224)), 
    T.ToTensor(), 
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  

    T.RandomApply([T.RandomHorizontalFlip()], p=0.1),
    T.RandomApply([T.RandomVerticalFlip()], p=0.1),
    T.RandomApply([T.RandomRotation(10)], p=0.1),

    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
    T.RandomApply([T.RandomSolarize(threshold=1.0)], p=0.05),
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),

    T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
    T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
    T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.1),
    T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),
])
```

## 輸出結果
- loss = 0.7
- train data acc = 86%
- valid data acc = 56%
- test data acc = 35%

## training loss 曲線圖
- 附圖是訓練兩次 epoch 的 loss 曲線圖

  ![image](https://github.com/Kuo-chia-yuan/the-simpsons-characters-recognition/assets/56677419/9275d8af-2964-4fd2-a7ce-a831b4e7c57c)

## CNN 第一層 16 個 3*3 kernels 權重

![image](https://github.com/Kuo-chia-yuan/the-simpsons-characters-recognition/assets/56677419/f4514fed-5f0b-495b-99e3-25362d1f25ac)

- 由圖可知，有明確抓出特徵色塊

## Confusion Matrix

![image](https://github.com/Kuo-chia-yuan/the-simpsons-characters-recognition/assets/56677419/bb224e9d-e527-4b47-8401-c942966528cc)

- 由圖可知，第10種、第13種、第32種人物是最容易和其他人物搞混的
- 又第12種、第16種、第17種、第20種、第25種人物是辨識度最準確的

## 遇到困難 & 解決方法
1. train data 和 test data 檔案過大，無法直接上傳至 colab：
   - kaggle 有提供 Token，在 colab 打以下指令便能快速下載 data 及解壓縮
   ```
    !mkdir -p ~/.kaggle
    !cp /content/kaggle.json ~/.kaggle/
    !chmod 600 /root/.kaggle/kaggle.json
    !kaggle competitions download -c machine-learning-2023nycu-classification
    !unzip machine-learning-2023nycu-classification.zip
   ```
3. learning rate 設定不當，容易產生梯度爆炸，loss 下降又上升：
   - 不斷嘗試各種 lr，包括 0.1 ~ 0.0001，最後發現 lr = 0.01 能讓模型於前期迅速收斂，後期也不會產生梯度爆炸
5. batch size 設定太大(128以上)，導致 loss 無法迅速下降：
   - 原本設定 256，發現 train data acc 只能到 20% 左右，經過不斷調整大小後，最後發現小批次(batch size = 32)能讓模型能迅速收斂，並有利於逃出 local minimal
7. CNN 層數太多(5層以上)，導致 loss 完全沒有下降的趨勢：
   - 經過測試，CNN 5層時，train data acc = 6%，當 CNN 下降至 2層時，train data acc = 86%，準確度明顯提升非常多
9. optimizer 採用 Adam 的話，loss 會下降得非常緩慢，且後期容易產生梯度爆炸：
   - 改採用 SGD，雖然 Adam 會隨著訓練深度自動調整 lr，但 SGD 對於 data 噪音更加敏感，因為我的 data 相對較小，所以我只要將 lr 設定妥當，準確度皆會比使用 Adam 高
11. train data acc 和 validation data acc 有些差距，和 test data acc 差距則更大：
    - 我認為這是 overfitting 的徵兆，因此我增加 train data 多樣性，包括對圖片旋轉、縮放、調整銳利度、調整透明度等
12. 一次 epoch 的時間很久，將近2小時，colab 有時會自動斷線，訓練便會終止，非常浪費時間：
    - 在 colab 網頁的 HTML 介面的 Console 輸入以下指令，便能固定時間自動點擊頁面，讓 colab 偵測到該頁面並非無人使用，便不會輕易終止程式的運行
    ```
    function ConnectButton(){
        console.log("Connect pushed"); 
        document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
    }
    setInterval(ConnectButton,60000);
    ```
