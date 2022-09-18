# 如何用人工智能自动玩游戏
## 一、前言

让AI玩游戏的思想早在上世纪就已经有了，那个时候更偏向棋类游戏。像是五子棋、象棋等。在上世纪“深蓝”就击败了国际象棋冠军，而到2016年“Alpha Go”击败了人类围棋冠军。

到现在，AI涉略的不仅仅是棋类游戏。像是超级马里奥、王者荣耀这种游戏，AI也能有比较好的表现。今天我们就来用一个实际的例子讨论AI自动玩游戏这一话题，本文会用非常简单的机器学习算法让AI自动玩Google小恐龙游戏。

## 二、Google小恐龙与监督学习
### 2.1、Google小恐龙

如果你使用的是Chrome浏览器，那么相信你应该见过下面这个恐龙：

![在这里插入图片描述](https://img-blog.csdnimg.cn/f6cd242ae82248759165688539d6c521.png#pic_center)
当我们用Chrome断网访问网页时，就会显示这个恐龙，或者直接在地址栏输入：[chrome://dino](chrome://dino)直接访问该游戏。

游戏的玩法非常简单，只需要按空格键即可。比如下面左图，快碰到障碍物，这时需要按空格，而下面右图没有障碍（或离障碍比较远），则不需要按按键。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2ba95fbf96e84d0aa3bc49fe1fb9fc9b.png#pic_center)
当然还有出现鸟的情况，我们也可以归为跳的情况。大家可以玩一下。
### 2.2、监督学习
玩游戏很多时候会使用一个叫强化学习的方式来实现，而本文使用比较简单的监督学习来实现。

本文会使用逻辑回归算法实现，其代码如下：
```python
from sklearn.linear_model import LogisticRegression # 逻辑回归模型
from sklearn.model_selection import train_test_split    # 数据集拆分
# 1、准备数据
X = [
    # 天河区的坐标
    [1, 1],
    [1, 2],
    [2, 0],
    [3, 2],
    [3, 3],
    # 花都区的坐标
    [7, 7],
    [6, 7],
    [7, 6],
    [8, 6],
    [8, 5]
]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
# 2、拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 3、定义模型
model = LogisticRegression()
# 4、填充数据并训练
model.fit(X_train, y_train)
# 5、评估模型
score1 = model.score(X_train, y_train)
score2 = model.score(X_test, y_test)
print(score1, score2)
# 6、预测
input = [
    [4, 4]
]
pred = model.predict(input)
print(pred)
```

关于逻辑回归的讲解可以查看：[Python快速构建神经网络
](https://blog.csdn.net/ZackSock/article/details/115292795)。

我们可以把玩游戏看作一个分类问题，即输入为当前游戏的图像，输出为0、1的一个二分类问题（0表示跳，1表示不跳）。要让AI实现自动玩游戏，我们需要做几件事情。分别如下：
1. 玩游戏，收集一些需要跳的图片和一些不需要条的图片
2. 选择合适的分类算法，训练一个模型
3. 截取当前游戏画面，预测结果，判断是否需要跳跃
4. 如果需要跳跃，则用程序控制键盘，按下跳跃键

下面我们来依次完成上面的事情。

## 三、收集数据
收集数据我们需要在玩游戏的过程中不停地截图，这里可以用`Pillow`模块来实现截图。`Pillow`模块需要单独安装，安装语句如下：
```python
pip install pillow
```
截图的代码如下：
```python
import time
from PIL import ImageGrab   # 截图
time.sleep(3)
while True:
    # 截图
    img = ImageGrab.grab()
    # print(img.size) # 960 540 480 270
    img = img.resize((960, 540))
    # 保存图片
    img.save(f'imgs/{str(time.time())}.jpg')
    # 修改name
    time.sleep(0.1)
```
运行程序后就可以切换到Chrome开始游戏了。进行一段时间后，我们会截取一些图片，大致如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/199fdf51b92742fe9d0afcca1df497c0.png#pic_center)
这时就轮到人类智能上场了，我们手动的把我们决定需要跳的场景放置到`imgs/jump`目录下，把觉得不需要跳的场景放到`imgs/none`目录下。然后就可以进行下一步了，这里截取的图片通常不需要跳的要多很多，所有可以多收集几次。

收集完成后我们就可以把图片读入，并转换成一个1维数组，这部分代码如下：
```python
import os
import cv2
# 所有图片的全路径
files = [os.path.join(jump_path, jump) for jump in os.listdir(jump_path)] + \
        [os.path.join(none_path, none) for none in os.listdir(none_path)]
X = []
y = [0] * len(os.listdir(jump_path)) + [1] * len(os.listdir(none_path))
# 遍历jump目录下的图片
for idx, file in enumerate(files):
    filepath = os.path.join(none_path, file)
    x = cv2.imread(filepath, 0).reshape(-1)
    X.append(x)
```
此时`X`和`y`就是我们的特征和目标了。有了`X`和`y`就可以开始训练模型了。
## 四、训练分类模型
训练部分的代码非常简单，我们可以在训练完成后保存模型。代码如下：
```python
import os
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
jump_path = os.path.join('imgs', 'jump')    # 需要跳的图片的根目录
none_path = os.path.join('imgs', 'none')    # 不需要跳的图片的根目录
# 所有图片的全路径
files = [os.path.join(jump_path, jump) for jump in os.listdir(jump_path)] + \
        [os.path.join(none_path, none) for none in os.listdir(none_path)]
X = []
y = [0] * len(os.listdir(jump_path)) + [1] * len(os.listdir(none_path))
# 遍历jump目录下的图片
for file in files:
    x = cv2.imread(file, 0).reshape(-1)
    X.append(x)

# 2、拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 3、定义模型
model = LogisticRegression(max_iter=500)
# 4、训练模型
model.fit(X_train, y_train)
# 5、评估模型
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(train_score, test_score)
# 保存模型
joblib.dump(model, 'auto_play.m')
```
在我电脑上训练的准确率在90%以上，总体效果还是不错的。不过有几个可以改进的地方。这里说几点：
1. 图像只有中间部分会对下一步操作有影响，因此可以选择对训练图片进行一些处理。把上面和下面部分设置为0。如果做了这个处理，那么在实际应用时也要做同样的处理。
2. 这些图片如果移植到其它电脑可能不适用，因为分辨率等原因。所有可以选择使用更复杂的模型，比如CNN网络。
3. 因为手动收集数据比较麻烦，可以选择做一下数据增强。

在这里我们不做这些改进，直接使用最简单的模型。

## 五、自动玩游戏
自动玩游戏需要借助pynput模块来实现，其安装如下：
```python
pip install pynput
```
我们可以用下面的代码实现按下键盘的空格键：
```python
from pynput import keyboard
from pynput.keyboard import Key
# 创建键盘
kb = keyboard.Controller()
# 按下空格键
kb.press(Key.space)
```
知道了如何控制键盘后，我们就可以使用模型截取预测，如何判断是否要按空格，代码如下：
```python
import time
import cv2
import joblib
import numpy as np
from PIL import ImageGrab
from pynput import keyboard
from pynput.keyboard import Key

time.sleep(3)
# 0、创建键盘
kb = keyboard.Controller()
# 1、加载模型
model = joblib.load('auto_play.m')
while True:
    # 2、准备数据
    ImageGrab.grab().resize((960, 540)).save('current.jpg')  # 保存当前屏幕截屏
    x = cv2.imread('current.jpg', 0).reshape(-1)
    x = [x]
    # 3、预测
    pred = model.predict(x)
    print(pred)
    # 如果需要跳，则按下空格
    if pred[0] == 0:
        kb.press(Key.space)
```
运行上面的程序后，打开浏览器即可开始游戏。程序的代码和图片文件：[https://download.csdn.net/download/ZackSock/86543410](https://download.csdn.net/download/ZackSock/86543410)
GitHub地址为：[https://github.com/IronSpiderMan/AutoPlayGoogleDino](https://github.com/IronSpiderMan/AutoPlayGoogleDino)
